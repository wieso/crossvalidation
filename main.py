import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics, svm, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize

pd.set_option('display.width', 256)


# https://www.dataquest.io/course/kaggle-competitions
#
# PassengerId -- A numerical id assigned to each passenger.
# Survived -- Whether the passenger survived (1), or didn't (0). We'll be making predictions for this column.
# Pclass -- The class the passenger was in -- first class (1), second class (2), or third class (3).
# Name -- the name of the passenger.
# Sex -- The gender of the passenger -- male or female.
# Age -- The age of the passenger. Fractional.
# SibSp -- The number of siblings and spouses the passenger had on board.
# Parch -- The number of parents and children the passenger had on board.
# Ticket -- The ticket number of the passenger.
# Fare -- How much the passenger paid for the ticker.
# Cabin -- Which cabin the passenger was in.
# Embarked -- Where the passenger boarded the Titanic.


class DataDigest:

    def __init__(self):
        self.ages = None
        self.fares = None
        self.titles = None
        self.cabins = None
        self.families = None
        self.tickets = None


def get_title(name):
    if pd.isnull(name):
        return "Null"

    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1).lower()
    else:
        return "None"


def get_family(row):
    last_name = row["Name"].split(",")[0]
    if last_name:
        family_size = 1 + row["Parch"] + row["SibSp"]
        if family_size > 3:
            return "{0}_{1}".format(last_name.lower(), family_size)
        else:
            return "nofamily"
    else:
        return "unknown"


def get_index(item, index):
    if pd.isnull(item):
        return -1

    try:
        return index.get_loc(item)
    except KeyError:
        return -1


def munge_data(data, digest):
    # Age
    data["AgeF"] = data.apply(lambda r: digest.ages[r["Sex"]] if pd.isnull(r["Age"]) else r["Age"], axis=1)

    # Fare
    data["FareF"] = data.apply(lambda r: digest.fares[r["Pclass"]] if pd.isnull(r["Fare"]) else r["Fare"], axis=1)

    # Gender
    genders = {"male": 1, "female": 0}
    data["SexF"] = data["Sex"].apply(lambda s: genders.get(s))

    gender_dummies = pd.get_dummies(data["Sex"], prefix="SexD", dummy_na=False)
    data = pd.concat([data, gender_dummies], axis=1)

    # Embarkment
    embarkments = {"U": 0, "S": 1, "C": 2, "Q": 3}
    data["EmbarkedF"] = data["Embarked"].fillna("U").apply(lambda e: embarkments.get(e))

    embarkment_dummies = pd.get_dummies(data["Embarked"], prefix="EmbarkedD", dummy_na=False)
    data = pd.concat([data, embarkment_dummies], axis=1)

    # Relatives
    data["RelativesF"] = data["Parch"] + data["SibSp"]
    data["SingleF"] = data["RelativesF"].apply(lambda r: 1 if r == 0 else 0)

    # Deck
    decks = {"U": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
    data["DeckF"] = data["Cabin"].fillna("U").apply(lambda c: decks.get(c[0], -1))

    deck_dummies = pd.get_dummies(data["Cabin"].fillna("U").apply(lambda c: c[0]), prefix="DeckD", dummy_na=False)
    data = pd.concat([data, deck_dummies], axis=1)

    # Titles
    title_dummies = pd.get_dummies(data["Name"].apply(lambda n: get_title(n)), prefix="TitleD", dummy_na=False)
    data = pd.concat([data, title_dummies], axis=1)

    # Lookups
    data["CabinF"] = data["Cabin"].fillna("unknown").apply(lambda c: get_index(c, digest.cabins))

    data["TitleF"] = data["Name"].apply(lambda n: get_index(get_title(n), digest.titles))

    data["TicketF"] = data["Ticket"].apply(lambda t: get_index(t, digest.tickets))

    data["FamilyF"] = data.apply(lambda r: get_index(get_family(r), digest.families), axis=1)

    # Stat
    age_bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]
    data["AgeR"] = pd.cut(data["Age"].fillna(-1), bins=age_bins).astype(object)

    return data


def linear_scorer(estimator, x, y):
    scorer_predictions = estimator.predict(x)

    scorer_predictions[scorer_predictions > 0.5] = 1
    scorer_predictions[scorer_predictions <= 0.5] = 0

    return metrics.accuracy_score(y, scorer_predictions)


# -----------------------------------------------------------------------------
# load
# -----------------------------------------------------------------------------

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
all_data = pd.concat([train_data, test_data])

# -----------------------------------------------------------------------------
# stat
# -----------------------------------------------------------------------------

# print("===== survived by class and sex")
# print(train_data.groupby(["Pclass", "Sex"])["Survived"].value_counts(normalize=True))

# -----------------------------------------------------------------------------
# describe
# -----------------------------------------------------------------------------

describe_fields = ["Age", "Fare", "Pclass", "SibSp", "Parch"]
#
# print("===== train: males")
# print(train_data[train_data["Sex"] == "male"][describe_fields].describe())
#
# print("===== test: males")
# print(test_data[test_data["Sex"] == "male"][describe_fields].describe())
#
# print("===== train: females")
# print(train_data[train_data["Sex"] == "female"][describe_fields].describe())
#
# print("===== test: females")
# print(test_data[test_data["Sex"] == "female"][describe_fields].describe())

# -----------------------------------------------------------------------------
# munge
# -----------------------------------------------------------------------------

data_digest = DataDigest()

data_digest.ages = all_data.groupby("Sex")["Age"].median()
data_digest.fares = all_data.groupby("Pclass")["Fare"].median()

titles_trn = pd.Index(train_data["Name"].apply(get_title).unique())
titles_tst = pd.Index(test_data["Name"].apply(get_title).unique())
data_digest.titles = titles_tst

families_trn = pd.Index(train_data.apply(get_family, axis=1).unique())
families_tst = pd.Index(test_data.apply(get_family, axis=1).unique())
data_digest.families = families_tst

cabins_trn = pd.Index(train_data["Cabin"].fillna("unknown").unique())
cabins_tst = pd.Index(test_data["Cabin"].fillna("unknown").unique())
data_digest.cabins = cabins_tst

tickets_trn = pd.Index(train_data["Ticket"].fillna("unknown").unique())
tickets_tst = pd.Index(test_data["Ticket"].fillna("unknown").unique())
data_digest.tickets = tickets_tst

train_data_munged = munge_data(train_data, data_digest)
test_data_munged = munge_data(test_data, data_digest)
all_data_munged = pd.concat([train_data_munged, test_data_munged])

predictors = ["Pclass",
              "AgeF",
              "TitleF",
              "TitleD_mr", "TitleD_mrs", "TitleD_miss", "TitleD_master", "TitleD_ms",
              "TitleD_col", "TitleD_rev", "TitleD_dr",
              "CabinF",
              "DeckF",
              "DeckD_U", "DeckD_A", "DeckD_B", "DeckD_C", "DeckD_D", "DeckD_E", "DeckD_F", "DeckD_G",
              "FamilyF",
              "TicketF",
              "SexF",
              "SexD_male", "SexD_female",
              "EmbarkedF",
              "EmbarkedD_S", "EmbarkedD_C", "EmbarkedD_Q",
              "FareF",
              "SibSp", "Parch",
              "RelativesF",
              "SingleF"]

datasets = [
    {
        'Name': 'Titanic',
        'X': train_data_munged[predictors],
        'Y': train_data_munged['Survived']
    },
    {
        'Name': 'Iris',
        'X': datasets.load_iris().data,
        'Y': datasets.load_iris().target
    },
    {
        'Name': 'Wine',
        'X': datasets.load_wine().data,
        'Y': datasets.load_wine().target
    },
    {
        'Name': 'Breast cancer',
        'X': datasets.load_breast_cancer().data,
        'Y': datasets.load_breast_cancer().target
    },
    {
        'Name': 'Digits',
        'X': datasets.load_digits().data,
        'Y': datasets.load_digits().target
    }
]

for dataset in datasets:
    print(dataset['Name'])
    print('=' * 40)

    X, X_valid, y, y_valid = train_test_split(dataset['X'], dataset['Y'],
                                              test_size=0.3)
    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    scaler.fit(dataset['X'])

    X = scaler.transform(X)
    X_valid = scaler.transform(X_valid)

    clf_list = [
        RandomForestClassifier(random_state=1, max_depth=15, n_estimators=500, min_samples_split=8, min_samples_leaf=2),
        svm.SVC(probability=True),
        KNeighborsClassifier(n_neighbors=3),
    ]

    for clf in clf_list:
        print(clf.__class__)
        n_splits = 5
        kf = KFold(n_splits=n_splits)
        scores_sum = 0
        i = 1
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = metrics.accuracy_score(y_test, y_pred)
            scores_sum += score

            y_score = clf.predict_proba(X_valid)
            y_lb_valid = label_binarize(y_valid, classes=range(len(np.unique(dataset['Y']))))
            n_classes = y_lb_valid.shape[1]
            if n_classes == 1:
                fpr, tpr, _ = roc_curve(y_valid, y_score[:, 1])
            else:
                fpr, tpr, _ = roc_curve(y_lb_valid.ravel(), y_score.ravel())

            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        y_pred = clf.predict(X_valid)
        score = metrics.accuracy_score(y_valid, y_pred)
        print('K-Fold score', scores_sum / n_splits)
        print('Valid score', score)

        plt.title(dataset['Name'] + str(clf.__class__))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend()
        plt.show()

    print('=' * 40)
