import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def load_data():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    return data

def preprocess_data(data):
    data['sepal_petal_ratio'] = data['sepal length (cm)'] / data['petal length (cm)']
    scaler = StandardScaler()
    features = data.drop(columns=['target', 'sepal_petal_ratio'])
    data[features.columns] = scaler.fit_transform(features)
    data['target'] = data['target'].astype(int)
    return data

if __name__ == "__main__":
    data = load_data()
    data = preprocess_data(data)
    data.to_csv("cleaned_data.csv", index=False)
    print("Preprocessed data saved as 'cleaned_data.csv'.")