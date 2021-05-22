from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn.externals import joblib
app = Flask(__name__)


# Machine Learning code goes here
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("data/Youtube01-Psy.csv")
    df_data = df[['CONTENT', 'CLASS']]
    # Features and Labels
    df_x = df_data['CONTENT']
    df_y = df_data.CLASS
    # Extract the features with countVectorizer
    corpus = df_x
    cv = TfidfVectorizer()
    X1 = cv.fit_transform(corpus)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X1, df_y, test_size=0.2, random_state=2)
    # LOGISTIC REGRESSION
    clf=LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5555, debug=True)
