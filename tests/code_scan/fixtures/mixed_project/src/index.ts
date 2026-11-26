import React from "react";
import ReactDOM from "react-dom";

interface AppProps {
  title: string;
}

const App: React.FC<AppProps> = ({ title }) => {
  return (
    <div className="app">
      <h1>{title}</h1>
      <p>Welcome to the mixed project.</p>
    </div>
  );
};

const root = document.getElementById("root");
if (root) {
  ReactDOM.render(<App title="Mixed Project" />, root);
}

export default App;
