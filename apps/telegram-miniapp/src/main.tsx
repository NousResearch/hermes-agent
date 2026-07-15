import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./fonts/fonts.css";
import "./styles.css";

createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
