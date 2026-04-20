import React from "react";
import { getLang, setLang } from "../i18n";

export function LangSwitcher() {
  const lang = getLang();
  const next = lang === "zh-CN" ? "en" : "zh-CN";
  return (
    <button
      className="btn-ghost text-sm"
      title={`switch to ${next}`}
      aria-label="language"
      onClick={() => setLang(next)}
    >
      {lang === "zh-CN" ? "中" : "EN"}
    </button>
  );
}
