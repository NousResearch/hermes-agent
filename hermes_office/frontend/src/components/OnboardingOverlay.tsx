import React from "react";
import { t } from "../i18n";

interface Props {
  onHire: () => void;
}

export function OnboardingOverlay({ onHire }: Props) {
  return (
    <div className="absolute inset-0 grid place-items-center bg-white/60 backdrop-blur-sm pointer-events-none">
      <div className="card p-8 max-w-md text-center pointer-events-auto">
        <div className="text-6xl mb-4" aria-hidden>🏢</div>
        <h2 className="font-display text-2xl font-bold mb-2">{t("welcomeTitle")}</h2>
        <p className="text-slate-600 mb-4">{t("welcomeBody")}</p>
        <button onClick={onHire} className="btn-primary !bg-emerald-500 hover:!bg-emerald-600 !px-6 !py-3 text-lg">
          ＋ {t("hireBtn")}
        </button>
      </div>
    </div>
  );
}
