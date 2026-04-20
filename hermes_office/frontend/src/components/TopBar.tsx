import React from "react";
import { useStore } from "../state";
import { CapacityBadge } from "./CapacityBadge";
import { LangSwitcher } from "./LangSwitcher";
import { t } from "../i18n";

interface Props {
  onHire: () => void;
  onAddDept: () => void;
}

export function TopBar({ onHire, onAddDept }: Props) {
  const employees = useStore((s) => s.employees);
  const departments = useStore((s) => s.departments);

  return (
    <header className="flex items-center justify-between px-5 py-3 bg-white/80 backdrop-blur-md border-b border-slate-200 shadow-soft z-20">
      <div className="flex items-center gap-3">
        <div
          aria-hidden
          className="w-9 h-9 rounded-xl bg-gradient-to-br from-sky2 to-violet-500 grid place-items-center shadow-soft"
        >
          <span className="text-white font-bold text-lg">H</span>
        </div>
        <div>
          <div className="font-display font-bold text-lg leading-none">{t("appTitle")}</div>
          <div className="text-xs text-slate-500">
            {employees.length} {t("employees")} · {departments.length} dept
          </div>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <CapacityBadge />
        <LangSwitcher />
        <button className="btn-ghost" onClick={onAddDept} aria-label={t("addDeptBtn")}>
          <span aria-hidden>🏛️</span> {t("addDeptBtn")}
        </button>
        <button
          className="btn-primary !bg-emerald-500 hover:!bg-emerald-600 !text-white"
          onClick={onHire}
          aria-label={t("hireBtn")}
        >
          <span aria-hidden>＋</span> {t("hireBtn")}
        </button>
      </div>
    </header>
  );
}
