import React from "react";
import { useStore } from "../state";
import { t } from "../i18n";

interface Props {
  onPickEmployee: (id: string) => void;
}

export function Sidebar({ onPickEmployee }: Props) {
  const departments = useStore((s) => s.departments);
  const employees = useStore((s) => s.employees);
  const selectedDeptId = useStore((s) => s.selectedDeptId);
  const selectDept = useStore((s) => s.selectDept);

  return (
    <aside className="w-64 shrink-0 border-r border-slate-200 bg-white/70 backdrop-blur-sm overflow-y-auto">
      <div className="p-3 text-xs font-semibold uppercase tracking-wider text-slate-500">Departments</div>
      <ul className="px-2 space-y-1">
        <li>
          <button
            className={`w-full text-left px-3 py-2 rounded-xl transition-colors ${
              selectedDeptId == null ? "bg-sky-100 text-sky-900" : "hover:bg-slate-100"
            }`}
            onClick={() => selectDept(null)}
          >
            <span className="inline-block w-3 h-3 rounded-full mr-2 align-middle bg-slate-400" />
            All
            <span className="ml-2 text-xs text-slate-500">({employees.length})</span>
          </button>
        </li>
        {departments.map((d) => {
          const count = employees.filter((e) => e.department_id === d.id).length;
          const active = selectedDeptId === d.id;
          return (
            <li key={d.id}>
              <button
                className={`w-full text-left px-3 py-2 rounded-xl transition-colors ${
                  active ? "bg-sky-100 text-sky-900" : "hover:bg-slate-100"
                }`}
                onClick={() => selectDept(d.id)}
                title={d.mission}
              >
                <span
                  className="inline-block w-3 h-3 rounded-full mr-2 align-middle"
                  style={{ background: d.color }}
                />
                {d.name}
                <span className="ml-2 text-xs text-slate-500">({count})</span>
              </button>
            </li>
          );
        })}
      </ul>

      <div className="p-3 mt-4 text-xs font-semibold uppercase tracking-wider text-slate-500">{t("employees")}</div>
      <ul className="px-2 pb-4 space-y-1">
        {employees
          .filter((e) => selectedDeptId == null || e.department_id === selectedDeptId)
          .map((e) => (
            <li key={e.id}>
              <button
                className="w-full text-left flex items-center gap-2 px-3 py-2 rounded-xl hover:bg-slate-100"
                onClick={() => onPickEmployee(e.id)}
                title={`${e.role} · ${e.activity}`}
              >
                <span
                  className="w-7 h-7 rounded-full grid place-items-center shrink-0"
                  style={{ background: `hsl(${e.avatar.hue} 80% 65%)` }}
                  aria-hidden
                >
                  {e.name.slice(0, 1).toUpperCase()}
                </span>
                <span className="min-w-0 flex-1">
                  <div className="text-sm truncate">{e.name}</div>
                  <div className="text-xs text-slate-500 truncate">{e.role}</div>
                </span>
                <span className={`text-[10px] px-2 py-0.5 rounded-full ${
                  e.activity === "working" ? "bg-emerald-100 text-emerald-700"
                  : e.activity === "talking" ? "bg-pink-100 text-pink-700"
                  : e.activity === "learning" ? "bg-sky-100 text-sky-700"
                  : "bg-amber-100 text-amber-700"
                }`}>{e.activity}</span>
              </button>
            </li>
          ))}
        {employees.length === 0 && (
          <li className="px-3 py-2 text-xs text-slate-500 italic">no employees yet</li>
        )}
      </ul>
    </aside>
  );
}
