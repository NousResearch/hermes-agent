import React, { useEffect, useMemo, useState } from "react";
import { useStore } from "../state";
import type { ActivityEvent } from "../types";
import { t } from "../i18n";

const ICONS: Record<ActivityEvent["kind"], string> = {
  state_change: "🔄",
  tool_call: "🔧",
  tool_result: "📦",
  assistant: "💬",
  clarify: "❓",
  error: "⚠️",
};

export function ActivityRail() {
  const map = useStore((s) => s.activityByEmp);
  const employees = useStore((s) => s.employees);
  const [, force] = useState(0);

  // re-render once a second so timestamps tick
  useEffect(() => {
    const id = setInterval(() => force((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const items = useMemo(() => {
    const all: ActivityEvent[] = [];
    Object.values(map).forEach((list) => all.push(...list));
    all.sort((a, b) => new Date(b.ts).getTime() - new Date(a.ts).getTime());
    return all.slice(0, 80);
  }, [map]);

  const nameOf = (id: string) => employees.find((e) => e.id === id)?.name ?? id.slice(0, 6);

  return (
    <aside className="w-80 shrink-0 border-l border-slate-200 bg-white/70 backdrop-blur-sm overflow-y-auto">
      <div className="p-3 text-xs font-semibold uppercase tracking-wider text-slate-500 sticky top-0 bg-white/80 backdrop-blur z-10">
        {t("activity")}
      </div>
      <ul className="px-3 pb-4 space-y-2">
        {items.length === 0 && (
          <li className="text-sm text-slate-400 italic">no activity yet</li>
        )}
        {items.map((e, i) => (
          <li key={i} className={`text-sm border-l-4 pl-3 py-1 ${
            e.kind === "error" ? "border-rose-500"
              : e.kind === "tool_call" ? "border-amber-500"
              : e.kind === "assistant" ? "border-emerald-500"
              : "border-slate-300"
          }`}>
            <div className="flex items-baseline gap-2">
              <span aria-hidden>{ICONS[e.kind]}</span>
              <span className="font-medium">{nameOf(e.employee_id)}</span>
              <span className="text-xs text-slate-400 ml-auto">
                {new Date(e.ts).toLocaleTimeString()}
              </span>
            </div>
            <div className="text-slate-700 break-words">{e.text}</div>
          </li>
        ))}
      </ul>
    </aside>
  );
}
