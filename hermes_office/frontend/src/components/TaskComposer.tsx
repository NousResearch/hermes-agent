import React, { useState } from "react";
import { useStore } from "../state";
import { api } from "../api";
import { t } from "../i18n";

export function TaskComposer() {
  const employees = useStore((s) => s.employees);
  const departments = useStore((s) => s.departments);
  const selectedDeptId = useStore((s) => s.selectedDeptId);

  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);

  const send = async () => {
    if (!text.trim()) return;
    setBusy(true);
    try {
      // @name → employee, @dept → department, otherwise the currently
      // selected department (or first department) catches the task.
      const m = text.match(/@([^\s]+)/);
      let body: { text: string; employee_id?: string; department_id?: string } = { text };
      if (m) {
        const tag = m[1].toLowerCase();
        const emp = employees.find((e) => e.name.toLowerCase() === tag);
        const dept = departments.find((d) => d.name.toLowerCase() === tag);
        if (emp) body.employee_id = emp.id;
        else if (dept) body.department_id = dept.id;
      }
      if (!body.employee_id && !body.department_id) {
        body.department_id = selectedDeptId ?? departments[0]?.id;
      }
      if (!body.employee_id && !body.department_id) {
        alert("Create a department or hire someone first.");
        return;
      }
      await api.createTask(body);
      setText("");
    } catch (e) {
      alert(`task failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="p-3 border-t border-slate-200 bg-white/80 backdrop-blur-sm">
      <div className="flex items-center gap-2">
        <input
          aria-label="task"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          placeholder={t("composerPlaceholder")}
          className="flex-1 rounded-2xl border border-slate-200 px-4 py-3 text-sm focus:border-sky2 focus:outline-none focus:ring-2 focus:ring-sky-200"
          disabled={busy}
        />
        <button onClick={send} disabled={busy || !text.trim()} className="btn-primary">
          {busy ? "…" : t("send")}
        </button>
      </div>
    </div>
  );
}
