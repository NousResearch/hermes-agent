import React, { useState } from "react";
import { Modal } from "./Modal";
import { useStore } from "../state";
import { api } from "../api";
import { t } from "../i18n";

interface Props {
  onClose: () => void;
  onCreated: () => void;
}

const COLORS = [
  "#7c3aed", "#0ea5e9", "#10b981", "#f97316", "#ef4444",
  "#a855f7", "#06b6d4", "#84cc16", "#eab308", "#ec4899",
];

export function DepartmentManager({ onClose, onCreated }: Props) {
  const departments = useStore((s) => s.departments);
  const refresh = useStore((s) => s.refreshDepartments);

  const [name, setName] = useState("");
  const [mission, setMission] = useState("");
  const [color, setColor] = useState("#7c3aed");
  const [busy, setBusy] = useState(false);

  async function create() {
    if (!name.trim()) return;
    setBusy(true);
    try {
      await api.createDepartment({ name: name.trim(), mission: mission.trim(), color });
      onCreated();
    } catch (e) {
      alert(`failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  async function remove(id: string) {
    if (!confirm("Delete this department and its employees?")) return;
    try {
      await api.deleteDepartment(id);
      await refresh();
    } catch (e) {
      alert(`failed: ${(e as Error).message}`);
    }
  }

  return (
    <Modal onClose={onClose} title={t("addDeptBtn")}>
      <div className="space-y-4">
        <div>
          <label className="text-sm font-medium text-slate-700">{t("deptName")}</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Research"
            maxLength={50}
            className="mt-1 w-full rounded-2xl border border-slate-200 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-sky-200"
          />
        </div>
        <div>
          <label className="text-sm font-medium text-slate-700">{t("deptMission")}</label>
          <textarea
            value={mission}
            onChange={(e) => setMission(e.target.value)}
            placeholder="What does this team do?"
            maxLength={500}
            className="mt-1 w-full h-24 rounded-2xl border border-slate-200 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-sky-200"
          />
        </div>
        <div>
          <label className="text-sm font-medium text-slate-700">{t("deptColor")}</label>
          <div className="flex gap-2 mt-2 flex-wrap">
            {COLORS.map((c) => (
              <button
                key={c}
                onClick={() => setColor(c)}
                aria-label={`color ${c}`}
                className={`w-9 h-9 rounded-full border-2 ${color === c ? "border-slate-900 scale-110" : "border-slate-200"}`}
                style={{ background: c }}
              />
            ))}
          </div>
        </div>
        <div className="flex justify-end">
          <button className="btn-primary" disabled={busy || !name.trim()} onClick={create}>
            {t("deptCreate")}
          </button>
        </div>

        {departments.length > 0 && (
          <div className="pt-2 border-t border-slate-100">
            <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">existing</div>
            <ul className="space-y-1">
              {departments.map((d) => (
                <li key={d.id} className="flex items-center justify-between gap-2 px-3 py-2 rounded-xl hover:bg-slate-50">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="w-3 h-3 rounded-full shrink-0" style={{ background: d.color }} />
                    <span className="font-medium truncate">{d.name}</span>
                    <span className="text-xs text-slate-500 truncate">{d.mission}</span>
                  </div>
                  <button onClick={() => remove(d.id)} className="text-rose-500 text-sm hover:underline">
                    {t("delete")}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Modal>
  );
}
