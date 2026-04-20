import React, { useEffect, useMemo, useState } from "react";
import { Modal } from "./Modal";
import { useStore } from "../state";
import { api } from "../api";
import type { Employee, ResolvedRole } from "../types";
import { t } from "../i18n";

interface Props {
  empId: string;
  onClose: () => void;
  onSaved: () => void;
  onDeleted: () => void;
}

export function EmployeeEditor({ empId, onClose, onSaved, onDeleted }: Props) {
  const employees = useStore((s) => s.employees);
  const departments = useStore((s) => s.departments);
  const toolsetsCatalog = useStore((s) => s.toolsets);
  const skillsCatalog = useStore((s) => s.skills);
  const activityByEmp = useStore((s) => s.activityByEmp);
  const refreshEmployees = useStore((s) => s.refreshEmployees);

  const initial: Employee | undefined = useMemo(
    () => employees.find((e) => e.id === empId),
    [employees, empId],
  );

  const [draft, setDraft] = useState<Employee | null>(initial ?? null);
  const [cli, setCli] = useState<string>("");
  const [resolved, setResolved] = useState<ResolvedRole | null>(null);
  const [resolving, setResolving] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!initial) return;
    setDraft(initial);
    api.getEmployee(empId).then((d) => setCli(d.cli_command)).catch(() => undefined);
  }, [initial, empId]);

  if (!draft) {
    return (
      <Modal onClose={onClose} title="…">
        <div className="text-slate-500">employee not found</div>
      </Modal>
    );
  }

  const events = activityByEmp[empId] ?? [];

  const updateLocal = <K extends keyof Employee>(k: K, v: Employee[K]) =>
    setDraft({ ...draft, [k]: v });

  async function suggestFromRole() {
    setResolving(true);
    try {
      const r = await api.resolveRole(`${draft!.role} ${draft!.system_prompt}`);
      setResolved(r);
    } finally {
      setResolving(false);
    }
  }

  async function applySuggestion() {
    if (!resolved) return;
    updateLocal("enabled_toolsets", resolved.recommended_toolsets);
    updateLocal("skills", resolved.recommended_skills);
    if (resolved.model_hint) updateLocal("model", resolved.model_hint);
    setResolved(null);
  }

  async function save() {
    setBusy(true);
    try {
      await api.patchEmployee(draft!.id, {
        name: draft!.name,
        role: draft!.role,
        avatar: draft!.avatar,
        model: draft!.model,
        provider: draft!.provider,
        base_url: draft!.base_url,
        enabled_toolsets: draft!.enabled_toolsets,
        skills: draft!.skills,
        system_prompt: draft!.system_prompt,
        runtime: draft!.runtime,
      });
      await refreshEmployees();
      onSaved();
    } catch (e) {
      alert(`save failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (!confirm(`Delete ${draft!.name}?`)) return;
    setBusy(true);
    try {
      await api.deleteEmployee(draft!.id);
      onDeleted();
    } catch (e) {
      alert(`delete failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  function copyCli() {
    if (!cli) return;
    navigator.clipboard.writeText(cli).then(
      () => alert(t("cliCopy") + " ✓"),
      () => alert("clipboard blocked")
    );
  }

  const toggleInArray = (arr: string[], id: string): string[] =>
    arr.includes(id) ? arr.filter((x) => x !== id) : [...arr, id];

  return (
    <Modal onClose={onClose} title={`${t("name")}: ${draft.name}`} wide>
      <div className="grid lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 space-y-4">
          <div className="grid sm:grid-cols-2 gap-3">
            <label className="block">
              <span className="text-sm font-medium text-slate-700">{t("name")}</span>
              <input
                value={draft.name}
                onChange={(e) => updateLocal("name", e.target.value)}
                className="mt-1 w-full rounded-2xl border border-slate-200 px-4 py-2"
                maxLength={64}
              />
            </label>
            <label className="block">
              <span className="text-sm font-medium text-slate-700">{t("role")}</span>
              <input
                value={draft.role}
                onChange={(e) => updateLocal("role", e.target.value)}
                className="mt-1 w-full rounded-2xl border border-slate-200 px-4 py-2"
                maxLength={80}
              />
            </label>
          </div>
          <label className="block">
            <span className="text-sm font-medium text-slate-700">{t("model")}</span>
            <input
              value={draft.model}
              onChange={(e) => updateLocal("model", e.target.value)}
              className="mt-1 w-full rounded-2xl border border-slate-200 px-4 py-2"
            />
          </label>
          <div className="grid sm:grid-cols-2 gap-3">
            <label className="block">
              <span className="text-sm font-medium text-slate-700">provider</span>
              <input
                value={draft.provider ?? ""}
                onChange={(e) => updateLocal("provider", e.target.value || null)}
                placeholder="auto-detect"
                className="mt-1 w-full rounded-2xl border border-slate-200 px-4 py-2"
              />
            </label>
            <label className="block">
              <span className="text-sm font-medium text-slate-700">runtime</span>
              <select
                value={draft.runtime}
                onChange={(e) => updateLocal("runtime", e.target.value as "simulated" | "hermes")}
                className="mt-1 w-full rounded-2xl border border-slate-200 px-4 py-2"
              >
                <option value="simulated">simulated (demo)</option>
                <option value="hermes">hermes (live)</option>
              </select>
            </label>
          </div>
          <label className="block">
            <span className="text-sm font-medium text-slate-700">{t("persona")}</span>
            <textarea
              value={draft.system_prompt}
              onChange={(e) => updateLocal("system_prompt", e.target.value)}
              className="mt-1 w-full h-24 rounded-2xl border border-slate-200 px-4 py-2 text-sm font-mono"
              maxLength={4000}
            />
          </label>

          <div className="flex flex-wrap gap-2">
            <button className="btn-ghost" onClick={suggestFromRole} disabled={resolving}>
              {resolving ? "…" : t("suggestSkills")}
            </button>
            {resolved && (
              <button className="btn-primary" onClick={applySuggestion}>apply suggestion</button>
            )}
          </div>

          <div className="card p-3">
            <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">{t("toolsets")}</div>
            <div className="flex flex-wrap gap-1">
              {toolsetsCatalog.map((ts) => {
                const on = draft.enabled_toolsets.includes(ts.id);
                return (
                  <button
                    key={ts.id}
                    onClick={() => updateLocal("enabled_toolsets", toggleInArray(draft.enabled_toolsets, ts.id))}
                    className={`chip ${on ? "!bg-sky-500 !text-white" : ""}`}
                    title={ts.description}
                  >
                    {ts.id}
                  </button>
                );
              })}
            </div>
          </div>
          <div className="card p-3">
            <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">{t("skills")}</div>
            {skillsCatalog.length === 0 && <div className="text-sm text-slate-400">no SKILL.md files installed yet</div>}
            <div className="flex flex-wrap gap-1">
              {skillsCatalog.map((sk) => {
                const on = draft.skills.includes(sk.id);
                return (
                  <button
                    key={sk.id}
                    onClick={() => updateLocal("skills", toggleInArray(draft.skills, sk.id))}
                    className={`chip ${on ? "!bg-sky-500 !text-white" : ""}`}
                    title={sk.title}
                  >
                    {sk.id}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        <aside className="space-y-3">
          <div className="card p-3">
            <div className="text-xs uppercase tracking-wider text-slate-500 mb-1">{t("activity")}</div>
            <div className="space-y-1 text-sm max-h-72 overflow-y-auto">
              {events.length === 0
                ? <div className="text-slate-400">no recent activity</div>
                : events.slice(-12).reverse().map((e, i) => (
                    <div key={i} className="text-slate-700">
                      <span className="text-xs text-slate-400 mr-2">{new Date(e.ts).toLocaleTimeString()}</span>
                      <span className="font-mono text-xs text-slate-500 mr-1">{e.kind}</span>
                      {e.text}
                    </div>
                ))}
            </div>
          </div>
          <button className="btn-ghost w-full" onClick={copyCli} disabled={!cli}>{t("cliCopy")}</button>
          <details className="card p-3">
            <summary className="cursor-pointer text-rose-600 font-medium">{t("danger")}</summary>
            <div className="pt-2 text-sm text-slate-700">
              department: {departments.find((d) => d.id === draft.department_id)?.name ?? "?"}
              <button className="mt-2 btn !bg-rose-500 !text-white !rounded-2xl px-4 py-2" onClick={remove} disabled={busy}>
                {t("delete")}
              </button>
            </div>
          </details>
        </aside>
      </div>

      <div className="mt-6 flex justify-between">
        <button className="btn-ghost" onClick={onClose}>{t("cancel")}</button>
        <button className="btn-primary" onClick={save} disabled={busy}>save</button>
      </div>
    </Modal>
  );
}
