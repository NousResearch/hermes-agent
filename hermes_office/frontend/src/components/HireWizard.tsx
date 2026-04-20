import React, { useEffect, useMemo, useState } from "react";
import { Modal } from "./Modal";
import { useStore } from "../state";
import { api } from "../api";
import type { Preset, ResolvedRole } from "../types";
import { t, getLang } from "../i18n";

interface Props {
  onClose: () => void;
  onHired: () => void;
}

const SPRITES = [
  { id: "robot-1", emoji: "🤖" },
  { id: "cat", emoji: "🐱" },
  { id: "fox", emoji: "🦊" },
  { id: "panda", emoji: "🐼" },
  { id: "wizard", emoji: "🧙" },
  { id: "scientist", emoji: "🔬" },
  { id: "writer", emoji: "✍️" },
  { id: "designer", emoji: "🎨" },
  { id: "analyst", emoji: "📊" },
  { id: "translator", emoji: "🌐" },
  { id: "tutor", emoji: "🎓" },
] as const;

const HUES = [200, 25, 140, 280, 340, 60, 180, 100];

export function HireWizard({ onClose, onHired }: Props) {
  const departments = useStore((s) => s.departments);
  const presets = useStore((s) => s.presets);
  const refreshDepts = useStore((s) => s.refreshDepartments);
  const lang = getLang();

  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [sprite, setSprite] = useState<string>("robot-1");
  const [hue, setHue] = useState<number>(200);
  const [name, setName] = useState<string>("");
  const [pickedPresetId, setPickedPresetId] = useState<string | null>(null);
  const [freeText, setFreeText] = useState<string>("");
  const [resolved, setResolved] = useState<ResolvedRole | null>(null);
  const [resolving, setResolving] = useState(false);
  const [departmentId, setDepartmentId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [createdDeptName, setCreatedDeptName] = useState("");

  useEffect(() => {
    setDepartmentId(departments[0]?.id ?? null);
  }, [departments]);

  // Auto-resolve role text after 500ms idle
  useEffect(() => {
    if (!freeText.trim()) {
      setResolved(null);
      return;
    }
    setResolving(true);
    const id = setTimeout(async () => {
      try {
        const r = await api.resolveRole(freeText);
        setResolved(r);
      } catch (e) {
        console.error(e);
      } finally {
        setResolving(false);
      }
    }, 450);
    return () => clearTimeout(id);
  }, [freeText]);

  const picked: Preset | null = useMemo(
    () => presets.find((p) => p.id === pickedPresetId) ?? null,
    [presets, pickedPresetId],
  );

  const finalRole = picked?.label ?? (freeText.trim() || "Helper");
  const finalToolsets = resolved?.recommended_toolsets ?? picked?.toolsets ?? [];
  const finalSkills = resolved?.recommended_skills ?? picked?.skills ?? [];
  const finalSystem = picked?.system_prompt ?? "";
  const finalModelHint = resolved?.model_hint ?? null;
  const finalName = name.trim() || picked?.default_name || "Helper";

  async function ensureDept(): Promise<string> {
    if (departmentId) return departmentId;
    const d = await api.createDepartment({
      name: createdDeptName.trim() || "General",
      mission: "Auto-created on first hire.",
    });
    await refreshDepts();
    setDepartmentId(d.id);
    return d.id;
  }

  async function hire() {
    setBusy(true);
    try {
      const did = await ensureDept();
      const cap = await api.capacity().catch(() => null);
      const model = finalModelHint || cap?.model.name || "gemma4-e2b-hermes";
      await api.createEmployee({
        department_id: did,
        name: finalName,
        role: finalRole,
        avatar: { sprite_id: sprite as any, hue },
        model,
        provider: undefined,
        enabled_toolsets: finalToolsets,
        skills: finalSkills,
        system_prompt: finalSystem,
        runtime: "simulated",
      });
      onHired();
    } catch (e) {
      alert(`hire failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal onClose={onClose} title={`${t("hireBtn")} (${step}/3)`} wide>
      <div className="px-2">
        <Stepper step={step} />

        {step === 1 && (
          <section className="grid gap-4">
            <h2 className="text-xl font-display font-semibold">{t("step1Title")}</h2>
            <div className="grid grid-cols-4 gap-3">
              {SPRITES.map((s) => (
                <button
                  key={s.id}
                  onClick={() => setSprite(s.id)}
                  className={`p-4 rounded-2xl border-2 transition-all ${
                    sprite === s.id ? "border-sky-500 bg-sky-50 scale-105" : "border-slate-200 hover:border-slate-300"
                  }`}
                  aria-pressed={sprite === s.id}
                >
                  <div className="text-4xl">{s.emoji}</div>
                </button>
              ))}
            </div>
            <div>
              <label className="text-sm font-medium text-slate-700">color</label>
              <div className="flex gap-2 mt-2 flex-wrap">
                {HUES.map((h) => (
                  <button
                    key={h}
                    aria-label={`hue ${h}`}
                    onClick={() => setHue(h)}
                    className={`w-10 h-10 rounded-full border-2 ${hue === h ? "border-slate-900 scale-110" : "border-slate-200"}`}
                    style={{ background: `hsl(${h} 80% 65%)` }}
                  />
                ))}
              </div>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-700">{t("name")}</label>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Aria"
                className="mt-1 w-full rounded-2xl border border-slate-200 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-sky-200"
                maxLength={64}
              />
            </div>
          </section>
        )}

        {step === 2 && (
          <section className="grid gap-4">
            <h2 className="text-xl font-display font-semibold">{t("step2Title")}</h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {presets.map((p) => (
                <button
                  key={p.id}
                  onClick={() => { setPickedPresetId(p.id); setFreeText(""); setResolved(null); }}
                  className={`p-4 rounded-2xl border-2 text-left transition-all ${
                    pickedPresetId === p.id ? "border-sky-500 bg-sky-50" : "border-slate-200 hover:border-slate-300"
                  }`}
                >
                  <div className="text-3xl mb-1" aria-hidden>{p.pictogram}</div>
                  <div className="font-semibold">{lang === "zh-CN" ? p.label_zh : p.label}</div>
                  <div className="text-xs text-slate-500 line-clamp-2">{p.summary}</div>
                </button>
              ))}
            </div>
            <details className="card p-3" open={!pickedPresetId}>
              <summary className="cursor-pointer font-medium">… or describe your own</summary>
              <textarea
                value={freeText}
                onChange={(e) => { setFreeText(e.target.value); setPickedPresetId(null); }}
                placeholder={t("describePlaceholder")}
                className="mt-3 w-full h-28 rounded-2xl border border-slate-200 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-sky-200"
              />
              <div className="mt-2 text-sm">
                {resolving && <span className="text-slate-500">resolving…</span>}
                {resolved && (
                  <div className="space-y-1">
                    <div className="text-slate-600">
                      <span className="font-medium">{t("toolsets")}:</span>{" "}
                      {resolved.recommended_toolsets.length
                        ? resolved.recommended_toolsets.map((id) => <span key={id} className="chip mr-1">{id}</span>)
                        : <span className="text-slate-400">none picked</span>}
                    </div>
                    <div className="text-slate-600">
                      <span className="font-medium">{t("skills")}:</span>{" "}
                      {resolved.recommended_skills.length
                        ? resolved.recommended_skills.map((id) => <span key={id} className="chip mr-1">{id}</span>)
                        : <span className="text-slate-400">none picked</span>}
                    </div>
                    {resolved.model_hint && (
                      <div className="text-slate-600">
                        <span className="font-medium">{t("model")}:</span>{" "}
                        <span className="chip">{resolved.model_hint}</span>
                      </div>
                    )}
                    <div className="text-xs text-slate-400">confidence {(resolved.confidence * 100).toFixed(0)}%</div>
                  </div>
                )}
              </div>
            </details>
          </section>
        )}

        {step === 3 && (
          <section className="grid gap-4">
            <h2 className="text-xl font-display font-semibold">{t("step3Title")}</h2>
            <div className="card p-4 flex items-center gap-4">
              <div
                className="w-16 h-16 rounded-full grid place-items-center text-3xl"
                style={{ background: `hsl(${hue} 80% 65%)` }}
                aria-hidden
              >
                {SPRITES.find((s) => s.id === sprite)?.emoji ?? "🤖"}
              </div>
              <div>
                <div className="font-semibold text-lg">{finalName}</div>
                <div className="text-sm text-slate-600">{finalRole}</div>
              </div>
            </div>
            <div className="grid sm:grid-cols-2 gap-3">
              <div className="card p-3">
                <div className="text-xs uppercase tracking-wider text-slate-500">{t("toolsets")}</div>
                <div className="mt-1 flex flex-wrap gap-1">
                  {finalToolsets.length
                    ? finalToolsets.map((id) => <span key={id} className="chip">{id}</span>)
                    : <span className="text-slate-400 text-sm">none</span>}
                </div>
              </div>
              <div className="card p-3">
                <div className="text-xs uppercase tracking-wider text-slate-500">{t("skills")}</div>
                <div className="mt-1 flex flex-wrap gap-1">
                  {finalSkills.length
                    ? finalSkills.map((id) => <span key={id} className="chip">{id}</span>)
                    : <span className="text-slate-400 text-sm">none</span>}
                </div>
              </div>
            </div>
            <div>
              <label className="text-sm font-medium text-slate-700">department</label>
              {departments.length === 0 ? (
                <div className="mt-1">
                  <input
                    value={createdDeptName}
                    onChange={(e) => setCreatedDeptName(e.target.value)}
                    placeholder="department name (we'll create it)"
                    className="w-full rounded-2xl border border-slate-200 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-sky-200"
                  />
                  <div className="text-xs text-slate-500 mt-1">
                    no departments yet — we'll create one for this hire.
                  </div>
                </div>
              ) : (
                <select
                  value={departmentId ?? ""}
                  onChange={(e) => setDepartmentId(e.target.value)}
                  className="mt-1 w-full rounded-2xl border border-slate-200 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-sky-200"
                >
                  {departments.map((d) => (
                    <option key={d.id} value={d.id}>{d.name}</option>
                  ))}
                </select>
              )}
            </div>
          </section>
        )}

        <div className="mt-6 flex justify-between">
          <button
            className="btn-ghost"
            onClick={() => (step === 1 ? onClose() : setStep((s) => (s - 1) as 1 | 2 | 3))}
          >
            {step === 1 ? t("cancel") : t("back")}
          </button>
          {step < 3 ? (
            <button
              className="btn-primary"
              disabled={step === 2 && !pickedPresetId && !freeText.trim()}
              onClick={() => setStep((s) => (s + 1) as 1 | 2 | 3)}
            >
              {t("next")}
            </button>
          ) : (
            <button className="btn-primary !bg-emerald-500 hover:!bg-emerald-600" disabled={busy} onClick={hire}>
              {busy ? "…" : t("confirmHire")}
            </button>
          )}
        </div>
      </div>
    </Modal>
  );
}

function Stepper({ step }: { step: 1 | 2 | 3 }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      {[1, 2, 3].map((n) => (
        <div
          key={n}
          className={`flex-1 h-2 rounded-full ${n <= step ? "bg-sky-500" : "bg-slate-200"}`}
          aria-hidden
        />
      ))}
    </div>
  );
}
