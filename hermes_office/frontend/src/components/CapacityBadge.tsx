import React, { useEffect, useState } from "react";
import { useStore } from "../state";
import { api } from "../api";
import type { CapacityReport } from "../types";
import { t } from "../i18n";

function color(level: "ok" | "warn" | "bad"): string {
  return level === "ok" ? "bg-emerald-100 text-emerald-700"
       : level === "warn" ? "bg-amber-100 text-amber-700"
       : "bg-rose-100 text-rose-700";
}

function classify(report: CapacityReport): "ok" | "warn" | "bad" {
  if (report.recommended_concurrency <= 0) return "bad";
  if (report.employee_count > report.recommended_concurrency) return "warn";
  if (report.memory_headroom_gb < 1.0) return "warn";
  return "ok";
}

export function CapacityBadge() {
  const cap = useStore((s) => s.capacity);
  const refresh = useStore((s) => s.refreshCapacity);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!cap) refresh().catch(() => undefined);
  }, [cap, refresh]);

  if (!cap) return <div className="chip">{t("capacity")}: …</div>;

  const tone = classify(cap);
  const c = color(tone);

  return (
    <div className="relative">
      <button
        className={`chip ${c}`}
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="dialog"
        aria-expanded={open}
        title={`${t("capacity")} (model: ${cap.model.name})`}
      >
        <span aria-hidden>⚡</span>
        {t("capacity")}: {cap.recommended_concurrency} / {cap.employee_count}
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-2 z-30 w-80 card p-4 text-sm">
          <div className="font-semibold mb-1">{cap.model.name}</div>
          <div className="text-slate-500 text-xs mb-3">
            {cap.host.cores} cores · {cap.host.ram_gb.toFixed(1)} GB RAM
            {cap.host.gpus.length ? ` · ${cap.host.gpus[0].name} ${cap.host.gpus[0].vram_gb} GB` : ""}
          </div>
          <dl className="grid grid-cols-2 gap-y-1">
            <dt className="text-slate-500">recommended</dt><dd>{cap.recommended_concurrency}</dd>
            <dt className="text-slate-500">live</dt><dd>{cap.employee_count}</dd>
            <dt className="text-slate-500">p95 latency</dt><dd>{cap.expected_p95_latency_ms} ms</dd>
            <dt className="text-slate-500">$/hr est</dt><dd>${cap.est_usd_per_hour.toFixed(3)}</dd>
            <dt className="text-slate-500">RAM headroom</dt><dd>{cap.memory_headroom_gb.toFixed(1)} GB</dd>
          </dl>
          {cap.notes.length > 0 && (
            <ul className="mt-3 list-disc pl-4 text-xs text-slate-600 space-y-0.5">
              {cap.notes.map((n, i) => <li key={i}>{n}</li>)}
            </ul>
          )}
          <button className="mt-3 text-xs text-sky-600 hover:underline" onClick={() => refresh()}>refresh</button>
        </div>
      )}
    </div>
  );
}
