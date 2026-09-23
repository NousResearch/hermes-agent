import * as React from "react";
import { Cpu } from "lucide-react";
import { GlassPanel, SectionHeader, StatusPill } from "@/components/glass";
import { Hint } from "@/components/tooltip";
import type { Loaded } from "@/lib/api";
import { absolute, modelLabel, modelState, since } from "@/lib/state";
import type { ModelError, ModelStatus } from "./types";

/* Whether the workforce can reach its model.
 *
 * The one question every other screen depends on, and the one the dashboard used not to
 * answer: the header showed a green runtime pill while every agent call was refused. The
 * verdict is the runtime's own record — newest successful call against newest failure —
 * so it is evidence, not a probe, and it says "untested" rather than "healthy" when there
 * is none. */

const OWNER_LABEL: Record<string, string> = {
  "account admin": "Needs the cloud account admin",
  operator: "Fixable by the operator",
};

export function ModelAccessPanel({ model, compact }: { model: Loaded<ModelStatus>; compact?: boolean }) {
  if (model.state !== "ok") return null;
  const data = model.data;
  const failure = data.last_failure;
  const failing = data.state === "failing" && failure;
  const target = [data.configured.provider, data.configured.model].filter(Boolean).join(" · ") || "not declared";

  // Healthy or untested and asked to stay small: one quiet line, not a banner.
  if (compact && !failing) {
    return (
      <div className="flex flex-wrap items-center gap-2 text-[12.5px]">
        <StatusPill state={modelState(data.state)}>{modelLabel(data.state)}</StatusPill>
        <span className="text-ink-muted font-mono text-[12px]">{target}</span>
        {data.last_success ? (
          <Hint text={absolute(data.last_success.at)}>
            <span className="text-ink-faint">last answer {since(data.last_success.at)}</span>
          </Hint>
        ) : (
          <span className="text-ink-faint">no model call recorded yet</span>
        )}
      </div>
    );
  }

  return (
    <GlassPanel
      className={`p-5 ${failing ? "border-blocked/40" : ""}`}
      role={failing ? "alert" : undefined}
    >
      <SectionHeader
        title="Model access"
        icon={Cpu}
        detail="Whether agents' calls to their model are succeeding, from the runtime's own record."
        action={<StatusPill state={modelState(data.state)}>{modelLabel(data.state)}</StatusPill>}
      />
      <dl className="grid gap-x-6 gap-y-2 text-[13px] sm:grid-cols-[auto_1fr]">
        <dt className="text-ink-faint">Configured</dt>
        <dd className="text-ink font-mono text-[12.5px] break-all">
          {target}{data.configured.region ? ` · ${data.configured.region}` : ""}
        </dd>
        <dt className="text-ink-faint">Last answer</dt>
        <dd className="text-ink">
          {data.last_success ? (
            <Hint text={absolute(data.last_success.at)}>
              <span>{since(data.last_success.at)} · {data.last_success.profile}</span>
            </Hint>
          ) : "none recorded"}
        </dd>
        {failure ? (
          <>
            <dt className="text-ink-faint">Last failure</dt>
            <dd className="text-ink">
              <Hint text={absolute(failure.at)}>
                <span>{since(failure.at)} · {failure.profile}</span>
              </Hint>
            </dd>
          </>
        ) : null}
      </dl>
      {failing ? <FailureExplanation error={failure.error} /> : null}
      {data.state === "unknown" ? (
        <p className="text-ink-muted mt-3 text-[12.5px] leading-relaxed">
          No agent has called the model yet, so there is nothing to judge. The first task or
          channel conversation will settle it.
        </p>
      ) : null}
    </GlassPanel>
  );
}

/** A provider failure: what happened, who can fix it, what to do — and the provider's own
 *  words one click away, because a summary that replaced the evidence would make the next
 *  incident harder. */
export function FailureExplanation({ error }: { error: ModelError }) {
  return (
    <div className="bg-blocked/8 border-blocked/25 mt-4 rounded-lg border p-3.5">
      <p className="text-ink text-[13.5px] font-semibold">{error.headline}</p>
      <p className="text-ink-muted mt-1 text-[12.5px] leading-relaxed">{error.remedy}</p>
      <p className="text-ink-faint mt-2 text-[12px]">{OWNER_LABEL[error.owner] ?? error.owner}</p>
      <TechnicalDetails text={error.raw} />
    </div>
  );
}

export function TechnicalDetails({ text, label = "Technical details" }: { text?: string; label?: string }) {
  if (!text) return null;
  return (
    <details className="group mt-2" onClick={(e) => e.stopPropagation()}>
      <summary className="text-ink-faint hover:text-ink-muted cursor-pointer text-[12px] select-none">
        {label}
      </summary>
      <pre className="text-ink-muted bg-glass-1 border-glass-border mt-1.5 overflow-x-auto rounded-md border p-2 font-mono text-[11.5px] leading-relaxed whitespace-pre-wrap">
        {text}
      </pre>
    </details>
  );
}
