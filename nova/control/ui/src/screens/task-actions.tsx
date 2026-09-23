import * as React from "react";
import { RotateCcw, Play } from "lucide-react";
import { post } from "@/lib/api";
import type { ModelStatus, Task } from "./types";
import { TechnicalDetails } from "./model";

/* What went wrong with a task, and the one action that follows from it.
 *
 * The action sits behind an inline confirmation rather than firing on click: it changes
 * what the runtime does next, and a hover away is too close for that. The confirmation says
 * what will happen — and, when the model is still refusing calls, that a retry will most
 * likely fail the same way, so nobody spends a retry to learn what the Model access card
 * already knows. */

export function TaskProblem({ task }: { task: Task }) {
  const summary = task.error_summary;
  if (!task.last_error) return null;
  if (!summary?.headline) {
    return <TechnicalDetails text={task.last_error} label="Error details" />;
  }
  return (
    <div className="mt-1.5">
      <p className="text-blocked text-[12.5px] font-medium">{summary.headline}</p>
      {summary.cause ? (
        <p className="text-ink-muted mt-0.5 text-[12.5px]">
          Cause: {summary.cause.headline}.{" "}
          <span className="text-ink-faint">{summary.cause.remedy}</span>
        </p>
      ) : summary.detail ? (
        <p className="text-ink-faint mt-0.5 text-[12px]">{summary.detail}</p>
      ) : null}
      <TechnicalDetails
        text={[task.last_error, summary.cause?.raw].filter(Boolean).join("\n\nModel provider: ")}
      />
    </div>
  );
}

type Plan = { action: "resume" | "release"; label: string; explain: string; icon: typeof Play };

function planFor(task: Task): Plan | null {
  if (task.attention_kind === "failed") {
    return { action: "resume", label: "Retry", icon: RotateCcw,
             explain: "The task goes back on the board and a worker runs it again with the same instructions." };
  }
  if (String(task.runtime_status) === "review") {
    return { action: "release", label: "Approve & release", icon: Play,
             explain: "The reviewed work is let through to run. This is recorded under your name." };
  }
  if (task.attention_kind === "decision") {
    return { action: "resume", label: "Resume", icon: Play,
             explain: "The held task returns to where it was and a worker can pick it up." };
  }
  return null;
}

export function TaskActions({
  task, model, onDone,
}: { task: Task; model?: ModelStatus; onDone: () => void }) {
  const plan = planFor(task);
  const [confirming, setConfirming] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [result, setResult] = React.useState<{ ok: boolean; text: string } | null>(null);
  if (!plan) return null;

  const modelFailing = model?.state === "failing";
  const run = async () => {
    setBusy(true);
    setResult(null);
    try {
      const body: any = await post(`/work/${encodeURIComponent(task.task_id)}/decide`, {
        action: plan.action, reason: `${plan.label} from the Control Centre`,
      });
      setResult({ ok: true, text: `Done — now ${String(body?.resulting_status ?? "updated")}.` });
      setConfirming(false);
      onDone();
    } catch (error) {
      setResult({ ok: false, text: error instanceof Error ? error.message : String(error) });
    } finally {
      setBusy(false);
    }
  };

  const Icon = plan.icon;
  return (
    // Clicks here must not also open the row's record.
    <div className="border-glass-border mt-3 border-t pt-3" onClick={(e) => e.stopPropagation()}
         onKeyDown={(e) => e.stopPropagation()}>
      {confirming ? (
        <div className="space-y-2">
          <p className="text-ink text-[12.5px]">{plan.explain}</p>
          {modelFailing && task.attention_kind === "failed" ? (
            <p className="text-waiting text-[12.5px]">
              The model is still refusing calls, so this will most likely fail the same way.
            </p>
          ) : null}
          <div className="flex flex-wrap gap-2">
            <button type="button" disabled={busy} onClick={() => void run()}
              className="bg-accent text-accent-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium disabled:opacity-60">
              <Icon className="size-3.5" /> {busy ? "Working…" : `${plan.label} now`}
            </button>
            <button type="button" disabled={busy} onClick={() => setConfirming(false)}
              className="glass-solid text-ink rounded-lg px-3 py-1.5 text-[12.5px] font-medium">
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <button type="button" onClick={() => { setConfirming(true); setResult(null); }}
          className="glass-solid text-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium">
          <Icon className="size-3.5" /> {plan.label}
        </button>
      )}
      {result ? (
        <p role="status" className={`mt-2 text-[12px] ${result.ok ? "text-running" : "text-blocked"}`}>
          {result.text}
        </p>
      ) : null}
    </div>
  );
}
