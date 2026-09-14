import * as React from "react";
import {
  AlertTriangle, CalendarClock, Pause, Pencil, Play, Plus, ShieldCheck, Trash2, X,
} from "lucide-react";
import { Chip, EmptyState, GlassCard, GlassPanel, SectionHeader, StatusPill } from "@/components/glass";
import { PanelBody } from "@/components/panel";
import { Hint } from "@/components/tooltip";
import { absoluteIso, sinceIso } from "@/lib/state";
import { post } from "@/lib/api";
import type { Loaded } from "@/lib/api";
import type {
  Automation, AutomationGovernance, AutomationsPayload, SchedulerHealth,
} from "./types";

/** An ISO instant in the future, read the way a person reads a calendar. */
function until(iso?: string | null): string {
  if (!iso) return "";
  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) return "";
  const seconds = Math.round((ms - Date.now()) / 1000);
  if (seconds <= 0) return "due";
  if (seconds < 3600) return `in ${Math.max(1, Math.round(seconds / 60))}m`;
  if (seconds < 86_400) return `in ${Math.round(seconds / 3600)}h`;
  return `in ${Math.round(seconds / 86_400)}d`;
}

/** The banner that stops this screen from lying.
 *
 *  Hermes runs its scheduler inside the gateway and has no standalone cron daemon, so a
 *  deployment can hold a perfectly correct schedule that nothing ever executes — the
 *  runtime's own CLI calls this its most common support report. A list of schedules with
 *  no scheduler attached would read as a working automation suite. */
function SchedulerBanner({ health }: { health: Record<string, SchedulerHealth> }) {
  const agents = Object.entries(health);
  if (!agents.length) return null;
  const stopped = agents.filter(([, h]) => !h.running);
  const degraded = agents.filter(([, h]) => h.running && !h.healthy);
  if (!stopped.length && !degraded.length) {
    return (
      <GlassPanel solid className="p-3">
        <p className="text-running text-[12.5px] font-medium">
          Scheduler running. These automations will fire on their schedule.
        </p>
      </GlassPanel>
    );
  }
  const first = (stopped[0] ?? degraded[0])![1];
  return (
    <GlassPanel solid className="border-waiting/30 p-3">
      <p className="text-waiting flex items-start gap-2 text-[12.5px] font-medium">
        <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
        <span>
          {stopped.length
            ? `Nothing is running these schedules${
                stopped.length < agents.length ? ` for ${stopped.map(([a]) => a).join(", ")}` : ""
              }.`
            : "The scheduler is running but recent ticks have failed."}
        </span>
      </p>
      {first.detail ? (
        <p className="text-ink-muted mt-1.5 pl-5 text-[12px] leading-snug">{first.detail}</p>
      ) : null}
      {first.last_error ? (
        <p className="text-ink-faint mt-1 pl-5 font-mono text-[11px]">{first.last_error}</p>
      ) : null}
    </GlassPanel>
  );
}


/** Declare a new automation.
 *
 *  The fields are the spec, not a convenience wrapper over it: what this posts is an
 *  AutomationSpec document, and the server compiles it against the tenant before the
 *  runtime ever sees it. A refusal here is the compiler's own message — "declares
 *  permission(s) X that agent Y does not hold" — which is more useful than anything the
 *  form could guess in advance. */
function CreateAutomation({
  agents, onCreated, onCancel,
}: {
  agents: Array<{ id: string; display_name: string }>;
  onCreated: () => void;
  onCancel: () => void;
}) {
  const [form, setForm] = React.useState({
    id: "", title: "", agent: agents[0]?.id ?? "",
    schedule: "", objective: "", reason: "",
  });
  const [busy, setBusy] = React.useState(false);
  const [failure, setFailure] = React.useState("");

  const set = (key: keyof typeof form) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>,
  ) => setForm((f) => ({ ...f, [key]: e.target.value }));

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setFailure("");
    try {
      await post("/automations", {
        id: form.id.trim(),
        title: form.title.trim(),
        agent: form.agent,
        schedule: form.schedule.trim(),
        objective: form.objective.trim(),
        ...(form.reason.trim() ? { reason: form.reason.trim() } : {}),
      });
      onCreated();
    } catch (err) {
      setFailure(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const field =
    "glass-solid text-ink placeholder:text-ink-faint w-full rounded-lg px-2.5 py-1.5 text-[12.5px] outline-none focus-visible:border-glass-border-lit";
  const label = "text-ink-faint mb-1 block text-[11px] font-semibold tracking-[0.06em] uppercase";

  return (
    <GlassPanel elevated className="p-5">
      <div className="flex items-start justify-between gap-4">
        <SectionHeader title="Declare an automation" icon={ShieldCheck}
          detail="Compiled against this tenant before the runtime schedules it." />
        <button type="button" onClick={onCancel} aria-label="Cancel"
          className="interactive text-ink-faint hover:text-ink shrink-0 rounded-lg p-1.5 transition-colors">
          <X className="size-4" />
        </button>
      </div>

      <form onSubmit={submit} className="mt-3 space-y-3">
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <label className={label} htmlFor="a-id">Id</label>
            <input id="a-id" required value={form.id} onChange={set("id")}
              placeholder="nightly-reconciliation" className={`${field} font-mono`} />
          </div>
          <div>
            <label className={label} htmlFor="a-title">Title</label>
            <input id="a-title" required value={form.title} onChange={set("title")}
              placeholder="Nightly reconciliation" className={field} />
          </div>
          <div>
            <label className={label} htmlFor="a-agent">Agent</label>
            <select id="a-agent" required value={form.agent} onChange={set("agent")} className={field}>
              {agents.map((a) => (
                <option key={a.id} value={a.id}>{a.display_name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className={label} htmlFor="a-schedule">Schedule</label>
            <input id="a-schedule" required value={form.schedule} onChange={set("schedule")}
              placeholder="every day at 07:00" className={field} />
          </div>
        </div>
        <div>
          <label className={label} htmlFor="a-objective">Objective</label>
          <textarea id="a-objective" required rows={3} value={form.objective}
            onChange={set("objective")}
            placeholder="Reconcile yesterday's ledger against the bank export and report discrepancies."
            className={field} />
          <p className="text-ink-faint mt-1 text-[11px]">
            This runs inside the agent's profile, so the agent's policy is the ceiling —
            an automation can never reach past its agent.
          </p>
        </div>
        <div>
          <label className={label} htmlFor="a-reason">Reason (optional)</label>
          <input id="a-reason" value={form.reason} onChange={set("reason")}
            placeholder="SOX control 4.2" className={field} />
        </div>

        {failure ? (
          <p className="text-blocked text-[12px] leading-snug">{failure}</p>
        ) : null}

        <div className="flex justify-end gap-2">
          <button type="button" onClick={onCancel}
            className="text-ink-faint hover:text-ink rounded-lg px-2.5 py-1.5 text-[12px] transition-colors">
            Cancel
          </button>
          <button type="submit" disabled={busy}
            className="interactive glass-solid text-ink hover:border-glass-border-lit inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12px] font-medium transition-colors disabled:opacity-50">
            {busy ? "Declaring…" : "Declare automation"}
          </button>
        </div>
      </form>
    </GlassPanel>
  );
}

function AutomationCard({
  automation, governance, onDecide, onUpdate, onOpenAgent, busy,
}: {
  automation: Automation;
  governance?: AutomationGovernance;
  onDecide: (action: "pause" | "resume" | "delete") => void;
  onUpdate: (updates: { name?: string; schedule?: string }) => Promise<void>;
  onOpenAgent?: (agentId: string) => void;
  busy: boolean;
}) {
  const paused = !automation.enabled;
  const [confirming, setConfirming] = React.useState(false);
  const [editing, setEditing] = React.useState(false);
  const [draftName, setDraftName] = React.useState(automation.name);
  const [draftWhen, setDraftWhen] = React.useState(automation.schedule_display);
  const dirty = draftName !== automation.name || draftWhen !== automation.schedule_display;
  // Declared and verified are different facts. A schedule the runtime holds is a
  // declaration; an execution row is the only evidence anything ran.
  const everRan = Boolean(automation.last_run_at) || automation.runs.length > 0;
  return (
    <GlassCard className="p-4" interactive={false}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h3 className="text-ink truncate text-[13.5px] font-semibold">{automation.name}</h3>
          {onOpenAgent ? (
            <button
              type="button" onClick={() => onOpenAgent(automation.agent_id)}
              // Labelled by id, not by the display name shown: a screen reader (and a test)
              // needs something stable, and the visible text is the tenant's branding.
              aria-label={`Open agent ${automation.agent_id}`}
              className="text-ink-faint hover:text-ink mt-0.5 text-[11.5px] underline-offset-2 transition-colors hover:underline"
            >
              {automation.agent_display_name || automation.agent_id}
            </button>
          ) : (
            <p className="text-ink-faint mt-0.5 text-[11.5px]">
              {automation.agent_display_name || automation.agent_id}
            </p>
          )}
        </div>
        <StatusPill state={paused ? "waiting" : "running"}>
          {paused ? "Paused" : "Scheduled"}
        </StatusPill>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-1.5">
        <Chip>
          <CalendarClock className="mr-1 inline size-3" />
          {automation.schedule_display || automation.schedule_expression || "—"}
        </Chip>
        {automation.failure_streak > 0 ? (
          <Chip className="text-waiting border-waiting/25">
            {automation.failure_streak} consecutive failures
          </Chip>
        ) : null}
      </div>

      <dl className="border-glass-border mt-3 space-y-1.5 border-t pt-3 text-[12px]">
        <div className="flex items-baseline justify-between gap-3">
          <dt className="text-ink-faint">Next run</dt>
          <dd className="text-ink tabular-nums">
            {/* The runtime keeps next_run_at through a pause, so a paused automation
                still carries a future timestamp. Rendering it would say "in 18h" about
                something that will not run at all. */}
            {paused ? (
              <span className="text-ink-faint">held while paused</span>
            ) : automation.next_run_at ? (
              <Hint text={absoluteIso(automation.next_run_at)}>
                <span>{until(automation.next_run_at)}</span>
              </Hint>
            ) : (
              <span className="text-ink-faint">not scheduled</span>
            )}
          </dd>
        </div>
        <div className="flex items-baseline justify-between gap-3">
          <dt className="text-ink-faint">Last run</dt>
          <dd className="text-ink tabular-nums">
            {automation.last_run_at ? (
              <Hint text={absoluteIso(automation.last_run_at)}>
                <span>{sinceIso(automation.last_run_at)}</span>
              </Hint>
            ) : (
              // Said plainly: a schedule that has never fired is the symptom of a
              // scheduler that was never attached, and the banner above explains it.
              <span className="text-ink-faint">never run</span>
            )}
          </dd>
        </div>
      </dl>

      {automation.paused_reason ? (
        <p className="text-ink-muted mt-2 text-[11.5px]">Paused: {automation.paused_reason}</p>
      ) : null}
      {automation.last_error ? (
        <p className="text-blocked mt-2 text-[11.5px] leading-snug">{automation.last_error}</p>
      ) : null}

      <p className="text-ink-faint mt-2 text-[11px] leading-snug">
        {everRan
          ? "Declared, and the runtime has recorded an execution."
          : "Declared. The runtime has recorded no execution yet."}
      </p>

      {governance ? (
        <div className="border-glass-border mt-3 border-t pt-2">
          <p className="text-ink-faint mb-1 text-[10.5px] font-semibold tracking-[0.08em] uppercase">
            Governance
          </p>
          <dl className="space-y-0.5 text-[11.5px]">
            <div className="flex justify-between gap-3">
              <dt className="text-ink-faint">Declared by</dt>
              <dd className="text-ink">{governance.declared_by}</dd>
            </div>
            {governance.reason ? (
              <div className="flex justify-between gap-3">
                <dt className="text-ink-faint">Reason</dt>
                <dd className="text-ink truncate">{governance.reason}</dd>
              </div>
            ) : null}
            {governance.permissions.length ? (
              <div className="flex justify-between gap-3">
                <dt className="text-ink-faint">Permissions</dt>
                <dd className="text-ink truncate font-mono">{governance.permissions.join(", ")}</dd>
              </div>
            ) : null}
            <div className="flex justify-between gap-3">
              <dt className="text-ink-faint">Digest</dt>
              <dd className="text-ink-muted truncate font-mono">
                {governance.digest.replace("sha256:", "").slice(0, 12)}
              </dd>
            </div>
          </dl>
        </div>
      ) : (
        <p className="text-ink-faint mt-2 text-[11px]">
          Created outside NOVA — no declaration on record.
        </p>
      )}

      {/* Recent attempts, only when the runtime has any. An empty history is shown as
          nothing rather than as a zeroed chart. */}
      {automation.runs.length ? (
        <div className="border-glass-border mt-3 border-t pt-2">
          <p className="text-ink-faint mb-1.5 text-[10.5px] font-semibold tracking-[0.08em] uppercase">
            Recent runs
          </p>
          <ul className="space-y-1">
            {automation.runs.slice(0, 3).map((run) => (
              <li key={run.run_id} className="flex items-center gap-2 text-[11.5px]">
                <StatusPill
                  dot={false}
                  state={run.status === "completed" ? "running" : run.status === "failed" ? "blocked" : "waiting"}
                >
                  {run.status}
                </StatusPill>
                <span className="text-ink-faint ml-auto tabular-nums">{sinceIso(run.claimed_at)}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {confirming ? (
        // Deleting is not undoable and the declaration goes with it, so the confirmation
        // names the automation rather than asking "are you sure?" about nothing.
        <div className="border-blocked/30 mt-3 rounded-lg border p-2.5">
          <p className="text-ink text-[12px] leading-snug">
            Delete <span className="font-medium">{automation.name}</span>? The schedule and
            its declaration are removed. Executions already recorded are kept.
          </p>
          <div className="mt-2 flex justify-end gap-2">
            <button type="button" onClick={() => setConfirming(false)}
              className="text-ink-faint hover:text-ink rounded-lg px-2 py-1 text-[12px] transition-colors">
              Keep it
            </button>
            <button type="button" disabled={busy} onClick={() => onDecide("delete")}
              className="interactive text-blocked border-blocked/40 hover:bg-blocked/10 rounded-lg border px-2.5 py-1 text-[12px] font-medium transition-colors disabled:opacity-50">
              {busy ? "Deleting…" : "Delete"}
            </button>
          </div>
        </div>
      ) : (
        <div className="mt-3 flex justify-end gap-2">
          <button
            type="button"
            disabled={busy}
            onClick={() => setConfirming(true)}
            aria-label={`Delete ${automation.name}`}
            className="interactive text-ink-faint hover:text-blocked rounded-lg p-1.5 transition-colors disabled:opacity-50"
          >
            <Trash2 className="size-3.5" />
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => { setEditing((e) => !e); setDraftName(automation.name); setDraftWhen(automation.schedule_display); }}
            aria-label={`Edit ${automation.name}`}
            className="interactive text-ink-faint hover:text-ink rounded-lg p-1.5 transition-colors disabled:opacity-50"
          >
            <Pencil className="size-3.5" />
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => onDecide(paused ? "resume" : "pause")}
            className="interactive glass-solid text-ink hover:border-glass-border-lit inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[12px] font-medium transition-colors disabled:opacity-50"
          >
            {paused ? <Play className="size-3" /> : <Pause className="size-3" />}
            {busy ? "Working…" : paused ? "Resume" : "Pause"}
          </button>
        </div>
      )}

      {editing ? (
        <div className="border-glass-border mt-3 space-y-3 rounded-lg border p-3">
          <p className="text-ink-faint text-[11.5px] leading-relaxed">
            Name and timing only. What this automation <i>does</i> passed the compiler when it
            was declared; changing that means declaring a new one, so it goes through those
            checks again rather than around them.
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="block">
              <span className="text-ink mb-1 block text-[12px] font-medium">Name</span>
              <input
                type="text" value={draftName} onChange={(e) => setDraftName(e.target.value)}
                className="border-glass-border bg-glass text-ink focus-visible:ring-info/50 w-full rounded-lg border px-3 py-2 text-[12.5px] outline-none focus-visible:ring-2"
              />
            </label>
            <label className="block">
              <span className="text-ink mb-1 block text-[12px] font-medium">When</span>
              <input
                type="text" value={draftWhen} onChange={(e) => setDraftWhen(e.target.value)}
                className="border-glass-border bg-glass text-ink focus-visible:ring-info/50 w-full rounded-lg border px-3 py-2 font-mono text-[12.5px] outline-none focus-visible:ring-2"
              />
            </label>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button" disabled={!dirty || busy}
              onClick={async () => {
                await onUpdate({ name: draftName, schedule: draftWhen });
                setEditing(false);
              }}
              className="glass-solid text-ink rounded-lg px-3 py-1.5 text-[12px] font-medium disabled:opacity-40"
            >
              {busy ? "Updating…" : "Update schedule"}
            </button>
            {dirty ? <span className="text-waiting text-[11.5px]">Unsaved changes</span> : null}
            <span className="text-ink-faint text-[11.5px]">
              The runtime's own parser validates the timing before anything is written.
            </span>
          </div>
        </div>
      ) : null}
    </GlassCard>
  );
}

/** What runs on a schedule, and whether anything is running it.
 *
 *  Pause and resume are the only actions. Creating an automation would hand an agent an
 *  instruction NOVA never compiled and no policy reviewed — a governance decision, not a
 *  missing button. */
export function AutomationsScreen({
  automations, onChanged, onOpenAgent,
}: {
  automations: Loaded<AutomationsPayload>;
  onChanged: () => void;
  /** Open an agent's profile. Makes the agent ↔ schedule relationship work both ways. */
  onOpenAgent?: (agentId: string) => void;
}) {
  const [busy, setBusy] = React.useState<string | null>(null);
  const [failure, setFailure] = React.useState("");
  const [creating, setCreating] = React.useState(false);

  async function decide(id: string, action: "pause" | "resume" | "delete") {
    setBusy(id);
    setFailure("");
    try {
      await post(`/automations/${encodeURIComponent(id)}/decide`, { action });
      onChanged();
    } catch (err) {
      // Whatever the control plane said, verbatim — including "this endpoint requires
      // the admin role". A button is not an enforcement mechanism; the server is.
      setFailure(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  }

  async function update(id: string, updates: { name?: string; schedule?: string }) {
    setBusy(id);
    setFailure("");
    try {
      await post(`/automations/${encodeURIComponent(id)}/decide`, { action: "update", updates });
      onChanged();
    } catch (err) {
      setFailure(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  }

  return (
    <PanelBody
      state={automations}
      // Only the "no scheduler at all" case short-circuits to an empty state. An empty
      // *list* still renders the panel, because the way out of it is the Declare button
      // inside that panel — an empty state that hides the only remedy is a dead end.
      empty={(d) =>
        d.scheduling
          ? null
          : {
              title: "This runtime holds no scheduled work",
              detail: d.detail || "The connected runtime has no scheduler to read.",
            }
      }
    >
      {(data) => (
        <div className="space-y-5">
          <SchedulerBanner health={data.scheduler_health ?? {}} />

          {creating ? (
            <CreateAutomation
              agents={data.agents ?? []}
              onCreated={() => { setCreating(false); onChanged(); }}
              onCancel={() => setCreating(false)}
            />
          ) : null}

          {failure ? (
            <GlassPanel solid className="border-blocked/30 p-3">
              <p className="text-blocked text-[12.5px]">{failure}</p>
            </GlassPanel>
          ) : null}

          <GlassPanel className="p-5">
            <SectionHeader
              title="Scheduled work"
              icon={CalendarClock}
              detail="Recurring work the runtime holds for each agent."
              action={
                <div className="flex items-center gap-3">
                  <span className="text-ink-faint text-[12px]">
                    {data.counts?.enabled ?? 0} scheduled · {data.counts?.paused ?? 0} paused
                  </span>
                  <button type="button" onClick={() => setCreating(true)}
                    className="interactive glass-solid text-ink hover:border-glass-border-lit inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[12px] font-medium transition-colors">
                    <Plus className="size-3" /> Declare
                  </button>
                </div>
              }
            />
            {data.automations.length === 0 ? (
              <p className="text-ink-muted py-6 text-center text-[12.5px] leading-relaxed">
                No automations yet. An automation is recurring work an agent runs on a
                schedule — a nightly reconciliation, an hourly sweep — declared here and
                bounded by that agent's policy.
              </p>
            ) : null}
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {data.automations.map((a) => (
                <AutomationCard
                  key={a.automation_id}
                  automation={a}
                  governance={data.governance?.[a.automation_id]}
                  busy={busy === a.automation_id}
                  onDecide={(action) => decide(a.automation_id, action)}
                  onUpdate={(updates) => update(a.automation_id, updates)}
                  onOpenAgent={onOpenAgent}
                />
              ))}
            </div>
          </GlassPanel>
        </div>
      )}
    </PanelBody>
  );
}

export { EmptyState };
