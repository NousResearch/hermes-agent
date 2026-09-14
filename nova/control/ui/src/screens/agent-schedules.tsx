/* One agent's schedules, with the executions the runtime actually recorded.
 *
 * Reads `/agents/<id>/automations`, which returns the runtime's own records plus, for each,
 * whatever `cron.executions` holds. An empty execution list means the runtime has recorded
 * none — shown as exactly that, never softened into "not yet". Hermes' cron ticker lives
 * inside the gateway, so a deployment can hold a perfect schedule that nothing runs, and
 * that is the single most useful thing this screen can tell somebody.
 *
 * Editing goes to `/automations/<id>/update`, which reaches `cron.jobs.update_job`. There
 * is no second scheduler here; the Control Centre is the management layer.
 */

import * as React from "react";
import { CalendarClock, Loader2, Pause, Pencil, Play, Trash2 } from "lucide-react";

import { GlassCard, GlassPanel, SectionHeader, StatusPill } from "@/components/glass";
import { TextInput } from "@/components/form";
import { post } from "@/lib/api";
import { SaveResult, type SaveState, useEditable } from "@/lib/editing";
import { usePanel } from "@/lib/hooks";
import { absoluteIso, sinceIso } from "@/lib/state";

type Execution = {
  execution_id?: string; status?: string; started_at?: string; finished_at?: string;
  error?: string; outcome?: string;
};

type Automation = {
  automation_id: string; name: string; agent_id: string;
  schedule_display: string; schedule_expression?: string; schedule_kind?: string;
  enabled: boolean; state: string; next_run_at?: string | null;
  paused_reason?: string; executions?: Execution[];
};

type Payload = {
  agent_id: string;
  automations: Automation[];
  scheduler?: { running?: boolean; healthy?: boolean; detail?: string };
};

export function AgentSchedules({ agentId }: { agentId: string }) {
  const [nonce, setNonce] = React.useState(0);
  const loaded = usePanel<Payload>(
    `/agents/${encodeURIComponent(agentId)}/automations`, 30000, nonce,
  );
  const refresh = () => setNonce((n) => n + 1);

  if (loaded.state === "loading") {
    return <GlassPanel className="p-5"><p className="text-ink-faint text-[13px]">reading…</p></GlassPanel>;
  }
  if (loaded.state === "forbidden") {
    return <GlassPanel className="p-5"><p className="text-ink-muted text-[13px]">Not visible to your role.</p></GlassPanel>;
  }
  if (loaded.state === "error") {
    return (
      <GlassPanel className="p-5">
        <p className="text-blocked text-[13px]">
          <b>This agent's schedules could not be read.</b> {loaded.message}
        </p>
      </GlassPanel>
    );
  }

  const { automations, scheduler } = loaded.data;

  return (
    <div className="space-y-5">
      {scheduler && scheduler.running === false ? (
        <GlassPanel className="border-waiting/30 p-4">
          <p className="text-waiting text-[12.5px] font-medium">Nothing is running these schedules.</p>
          <p className="text-ink-muted mt-1 text-[12px] leading-relaxed">
            {scheduler.detail ||
              "No scheduler heartbeat for this agent. Schedules are recorded and nothing executes them."}
          </p>
        </GlassPanel>
      ) : null}

      <GlassPanel className="p-5">
        <SectionHeader
          icon={CalendarClock} title="Schedules"
          detail="Recurring work this agent holds."
          action={<span className="text-ink-faint text-[11.5px]">{automations.length} declared</span>}
        />
        {automations.length === 0 ? (
          <p className="text-ink-muted text-[12.5px]">
            This agent has no schedules. Declare one from the Automations screen, where it
            goes through the compiler that checks it against this agent's permissions.
          </p>
        ) : (
          <div className="space-y-3">
            {automations.map((a) => (
              <ScheduleRow key={a.automation_id} automation={a} onChanged={refresh} />
            ))}
          </div>
        )}
      </GlassPanel>
    </div>
  );
}

export function ScheduleRow({
  automation, onChanged, showAgent = false, onOpenAgent,
}: {
  automation: Automation; onChanged: () => void;
  showAgent?: boolean; onOpenAgent?: (id: string) => void;
}) {
  const [editing, setEditing] = React.useState(false);
  const [state, setState] = React.useState<SaveState>({ kind: "idle" });
  const [pending, setPending] = React.useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = React.useState(false);

  async function act(action: string, body: Record<string, unknown> = {}) {
    if (pending) return;
    setPending(action);
    setState({ kind: "saving" });
    try {
      // One route, with the verb in the body: `_write_route` maps /automations/<id>/decide
      // to the single declared write route. Posting /automations/<id>/pause resolves to an
      // undeclared route and is refused — correctly, since an undeclared write is
      // unroutable rather than admin-only.
      await post(`/automations/${encodeURIComponent(automation.automation_id)}/decide`, {
        action, ...body,
      });
      setState({ kind: "saved", applied: true, detail: "", files: [] });
      onChanged();
    } catch (cause) {
      setState({
        kind: "error",
        message: cause instanceof Error ? cause.message : "the request did not complete",
      });
    } finally {
      setPending(null);
    }
  }

  const runs = automation.executions ?? [];

  return (
    <GlassCard className="p-4" interactive={false}>
      <div className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-ink text-[13px] font-medium">{automation.name}</span>
            <StatusPill state={automation.enabled ? "running" : "waiting"}>
              {automation.enabled ? "Scheduled" : "Paused"}
            </StatusPill>
          </div>
          {showAgent ? (
            <button
              type="button" onClick={() => onOpenAgent?.(automation.agent_id)}
              className="text-ink-muted hover:text-ink mt-1 font-mono text-[11.5px] underline-offset-2 hover:underline"
            >
              {automation.agent_id}
            </button>
          ) : null}
          <p className="text-ink-muted mt-1.5 text-[12px]">
            {automation.schedule_display}
            {automation.schedule_expression ? (
              <span className="text-ink-faint font-mono"> · {automation.schedule_expression}</span>
            ) : null}
          </p>
        </div>

        <div className="flex items-center gap-1.5">
          <IconButton
            label={automation.enabled ? "Pause" : "Resume"}
            busy={pending === "pause" || pending === "resume"}
            onClick={() => act(automation.enabled ? "pause" : "resume")}
            icon={automation.enabled ? Pause : Play}
          />
          <IconButton label="Edit" busy={false} onClick={() => setEditing((e) => !e)} icon={Pencil} />
          <IconButton
            label="Delete" tone="blocked" busy={pending === "delete"}
            onClick={() => setConfirmDelete((c) => !c)} icon={Trash2}
          />
        </div>
      </div>

      <dl className="border-glass-border mt-3 grid gap-x-6 gap-y-1 border-t pt-3 text-[12px] sm:grid-cols-2">
        <Row label="Next run">
          {automation.enabled
            ? automation.next_run_at
              ? <span title={absoluteIso(automation.next_run_at)}>{sinceIso(automation.next_run_at)}</span>
              : "not computed"
            : "held while paused"}
        </Row>
        <Row label="Recorded executions">
          {runs.length ? String(runs.length) : "none recorded"}
        </Row>
      </dl>

      {runs.length ? (
        <ul className="border-glass-border mt-2 space-y-1 border-t pt-2">
          {runs.slice(0, 5).map((run, i) => (
            <li key={run.execution_id ?? i} className="flex flex-wrap items-center gap-2 text-[11.5px]">
              <StatusPill state={run.status === "failed" ? "blocked" : run.status === "running" ? "running" : "neutral"}>
                {run.status ?? "unknown"}
              </StatusPill>
              <span className="text-ink-faint">
                {run.started_at ? sinceIso(run.started_at) : "no start recorded"}
              </span>
              {run.error ? <span className="text-blocked">{run.error}</span> : null}
            </li>
          ))}
        </ul>
      ) : null}

      {editing ? (
        <ScheduleEditor
          automation={automation}
          onDone={() => { setEditing(false); onChanged(); }}
        />
      ) : null}

      {confirmDelete ? (
        <div className="border-blocked/30 mt-3 rounded-lg border p-3">
          <p className="text-ink-muted text-[12px]">
            Deleting removes this schedule from the runtime. Its recorded executions go with
            it, and this cannot be undone from here.
          </p>
          <button
            type="button" disabled={pending === "delete"}
            onClick={() => act("delete")}
            className="border-blocked/40 bg-blocked/10 text-blocked mt-2 inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-[12px] font-medium disabled:opacity-40"
          >
            {pending === "delete" ? <Loader2 className="size-3.5 animate-spin" /> : <Trash2 className="size-3.5" />}
            {pending === "delete" ? "Deleting…" : "Delete this schedule"}
          </button>
        </div>
      ) : null}

      <SaveResult state={state} />
    </GlassCard>
  );
}

/** Name and timing only. The objective is not editable: it passed the compiler on the way
 *  in, and a route that could rewrite it afterwards would make that a gate you walk through
 *  once and then step around. Changing what an automation does means declaring a new one. */
function ScheduleEditor({
  automation, onDone,
}: { automation: Automation; onDone: () => void }) {
  const server = React.useMemo(
    () => ({ name: automation.name, schedule: automation.schedule_display }),
    [automation],
  );
  const editor = useEditable(server, async (value) => {
    const result = await post(`/automations/${encodeURIComponent(automation.automation_id)}/decide`, {
      action: "update", updates: value,
    });
    onDone();
    return result;
  });
  const v = editor.value;

  return (
    <div className="border-glass-border mt-3 space-y-3 rounded-lg border p-3">
      <div className="grid gap-3 sm:grid-cols-2">
        <TextInput
          id={`sch-name-${automation.automation_id}`} label="Name" value={v.name}
          onChange={(name) => editor.edit((c) => ({ ...c, name }))}
        />
        <TextInput
          id={`sch-when-${automation.automation_id}`} label="When" value={v.schedule} mono
          onChange={(schedule) => editor.edit((c) => ({ ...c, schedule }))}
          hint="Plain English or cron. The runtime's own parser validates it before anything is written."
        />
      </div>
      <div className="flex items-center gap-3">
        <button
          type="button" disabled={!editor.dirty || editor.busy}
          onClick={editor.submit}
          className="glass-solid text-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12px] font-medium disabled:opacity-40"
        >
          {editor.busy ? <Loader2 className="size-3.5 animate-spin" /> : null}
          {editor.busy ? "Updating…" : "Update schedule"}
        </button>
        {editor.dirty ? <span className="text-waiting text-[11.5px]">Unsaved changes</span> : null}
      </div>
      <SaveResult state={editor.state} />
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-ink-faint">{label}</dt>
      <dd className="text-ink text-right">{children}</dd>
    </div>
  );
}

function IconButton({
  label, icon: Icon, onClick, busy, tone,
}: {
  label: string; icon: React.ComponentType<{ className?: string }>;
  onClick: () => void; busy: boolean; tone?: "blocked";
}) {
  return (
    <button
      type="button" onClick={onClick} disabled={busy} title={label} aria-label={label}
      className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[12px] font-medium transition-colors disabled:opacity-40 ${
        tone === "blocked"
          ? "border-blocked/30 text-blocked"
          : "border-glass-border text-ink-muted hover:text-ink"
      }`}
    >
      {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Icon className="size-3.5" />}
    </button>
  );
}
