/* What an agent actually did, and what it still needs to run.
 *
 * Every row here is a record the runtime kept. There is no synthesis, no "estimated", no
 * placeholder timeline — an empty section means the runtime recorded nothing, and says so.
 * That is the single most important property of this screen: a monitoring surface that
 * invents activity is worse than no monitoring surface, because it is believed.
 *
 * Logs are shown as a bounded tail and labelled as one. They are admin-only at the route,
 * because a log line can carry anything the runtime wrote — a prompt, a tool argument, part
 * of a document NOVA never saw — and no attempt is made to sanitise that.
 */

import * as React from "react";
import { AlertTriangle, FileText, Loader2, RefreshCw, ScrollText } from "lucide-react";

import { GlassPanel, SectionHeader, StatusPill } from "@/components/glass";
import { usePanel } from "@/lib/hooks";
import { sinceIso } from "@/lib/state";

type Stream = { stream: string; filename: string; present: boolean; bytes: number | null };
type Activity = {
  agent_id: string;
  tasks: any[];
  executions: any[];
  decisions: any[];
  logs: Stream[];
};
type LogBody = {
  stream: string; filename: string; present: boolean;
  lines: string[]; truncated: boolean; bytes: number; error?: string;
};

export function AgentActivity({ agentId }: { agentId: string }) {
  const [nonce, setNonce] = React.useState(0);
  const activity = usePanel<Activity>(
    `/agents/${encodeURIComponent(agentId)}/activity`, 20000, nonce,
  );

  if (activity.state === "loading") {
    return <GlassPanel className="p-5"><p className="text-ink-faint text-[13px]">reading…</p></GlassPanel>;
  }
  if (activity.state === "forbidden") {
    return (
      <GlassPanel className="p-5">
        <p className="text-ink-muted text-[13px]">
          Activity includes policy decisions and log excerpts, so it is not visible to your role.
        </p>
      </GlassPanel>
    );
  }
  if (activity.state === "error") {
    return (
      <GlassPanel className="p-5">
        <p className="text-blocked text-[13px]"><b>Activity could not be read.</b> {activity.message}</p>
        <button type="button" onClick={() => setNonce((n) => n + 1)}
          className="glass-solid text-ink mt-3 inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium">
          <RefreshCw className="size-3.5" /> Try again
        </button>
      </GlassPanel>
    );
  }

  const data = activity.data;

  return (
    <div className="space-y-5">
      <GlassPanel className="p-5">
        <SectionHeader
          title="Recent work" detail="Items the runtime holds for this agent."
          action={
            <button type="button" onClick={() => setNonce((n) => n + 1)}
              className="text-ink-faint hover:text-ink inline-flex items-center gap-1.5 text-[11.5px]">
              <RefreshCw className="size-3" /> Refresh
            </button>
          }
        />
        {data.tasks.length === 0 ? (
          <Empty>The runtime holds no work for this agent.</Empty>
        ) : (
          <ul className="divide-glass-border divide-y">
            {data.tasks.slice(0, 12).map((t: any) => (
              <li key={t.task_id} className="flex flex-wrap items-center gap-3 py-2 first:pt-0 last:pb-0">
                <StatusPill state={
                  t.runtime_status === "blocked" ? "blocked"
                  : t.runtime_status === "running" ? "running"
                  : t.runtime_status === "done" ? "neutral" : "waiting"
                }>{String(t.runtime_status ?? "unknown")}</StatusPill>
                <span className="text-ink min-w-0 flex-1 truncate text-[12.5px]">{t.title}</span>
                <span className="text-ink-faint text-[11.5px]">
                  {t.started_at ? sinceIso(new Date(t.started_at * 1000).toISOString()) : "not started"}
                </span>
              </li>
            ))}
          </ul>
        )}
      </GlassPanel>

      <GlassPanel className="p-5">
        <SectionHeader title="Scheduled executions" detail="What the runtime recorded when a schedule fired." />
        {data.executions.length === 0 ? (
          <Empty>
            No execution has been recorded for this agent. A schedule that exists is not a
            schedule that ran — Hermes runs its ticker inside the gateway, so a deployment
            can hold a perfect schedule with nothing executing it.
          </Empty>
        ) : (
          <ul className="divide-glass-border divide-y">
            {data.executions.slice(0, 12).map((e: any, i: number) => (
              <li key={e.execution_id ?? i} className="flex flex-wrap items-center gap-3 py-2 first:pt-0 last:pb-0">
                <StatusPill state={e.status === "failed" ? "blocked" : e.status === "running" ? "running" : "neutral"}>
                  {String(e.status ?? "unknown")}
                </StatusPill>
                <span className="text-ink min-w-0 flex-1 truncate text-[12.5px]">{e.automation_name}</span>
                {e.error ? <span className="text-blocked text-[11.5px]">{e.error}</span> : null}
                <span className="text-ink-faint text-[11.5px]">
                  {e.started_at ? sinceIso(e.started_at) : "no start recorded"}
                </span>
              </li>
            ))}
          </ul>
        )}
      </GlassPanel>

      <GlassPanel className="p-5">
        <SectionHeader title="Policy decisions" detail="What was refused or escalated for this agent." />
        {data.decisions.length === 0 ? (
          <Empty>Nothing has been refused or escalated for this agent.</Empty>
        ) : (
          <ul className="divide-glass-border divide-y">
            {data.decisions.slice(0, 12).map((d: any, i: number) => (
              <li key={i} className="flex flex-wrap items-center gap-3 py-2 first:pt-0 last:pb-0">
                <span className="text-ink font-mono text-[12px]">{String(d.tool ?? "—")}</span>
                <span className="text-ink-faint ml-auto text-[11.5px]">{String(d.effect ?? "")}</span>
              </li>
            ))}
          </ul>
        )}
      </GlassPanel>

      <LogViewer agentId={agentId} streams={data.logs ?? []} />
    </div>
  );
}

function LogViewer({ agentId, streams }: { agentId: string; streams: Stream[] }) {
  const available = streams.filter((s) => s.present);
  const [stream, setStream] = React.useState<string>(available[0]?.stream ?? "agent");
  const [lines, setLines] = React.useState(200);
  const [nonce, setNonce] = React.useState(0);

  const log = usePanel<LogBody>(
    `/agents/${encodeURIComponent(agentId)}/logs?stream=${encodeURIComponent(stream)}&lines=${lines}`,
    // Read on demand. A log poll is a lot of bytes for a screen nobody is watching, and
    // there is a Refresh button for when somebody is.
    0,
    nonce,
  );

  return (
    <GlassPanel className="p-5">
      <SectionHeader
        icon={ScrollText} title="Logs"
        detail="The end of this agent's log files, as the runtime wrote them."
        action={
          <button type="button" onClick={() => setNonce((n) => n + 1)}
            className="text-ink-faint hover:text-ink inline-flex items-center gap-1.5 text-[11.5px]">
            <RefreshCw className="size-3" /> Refresh
          </button>
        }
      />

      {streams.length === 0 ? (
        <Empty>This runtime keeps no logs for this agent.</Empty>
      ) : (
        <>
          <div className="mb-3 flex flex-wrap items-center gap-1.5">
            {streams.map((s) => (
              <button
                key={s.stream} type="button" disabled={!s.present}
                onClick={() => setStream(s.stream)}
                className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[12px] font-medium transition-colors disabled:opacity-35 ${
                  s.stream === stream ? "glass-solid text-ink" : "text-ink-muted hover:text-ink"
                }`}
              >
                <FileText className="size-3" /> {s.filename}
                {s.present ? null : <span className="text-ink-faint">· none</span>}
              </button>
            ))}
            <label className="text-ink-faint ml-auto flex items-center gap-1.5 text-[11.5px]">
              Lines
              <select
                value={lines} onChange={(e) => setLines(Number(e.target.value))}
                className="border-glass-border bg-glass text-ink rounded-md border px-1.5 py-0.5 text-[11.5px]"
              >
                {[50, 200, 500, 2000].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
          </div>

          {log.state === "loading" ? (
            <p className="text-ink-faint flex items-center gap-1.5 text-[12.5px]">
              <Loader2 className="size-3.5 animate-spin" /> reading…
            </p>
          ) : log.state === "forbidden" ? (
            <p className="text-ink-muted text-[12.5px]">
              A log can contain anything the runtime wrote, so it is not visible to your role.
            </p>
          ) : log.state === "error" ? (
            <p className="text-blocked text-[12.5px]"><b>This log could not be read.</b> {log.message}</p>
          ) : !log.data.present ? (
            <Empty>The runtime has not written {log.data.filename} for this agent.</Empty>
          ) : (
            <>
              {log.data.truncated ? (
                <p className="text-ink-faint mb-2 flex items-start gap-1.5 text-[11.5px]">
                  <AlertTriangle className="mt-0.5 size-3 shrink-0" />
                  Showing the end of a longer file ({(log.data.bytes / 1024).toFixed(0)} KB).
                </p>
              ) : null}
              <pre className="border-glass-border bg-glass text-ink max-h-[460px] overflow-auto rounded-lg border p-3 font-mono text-[11.5px] leading-relaxed whitespace-pre-wrap">
                {log.data.lines.join("\n") || "(empty)"}
              </pre>
            </>
          )}
        </>
      )}
    </GlassPanel>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="text-ink-muted text-[12.5px] leading-relaxed">{children}</p>;
}
