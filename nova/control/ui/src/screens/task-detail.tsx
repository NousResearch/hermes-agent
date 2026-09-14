import * as React from "react";
import { CircleCheck, FileText, History, MessageSquare, X } from "lucide-react";
import { EmptyState, GlassPanel, SectionHeader, StatusPill } from "@/components/glass";
import { PanelBody } from "@/components/panel";
import { Hint } from "@/components/tooltip";
import { usePanel } from "@/lib/hooks";
import { absolute, since, taskLabel, taskState } from "@/lib/state";
import { plural } from "@/lib/api";
import type { TaskDetail } from "./types";

/** Bytes as a person reads them. */
function fileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function runState(outcome: string, status: string) {
  const key = (outcome || status || "").toLowerCase();
  if (key === "completed" || key === "done") return "running" as const;
  if (["crashed", "timed_out", "failed", "spawn_failed", "gave_up"].includes(key)) {
    return "blocked" as const;
  }
  if (key === "running") return "running" as const;
  return "waiting" as const;
}

/** What the runtime actually recorded about one task: every attempt, every note, and
 *  the files it produced. All of it was already durable and none of it was reachable. */
export function TaskDetailPanel({ taskId, onClose }: { taskId: string; onClose: () => void }) {
  const detail = usePanel<TaskDetail>(`/tasks/${encodeURIComponent(taskId)}`);

  // Escape closes, as it does for every other transient surface here.
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <GlassPanel elevated className="p-5">
      <div className="flex items-start justify-between gap-4">
        <SectionHeader title="Task record" icon={History}
          detail="Every attempt, note and file the runtime recorded for this work." />
        <button type="button" onClick={onClose} aria-label="Close task record"
          className="interactive text-ink-faint hover:text-ink shrink-0 rounded-lg p-1.5 transition-colors">
          <X className="size-4" />
        </button>
      </div>

      <PanelBody state={detail}>
        {(data) => (
          <div className="space-y-5">
            <div className="flex flex-wrap items-center gap-2">
              <StatusPill state={taskState(data.task.runtime_status)}>
                {taskLabel(data.task.runtime_status)}
              </StatusPill>
              <span className="text-ink text-[13.5px] font-medium">{data.task.title}</span>
              <span className="text-ink-faint font-mono text-[11px]">{data.task.task_id}</span>
            </div>

            {/* Attempts. A task that failed twice and succeeded once is three rows here,
                which is the answer to "why did this take all morning?". */}
            <section>
              <h3 className="text-ink-faint mb-2 text-[11px] font-semibold tracking-[0.08em] uppercase">
                {data.runs.length ? plural(data.runs.length, "attempt") : "Attempts"}
              </h3>
              {data.runs.length ? (
                <ol className="divide-glass-border divide-y">
                  {data.runs.map((run) => (
                    <li key={run.run_id} className="flex flex-wrap items-center gap-2 py-2 first:pt-0 last:pb-0">
                      <StatusPill state={runState(run.outcome, run.status)} dot={false}>
                        {run.outcome || run.status}
                      </StatusPill>
                      <span className="text-ink-faint text-[11.5px]">{run.agent_id}</span>
                      {run.summary ? (
                        <span className="text-ink-muted min-w-0 flex-1 truncate text-[12.5px]">{run.summary}</span>
                      ) : <span className="flex-1" />}
                      <Hint text={absolute(run.started_at)}>
                        <span className="text-ink-faint text-[11px] tabular-nums">{since(run.started_at)}</span>
                      </Hint>
                      {run.error ? (
                        <p className="text-blocked w-full text-[11.5px] leading-snug">{run.error}</p>
                      ) : null}
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="text-ink-faint text-[12.5px]">No worker has picked this up yet.</p>
              )}
            </section>

            <section>
              <h3 className="text-ink-faint mb-2 flex items-center gap-1.5 text-[11px] font-semibold tracking-[0.08em] uppercase">
                <MessageSquare className="size-3" /> Notes
              </h3>
              {data.notes.length ? (
                <ul className="divide-glass-border divide-y">
                  {data.notes.map((note, i) => (
                    <li key={i} className="py-2 first:pt-0 last:pb-0">
                      <div className="flex items-baseline gap-2">
                        <span className="text-ink text-[12.5px] font-medium">{note.author}</span>
                        <span className="text-ink-faint text-[11px]">{since(note.created_at)}</span>
                      </div>
                      <p className="text-ink-muted mt-0.5 text-[12.5px] leading-snug">{note.body}</p>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-ink-faint text-[12.5px]">No one has left a note.</p>
              )}
            </section>

            {/* Files are listed, not linked: the runtime's path for an artifact stops at
                the adapter, and a download needs a NOVA-mediated route that re-checks the
                tenant per request. Offering a dead link would be worse than offering none. */}
            <section>
              <h3 className="text-ink-faint mb-2 flex items-center gap-1.5 text-[11px] font-semibold tracking-[0.08em] uppercase">
                <FileText className="size-3" /> Files produced
              </h3>
              {data.artifacts.length ? (
                <ul className="divide-glass-border divide-y">
                  {data.artifacts.map((art) => (
                    <li key={art.artifact_id} className="flex items-center gap-3 py-2 first:pt-0 last:pb-0">
                      <span className="text-ink min-w-0 flex-1 truncate font-mono text-[12px]">{art.filename}</span>
                      <span className="text-ink-faint shrink-0 text-[11px]">{art.content_type || "file"}</span>
                      <span className="text-ink-faint shrink-0 text-[11px] tabular-nums">{fileSize(art.size_bytes)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-ink-faint text-[12.5px]">This task produced no files.</p>
              )}
            </section>

            {(data.depends_on.length || data.blocks.length) ? (
              <section>
                <h3 className="text-ink-faint mb-2 flex items-center gap-1.5 text-[11px] font-semibold tracking-[0.08em] uppercase">
                  <CircleCheck className="size-3" /> Dependencies
                </h3>
                {data.depends_on.length ? (
                  <p className="text-ink-muted text-[12px]">
                    Waits on <span className="font-mono">{data.depends_on.join(", ")}</span>
                  </p>
                ) : null}
                {data.blocks.length ? (
                  <p className="text-ink-muted mt-1 text-[12px]">
                    Blocks <span className="font-mono">{data.blocks.join(", ")}</span>
                  </p>
                ) : null}
              </section>
            ) : null}
          </div>
        )}
      </PanelBody>
    </GlassPanel>
  );
}

export { EmptyState };
