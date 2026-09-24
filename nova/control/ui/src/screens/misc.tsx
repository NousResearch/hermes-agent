import * as React from "react";

import { CorpusPanel } from "@/screens/corpus";
import {
  Activity, Blocks, BookOpen, CircleCheck, Gauge, ListChecks, ShieldCheck, Target,
} from "lucide-react";
import { Chip, EmptyState, GlassCard, GlassPanel, SectionHeader, StatusPill } from "@/components/glass";
import { Hint, InfoDot } from "@/components/tooltip";
import { PanelBody } from "@/components/panel";
import { TaskDetailPanel } from "./task-detail";
import { TaskActions, TaskProblem } from "./task-actions";
import { CapabilityList } from "@/components/capabilities";
import { ApplyChannels, StartObjective } from "./run-actions";
import { plural } from "@/lib/api";
import {
  absolute, absoluteIso, channelLabel, channelState, dayLabel, decisionLabel, decisionState,
  objectiveLabel, objectiveState, since, sinceIso, taskLabel, taskState,
} from "@/lib/state";
import type { Loaded } from "@/lib/api";
import type {
  Budget, Channel, Decision, KnowledgeSource, ModelStatus, Objective, Policy, Task,
} from "./types";

/* ── Work ─────────────────────────────────────────────────────────────────── */

export function WorkScreen({
  tasks, model, onChanged,
}: {
  tasks: Loaded<{ tasks: Task[]; counts?: Record<string, number> }>;
  model?: ModelStatus; onChanged: () => void;
}) {
  // The open record is local to this screen: it is a view of one row, not navigation.
  const [openTask, setOpenTask] = React.useState<string | null>(null);
  return (
    <PanelBody state={tasks} empty={(d) => d.tasks.length ? null : {
      title: "Nothing on the board",
      detail: "Work appears when an objective is submitted or a channel routes a conversation to an agent.",
      hint: "nova objective submit <bundle> <id>",
    }}>
      {(data) => {
        // Three different asks, so three lists: a failure wants a fix and a retry, a held
        // or review item wants a decision, and everything else wants nothing.
        const failed = data.tasks.filter((t) => t.attention_kind === "failed");
        const decisions = data.tasks.filter((t) => t.attention_kind === "decision");
        const rest = data.tasks.filter((t) => !t.attention_kind);
        const row = (t: Task) => (
          <TaskRow key={t.task_id} task={t} model={model} onChanged={onChanged}
                   onOpen={() => setOpenTask(t.task_id)} />
        );
        return (
          <div className="space-y-6">
            {openTask ? (
              <TaskDetailPanel taskId={openTask} onClose={() => setOpenTask(null)} />
            ) : null}
            {failed.length ? (
              <section>
                <SectionHeader title="Failed — needs a fix"
                  detail="These stopped on an error. Fix the cause, then retry." icon={ListChecks} />
                <div className="space-y-2">{failed.map(row)}</div>
              </section>
            ) : null}
            {decisions.length ? (
              <section>
                <SectionHeader title="Waiting on a decision"
                  detail="Held or awaiting review. A person decides what happens next." icon={CircleCheck} />
                <div className="space-y-2">{decisions.map(row)}</div>
              </section>
            ) : null}
            {rest.length ? (
              <section>
                {failed.length || decisions.length ? <SectionHeader title="Everything else" /> : null}
                <div className="space-y-2">{rest.map(row)}</div>
              </section>
            ) : null}
          </div>
        );
      }}
    </PanelBody>
  );
}

function TaskRow({
  task, onOpen, model, onChanged,
}: { task: Task; onOpen?: () => void; model?: ModelStatus; onChanged?: () => void }) {
  return (
    <GlassCard
      className="w-full p-3.5 text-left"
      interactive={Boolean(onOpen)}
      {...(onOpen ? {
        role: "button", tabIndex: 0, onClick: onOpen,
        "aria-label": `Open the record for ${task.title}`,
        onKeyDown: (e: React.KeyboardEvent) => {
          if (e.target !== e.currentTarget) return;
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onOpen(); }
        },
      } : {})}
    >
      <div className="flex flex-wrap items-start gap-x-3 gap-y-1.5">
        <StatusPill state={task.attention_kind === "failed" ? "blocked" : taskState(task.runtime_status)}
                    className="mt-0.5 shrink-0">
          {task.attention_kind === "failed" ? "Failed" : taskLabel(task.runtime_status)}
        </StatusPill>
        <div className="min-w-0 flex-1 basis-64">
          <p className="text-ink text-[13.5px] leading-snug font-medium">{task.title}</p>
          <p className="text-ink-faint mt-0.5 text-[12px]">
            {task.agent_id ?? "unassigned"}
            {task.consecutive_failures ? ` · ${plural(task.consecutive_failures, "failed attempt")}` : ""}
          </p>
          <TaskProblem task={task} />
        </div>
        <Hint text={absolute(task.created_at)}>
          <span className="text-ink-faint shrink-0 text-[12px]">{since(task.created_at)}</span>
        </Hint>
      </div>
      {onChanged ? <TaskActions task={task} model={model} onDone={onChanged} /> : null}
    </GlassCard>
  );
}

/* ── Approvals ────────────────────────────────────────────────────────────────
   There is no pending-approval endpoint, so this screen is assembled from three
   things that are real: work already held for a human, escalations the policy layer
   has actually made, and the standing requirements declared per agent and channel.
   It must never look like an inbox of live requests that does not exist. */

export function ApprovalsScreen({
  tasks, decisions, agents, channels, canSeeDecisions, model, onChanged, onOpenWork,
}: {
  tasks: Task[]; decisions: Decision[]; canSeeDecisions: boolean;
  agents: Array<{ id: string; display_name?: string; approval_required_for?: string[] }>;
  channels: Channel[]; model?: ModelStatus; onChanged: () => void; onOpenWork: () => void;
}) {
  // Only what genuinely waits on a person. A crashed task is not an approval request, and
  // listing one here sent operators to approve something there was nothing to approve.
  const held = tasks.filter((t) => t.attention_kind === "decision");
  const failedCount = tasks.filter((t) => t.attention_kind === "failed").length;
  const escalations = decisions.filter(
    (d) => ["escalate", "require_approval"].includes(String(d.effect)),
  );
  const standing = [
    ...agents.filter((a) => (a.approval_required_for ?? []).length).map((a) => ({
      who: a.display_name ?? a.id, where: "Everywhere", what: a.approval_required_for!,
    })),
    ...channels.filter((c) => (c.approval_required_for ?? []).length).map((c) => ({
      who: (c.allowed_agents ?? []).join(", ") || "any granted agent",
      where: c.display_name ?? c.provider_label ?? c.provider,
      what: c.approval_required_for!,
    })),
  ];

  const failedNote = failedCount ? (
    <GlassPanel solid className="flex flex-wrap items-center gap-3 p-4">
      <StatusPill state="blocked">{plural(failedCount, "failed task")}</StatusPill>
      <span className="text-ink-muted min-w-0 flex-1 basis-60 text-[12.5px]">
        Failures need a fix and a retry, not an approval, so they are listed on the Work screen.
      </span>
      <button type="button" onClick={onOpenWork}
        className="glass-solid text-ink rounded-lg px-3 py-1.5 text-[12.5px] font-medium">
        Open Work
      </button>
    </GlassPanel>
  ) : null;

  if (!held.length && !escalations.length && !standing.length) {
    return (
      <div className="space-y-6">
        {failedNote}
        <EmptyState
          icon={CircleCheck}
          title="Nothing is waiting on a person"
          detail="Your workforce is operating inside its permitted actions, and no standing approval requirement is declared."
        />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {failedNote}
      <GlassPanel className="p-5">
        <SectionHeader
          title="Work held for a human"
          detail="Items the runtime stopped and will not resume without a decision."
          icon={CircleCheck}
        />
        {held.length ? (
          <div className="space-y-2.5">
            {held.map((task) => (
              <TaskRow key={task.task_id} task={task} model={model} onChanged={onChanged} />
            ))}
          </div>
        ) : (
          <EmptyState icon={CircleCheck} title="Nothing is waiting on a decision"
            detail="No work is held for review or paused for a person." />
        )}
      </GlassPanel>

      <GlassPanel className="p-5">
        <SectionHeader title="Standing requirements"
          detail="What always needs a person, and where that rule comes from." icon={ShieldCheck} />
        {standing.length ? (
          <ul className="divide-glass-border divide-y">
            {standing.map((row, i) => (
              <li key={i} className="flex flex-wrap items-center gap-3 py-2.5 first:pt-0 last:pb-0">
                <span className="text-ink min-w-0 flex-1 truncate text-[13px]">{row.who}</span>
                <Chip>{row.where}</Chip>
                <div className="flex flex-wrap gap-1.5">
                  {row.what.map((a) => (
                    <span key={a} className="text-waiting border-waiting/25 bg-waiting/10 rounded-md border px-1.5 py-0.5 font-mono text-[11px]">
                      {a}
                    </span>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <EmptyState icon={ShieldCheck} title="No standing approval requirements"
            detail="No agent or channel declares an action that always needs a person." />
        )}
      </GlassPanel>

      {canSeeDecisions ? (
        <GlassPanel className="p-5">
          <SectionHeader title="Escalations already made"
            detail="Calls the policy layer sent to a human rather than running." icon={Activity} />
          {escalations.length ? (
            <ul className="divide-glass-border divide-y">
              {escalations.slice(0, 12).map((d, i) => (
                <li key={i} className="flex items-center gap-3 py-2 first:pt-0 last:pb-0">
                  <StatusPill state="waiting">{decisionLabel(String(d.effect))}</StatusPill>
                  <span className="text-ink font-mono text-[12px]">{String(d.tool ?? "—")}</span>
                  <span className="text-ink-faint ml-auto truncate text-[11.5px]">
                    {String(d.agent_id ?? "")}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState icon={Activity} title="No escalation has happened yet"
              detail="The requirements above are declared; nothing has triggered one." />
          )}
        </GlassPanel>
      ) : null}
    </div>
  );
}

/* ── Activity ─────────────────────────────────────────────────────────────── */

export function ActivityScreen({ decisions }: {
  decisions: Loaded<{ decisions: Decision[]; total?: number; counts?: Record<string, number> }>;
}) {
  const [filter, setFilter] = React.useState<"all" | "deny" | "require_approval">("all");

  return (
    <PanelBody state={decisions} empty={(d) => d.decisions?.length ? null : {
      title: "No activity recorded yet",
      detail: "This timeline fills as agents call tools: every escalation and refusal the policy layer made.",
    }}>
      {(data) => {
        const rows = filter === "all"
          ? data.decisions
          : data.decisions.filter((d) => String(d.effect) === filter);

        // Group by day. A flat list of a hundred rows is a wall; the day a refusal
        // happened is the first thing an auditor scans for.
        const groups: Array<{ day: string; items: Decision[] }> = [];
        for (const row of rows) {
          const day = dayLabel(row.ts);
          const last = groups[groups.length - 1];
          if (last && last.day === day) last.items.push(row);
          else groups.push({ day, items: [row] });
        }

        const shown = data.decisions.length;
        const total = data.total ?? shown;
        // Counted over the loaded page, not over data.counts: those cover the whole
        // log, and a chip reading "Refused 75" above a list that can only ever show
        // 80 rows in total is a promise the list cannot keep.
        const pageCounts = data.decisions.reduce<Record<string, number>>((acc, d) => {
          const key = String(d.effect);
          acc[key] = (acc[key] ?? 0) + 1;
          return acc;
        }, {});

        return (
          <div className="space-y-5">
            <div className="flex flex-wrap items-center gap-2">
              {([
                ["all", "Everything", shown],
                ["deny", "Refused", pageCounts.deny ?? 0],
                ["require_approval", "Sent for approval", pageCounts.require_approval ?? 0],
              ] as const).map(([key, label, count]) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setFilter(key)}
                  aria-pressed={filter === key}
                  className={`interactive rounded-full px-3 py-1.5 text-[12px] font-medium transition-colors ${
                    filter === key ? "glass-solid text-ink" : "text-ink-faint hover:text-ink"
                  }`}
                >
                  {label}
                  <span className="text-ink-faint ml-1.5 tabular-nums">{count}</span>
                </button>
              ))}
              {total > shown ? (
                <span className="text-ink-faint ml-auto text-[11.5px]">
                  Showing the {shown} most recent of {total}.
                </span>
              ) : null}
            </div>

            {rows.length === 0 ? (
              <EmptyState icon={Activity} title="Nothing matches this filter"
                detail="The policy layer has recorded no event of that kind." />
            ) : (
              <div className="space-y-6">
                {groups.map((group) => (
                  <section key={group.day}>
                    <h3 className="text-ink-faint mb-3 text-[11px] font-semibold tracking-[0.08em] uppercase">
                      {group.day}
                    </h3>
                    <ol className="relative space-y-0">
                      {group.items.map((event, index) => {
                        const state = decisionState(String(event.effect));
                        return (
                          <li key={`${event.correlation_id}-${index}`} className="relative flex gap-4 pb-5 last:pb-0">
                            <div className="flex flex-col items-center">
                              <span className={`mt-1.5 size-2 shrink-0 rounded-full ${
                                state === "blocked" ? "bg-blocked" : state === "waiting" ? "bg-waiting" : "bg-running"}`} />
                              {index < group.items.length - 1 ? (
                                <span className="bg-glass-border mt-1 w-px flex-1" />
                              ) : null}
                            </div>
                            <div className="min-w-0 flex-1 pb-1">
                              <div className="flex flex-wrap items-center gap-2">
                                <StatusPill state={state} dot={false}>{decisionLabel(String(event.effect))}</StatusPill>
                                <span className="text-ink font-mono text-[12.5px]">{String(event.tool ?? "—")}</span>
                                <span className="text-ink-faint text-[11.5px]">{String(event.agent_id ?? "")}</span>
                                <span className="ml-auto">
                                  <Hint text={absoluteIso(event.ts)}>
                                    <span className="text-ink-faint text-[11.5px] tabular-nums">
                                      {sinceIso(event.ts)}
                                    </span>
                                  </Hint>
                                </span>
                              </div>
                              {event.reason ? (
                                <p className="text-ink-muted mt-1 text-[12.5px] leading-snug">{String(event.reason)}</p>
                              ) : null}
                            </div>
                          </li>
                        );
                      })}
                    </ol>
                  </section>
                ))}
              </div>
            )}
          </div>
        );
      }}
    </PanelBody>
  );
}

/* ── Objectives ───────────────────────────────────────────────────────────── */

export function ObjectivesScreen({ objectives }: { objectives: Loaded<{ objectives: Objective[]; detail?: string }> }) {
  return (
    <PanelBody state={objectives} empty={(d) => d.objectives?.length ? null : {
      title: "No objectives declared",
      detail: d.detail || "An objective is a repeatable business process: decomposed into steps, routed to agents, and re-runnable.",
      hint: "objectives/ in the tenant bundle",
    }}>
      {(data) => (
        <div className="grid gap-4 [grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr))]">
          {data.objectives.map((objective) => {
            const done = Number(objective.done ?? 0);
            const total = Number(objective.total ?? (objective.steps ?? []).length) || 1;
            const pct = Math.round((done / total) * 100);
            return (
              <GlassCard key={objective.id} className="p-5" interactive={false}>
                <div className="flex items-start gap-3">
                  <div className="min-w-0 flex-1">
                    <h3 className="text-ink text-[14.5px] leading-tight font-semibold">
                      {objective.title ?? objective.id}
                    </h3>
                    <p className="text-ink-faint mt-1 text-[12px]">
                      Owned by {objective.owner_display_name ?? objective.owner ?? "—"}
                    </p>
                  </div>
                  <StatusPill state={objectiveState(String(objective.state))}>
                    {objectiveLabel(String(objective.state ?? ""))}
                  </StatusPill>
                </div>

                {objective.description ? (
                  <p className="text-ink-muted mt-3 line-clamp-2 text-[12.5px] leading-relaxed">
                    {objective.description}
                  </p>
                ) : null}

                <div className="mt-4">
                  <div className="mb-1.5 flex items-baseline justify-between">
                    <span className="text-ink-faint text-[11px] tracking-wide uppercase">Progress</span>
                    <span className="text-ink text-[13px] font-semibold">{done} of {total} steps</span>
                  </div>
                  <div className="bg-glass-1 h-1.5 w-full overflow-hidden rounded-full">
                    <div
                      className="h-full rounded-full transition-[width] duration-700 ease-out"
                      style={{ width: `${pct}%`, background: "var(--accent)" }}
                      role="progressbar" aria-valuenow={done} aria-valuemin={0} aria-valuemax={total}
                      aria-label={`${objective.title ?? objective.id} progress`}
                    />
                  </div>
                </div>

                {(objective.blocking ?? []).length ? (
                  <p className="text-waiting mt-3 text-[12px]">
                    <Hint text="These steps have not finished, and the steps that depend on them cannot start until they do.">
                      Waiting on
                    </Hint>
                    : {(objective.blocking ?? []).join(", ")}
                  </p>
                ) : null}

                {(objective.refusals ?? []).length ? (
                  <p className="text-blocked mt-3 text-[12px]">
                    <Hint text="A step was routed to an agent its owner was never permitted to delegate to, so it was refused before any work was created.">
                      Routing refused
                    </Hint>
                    : {(objective.refusals ?? []).join("; ")}
                  </p>
                ) : null}

                {(objective.steps ?? []).length ? (
                  <ul className="border-glass-border mt-4 space-y-1.5 border-t pt-3">
                    {(objective.steps ?? []).slice(0, 5).map((step: any) => (
                      <li key={step.step_id} className="flex items-center gap-2.5">
                        <span className={`size-1.5 shrink-0 rounded-full ${
                          step.state === "done" ? "bg-running" : step.submitted ? "bg-waiting" : "bg-neutral"}`} />
                        <span className="text-ink-muted min-w-0 flex-1 truncate text-[12px]">{step.title}</span>
                        <span className="text-ink-faint shrink-0 text-[11px]">{step.assignee}</span>
                      </li>
                    ))}
                  </ul>
                ) : null}

                <StartObjective objective={objective} />
              </GlassCard>
            );
          })}
        </div>
      )}
    </PanelBody>
  );
}

/* ── Knowledge ────────────────────────────────────────────────────────────── */

export function KnowledgeScreen({
  knowledge, onChanged,
}: { onChanged?: () => void; knowledge: Loaded<{
  retrieval_enabled: boolean; sources: KnowledgeSource[]; undeclared_in_index?: string[];
  index_detail?: string; document_extraction?: boolean;
}> }) {
  const [open, setOpen] = React.useState<string | null>(null);
  return (
    <PanelBody state={knowledge} empty={(d) =>
      !d.retrieval_enabled ? {
        title: "Retrieval is unavailable on this runtime",
        detail: "Declared sources are recorded, but no agent can search them — so nothing here would reach a model.",
      } : d.sources.length ? null : {
        title: "No corpora declared",
        detail: "A corpus is a folder of the customer's own documents, chunked with provenance so an answer can cite where it came from.",
        hint: "nova knowledge ingest <bundle>",
      }}>
      {(data) => (
        <div className="space-y-4">
          {data.index_detail ? (
            // The endpoint says why the corpora read "Not indexed" and what to run. Without
            // it the page states a problem and offers no way out of it.
            <GlassPanel solid className="p-3">
              <p className="text-ink-muted text-[12.5px]">{data.index_detail}</p>
            </GlassPanel>
          ) : null}
          {(data.undeclared_in_index ?? []).length ? (
            <GlassPanel solid className="border-waiting/30 p-3">
              <p className="text-waiting text-[12.5px]">
                Indexed but no longer declared: {(data.undeclared_in_index ?? []).join(", ")} — still
                searchable by agents already granted them until re-ingested.
              </p>
            </GlassPanel>
          ) : null}
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {data.sources.map((source) => (
              <GlassCard key={source.id} className="p-4" interactive={false}>
                <div className="flex items-start gap-2">
                  <BookOpen className="text-ink-faint mt-0.5 size-4 shrink-0" />
                  <div className="min-w-0 flex-1">
                    <h3 className="text-ink truncate text-[13.5px] font-semibold">
                      {source.title ?? source.id}
                    </h3>
                    <p className="text-ink-faint line-clamp-2 text-[11.5px] leading-snug">
                      {source.description}
                    </p>
                  </div>
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-1.5">
                  <Chip>{source.classification}</Chip>
                  {source.indexed ? (
                    <>
                      <Chip>{plural(Number(source.documents ?? 0), "document")}</Chip>
                      <Chip>{plural(Number(source.chunks ?? 0), "chunk")}</Chip>
                    </>
                  ) : (
                    <StatusPill state="waiting">Not indexed</StatusPill>
                  )}
                </div>
                <div className="border-glass-border mt-3 flex items-center gap-2 border-t pt-2.5">
                  <p className="text-ink-faint min-w-0 flex-1 text-[11px]">
                    Readable by{" "}
                    <span className="text-ink-muted">
                      {(source.readable_by ?? []).length ? (source.readable_by ?? []).join(", ") : "nobody"}
                    </span>
                  </p>
                  {/* Collapsed by default: the card is the overview, and opening one is
                      asking about that corpus specifically. */}
                  <button
                    type="button"
                    aria-expanded={open === source.id}
                    onClick={() => setOpen(open === source.id ? null : source.id)}
                    className="text-ink-faint hover:text-ink shrink-0 text-[11.5px] underline-offset-2 transition-colors hover:underline"
                  >
                    {open === source.id ? "Hide documents" : "Documents"}
                  </button>
                </div>
              </GlassCard>
            ))}
          </div>

          {open ? <CorpusPanel sourceId={open} onChanged={onChanged} /> : null}
        </div>
      )}
    </PanelBody>
  );
}

/* ── Channels ─────────────────────────────────────────────────────────────── */

export function ChannelsScreen({
  channels,
}: { channels: Loaded<{ declared: boolean; channel_delivery: boolean; channels: Channel[]; catalogue: any[] }> }) {
  return (
    <PanelBody state={channels} empty={(d) =>
      // Only a runtime that cannot deliver at all is a dead end. "Nothing connected yet" is
      // not: that is exactly when someone needs to see what they could connect.
      !d.channel_delivery ? {
        title: "This runtime cannot deliver channels",
        detail: "A declared channel would be carried and never delivered, so none are offered.",
      } : null}>
      {(data) => (
        <div className="space-y-4">
          {data.channels.map((channel) => {
            const missing = Object.entries(channel.missing_by_agent ?? {})
              .filter(([, names]) => (names as string[])?.length);
            return (
              <GlassCard key={channel.id} className="p-5" interactive={false}>
                <div className="flex flex-wrap items-start gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <h3 className="text-ink text-[14.5px] font-semibold">
                        {channel.display_name ?? channel.id}
                      </h3>
                      <Hint text={
                        channel.live?.state === "connected"
                          ? `The gateway reports this connection up${channel.live.since ? ` since ${absoluteIso(channel.live.since)}` : ""}.`
                          : channel.live?.state === "disconnected"
                            ? `Credentials are in place but the gateway is not connected: ${channel.live.detail ?? "no detail"}. Restart the gateway after adding or changing a credential.`
                            : channel.status === "needs_credentials"
                              ? "A credential this provider needs is missing. See below."
                              : "Credentials are in place; this runtime does not report whether the connection is up."
                      }>
                        <span>
                          <StatusPill state={channelState(channel.status, channel.live?.state)}>
                            {channelLabel(channel.status, channel.live?.state)}
                          </StatusPill>
                        </span>
                      </Hint>
                    </div>
                    <p className="text-ink-faint mt-1 text-[12px]">
                      {channel.provider_label} · {channel.transport}
                      {channel.needs_public_endpoint ? (
                        <> · <Hint text="This provider calls in, so the deployment must expose a publicly reachable HTTPS endpoint. That is a security decision, not a checkbox.">needs a public endpoint</Hint></>
                      ) : null}
                    </p>
                  </div>
                </div>

                {channel.capabilities ? (
                  <div className="mt-3">
                    <CapabilityList capabilities={channel.capabilities} />
                  </div>
                ) : null}

                {/* The flow, because a routing table is a worse explanation than an arrow. */}
                <div className="border-glass-border mt-4 space-y-2 border-t pt-4">
                  {(channel.routes ?? []).map((route: any, i: number) => (
                    <div key={i} className="flex flex-wrap items-center gap-2 text-[12.5px]">
                      <Chip>{route.conversation || route.workspace || "everything else"}</Chip>
                      <span className="text-ink-faint">→</span>
                      <span className="text-ink font-medium">{route.agent}</span>
                    </div>
                  ))}
                  {!(channel.routes ?? []).length ? (
                    <p className="text-ink-faint text-[12px]">
                      No routes: inbound falls to the runtime's default profile rather than a granted agent.
                    </p>
                  ) : null}
                </div>

                {(channel.approval_required_for ?? []).length ? (
                  <div className="mt-3 flex flex-wrap items-center gap-1.5">
                    <Hint text="Anything reached over this channel escalates these to a human, on top of what the agent already escalates everywhere. A channel can tighten approval, never loosen it.">
                      <span className="text-waiting text-[12px]">Needs a human here</span>
                    </Hint>
                    {(channel.approval_required_for ?? []).map((a) => (
                      <span key={a} className="text-waiting border-waiting/25 bg-waiting/10 rounded-md border px-1.5 py-0.5 font-mono text-[11px]">
                        {a}
                      </span>
                    ))}
                  </div>
                ) : null}

                {missing.length ? (
                  <div className="border-glass-border mt-3 border-t pt-3">
                    <p className="text-ink-faint mb-1 text-[11px] tracking-wide uppercase">
                      Credentials still needed
                    </p>
                    {missing.map(([agent, names]) => (
                      <p key={agent} className="text-waiting text-[12px]">
                        <span className="font-mono">{agent}</span>: {(names as string[]).join(", ")}
                      </p>
                    ))}
                    <p className="text-ink-faint mt-1.5 text-[12px]">
                      Add it under Agents → that agent → Credentials. The value is written to the
                      agent&rsquo;s own <span className="font-mono">.env</span>; NOVA keeps only the name.
                    </p>
                  </div>
                ) : null}

                {channel.caveat ? (
                  <p className="text-ink-faint border-glass-border mt-3 border-t pt-2.5 text-[11.5px] leading-relaxed">
                    {channel.caveat}
                  </p>
                ) : null}
              </GlassCard>
            );
          })}

          {data.declared && data.channels.length ? (
            <GlassPanel className="px-5 pb-4 pt-1">
              <ApplyChannels />
            </GlassPanel>
          ) : null}

          <ChannelCatalogue
            catalogue={data.catalogue ?? []}
            connected={new Set(data.channels.map((c: any) => c.provider))}
          />
        </div>
      )}
    </PanelBody>
  );
}

/* Every platform this deployment can actually reach.
 *
 * Read from the runtime's own plugin manifests, so the list is what Hermes bundles rather
 * than a shorter one NOVA remembered. Each entry shows the credential *names* its adapter
 * reads and never a value — NOVA does not hold them, and `.env` is on the materialiser's
 * never-write list precisely so it cannot.
 *
 * There is no Connect button, and that is deliberate rather than unfinished. Connecting
 * means two things NOVA will not do from a browser: writing a customer's secret, and
 * claiming a connection works without having opened one. What it can do is say exactly
 * what a connection needs and where to put it, which is what this does.
 */
function ChannelCatalogue({
  catalogue, connected,
}: { catalogue: any[]; connected: Set<string> }) {
  const [open, setOpen] = React.useState<string | null>(null);
  if (!catalogue.length) return null;

  return (
    <GlassPanel className="p-5">
      <SectionHeader
        title="Available to connect"
        detail="Every platform this runtime bundles an adapter for."
        action={<span className="text-ink-faint text-[11.5px]">{catalogue.length} platforms</span>}
      />
      <ul className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
        {catalogue.map((provider: any) => {
          const live = connected.has(provider.id);
          const required = (provider.credentials ?? []).filter((c: any) => c.required);
          const expanded = open === provider.id;
          return (
            <li key={provider.id}>
              <button
                type="button"
                aria-expanded={expanded}
                onClick={() => setOpen(expanded ? null : provider.id)}
                className="border-glass-border hover:bg-glass w-full rounded-lg border p-3 text-left transition-colors"
              >
                <div className="flex items-center gap-2">
                  <span className="text-ink truncate text-[13px] font-medium">{provider.label}</span>
                  {live ? <StatusPill state="info" dot={false}>In use</StatusPill> : null}
                </div>
                <p className="text-ink-faint mt-1 font-mono text-[11px]">{provider.id}</p>
                <p className="text-ink-faint mt-1.5 text-[12px]">
                  {required.length
                    ? `${plural(required.length, "credential")} required`
                    : "no credentials declared"}
                </p>
                {provider.capabilities ? (
                  <p className="text-ink-muted mt-1 text-[12px]">
                    {nativeSummary(provider.capabilities)}
                  </p>
                ) : null}

                {expanded ? (
                  <div className="border-glass-border mt-2.5 space-y-2 border-t pt-2.5">
                    <CapabilityList capabilities={provider.capabilities} />
                    {provider.description ? (
                      <p className="text-ink-muted text-[12px] leading-relaxed">
                        {provider.description}
                      </p>
                    ) : null}
                    {(provider.credentials ?? []).length ? (
                      <div>
                        <p className="text-ink-faint text-[11px] tracking-wide uppercase">
                          Variables its adapter reads
                        </p>
                        <ul className="mt-1.5 space-y-1">
                          {(provider.credentials ?? []).map((c: any) => (
                            <li key={c.name} className="flex flex-wrap items-center gap-1.5">
                              <span className="text-ink font-mono text-[11.5px]">{c.name}</span>
                              {c.required ? (
                                <span className="text-waiting text-[11px]">required</span>
                              ) : (
                                <span className="text-ink-faint text-[11px]">optional</span>
                              )}
                              {c.secret ? (
                                <span className="text-ink-faint text-[11px]">· secret</span>
                              ) : null}
                            </li>
                          ))}
                        </ul>
                        <p className="text-ink-faint mt-2 text-[11px] leading-relaxed">
                          Set these in the agent&rsquo;s <span className="font-mono">.env</span>.
                          NOVA reports the names and never reads or writes the values.
                        </p>
                      </div>
                    ) : null}
                    {provider.caveat ? (
                      <p className="text-ink-faint text-[11.5px] leading-relaxed">{provider.caveat}</p>
                    ) : null}
                  </div>
                ) : null}
              </button>
            </li>
          );
        })}
      </ul>
      <p className="text-ink-faint mt-3 text-[11.5px] leading-relaxed">
        Declare a connection in <span className="font-mono">channels.yaml</span> to route
        conversations to an agent over it.
      </p>
    </GlassPanel>
  );
}

/* ── Policies ─────────────────────────────────────────────────────────────── */

export function PoliciesScreen({ policy }: { policy: Loaded<Policy> }) {
  return (
    <PanelBody state={policy} empty={(d) => d.agents?.length ? null : {
      title: "No policy declared",
      detail: "Without policy.yaml no enforcement plugin is installed, and agents behave exactly as they did before governance existed.",
      hint: "policy.yaml in the tenant bundle",
    }}>
      {(data) => (
        <div className="space-y-5">
          <GlassPanel solid className="flex flex-wrap items-center gap-3 p-4">
            <StatusPill state={data.enforced ? "running" : "waiting"}>
              {data.enforced ? "Enforced in the runtime" : "Declared, not enforced"}
            </StatusPill>
            <InfoDot text={data.enforced
              ? "Policy compiles to a plugin on the runtime's pre-tool-call hook, which vetoes a call before it runs and fails closed when it cannot decide."
              : "This runtime cannot enforce a declared policy, so it is recorded only."} />
            <span className="text-ink-faint text-[12px]">
              {plural(Object.keys(data.actions ?? {}).length, "business action")} ·{" "}
              {plural((data.baseline_tools ?? []).length, "baseline tool")}
            </span>
          </GlassPanel>

          <div className="grid gap-4 [grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr))]">
            {(data.agents ?? []).map((agent) => (
              <GlassCard key={agent.id} className="p-4" interactive={false}>
                <div className="flex items-center gap-2">
                  <h3 className="text-ink min-w-0 flex-1 truncate text-[13.5px] font-semibold">
                    {agent.display_name ?? agent.id}
                  </h3>
                  <StatusPill state={agent.unlisted_tool === "deny" ? "running" : "waiting"}>
                    {agent.unlisted_tool === "deny" ? "Default deny" : "Default allow"}
                  </StatusPill>
                </div>
                <div className="mt-3 grid grid-cols-3 gap-2 text-center">
                  <Tally label="Allowed" value={(agent.allow ?? []).length} tone="running" />
                  <Tally label="Denied" value={(agent.deny ?? []).length} tone="blocked" />
                  <Tally label="Needs a human" value={(agent.approval_actions ?? []).length} tone="waiting" />
                </div>
                {(agent.warnings ?? []).length ? (
                  <p className="text-waiting mt-3 text-[11.5px]">{(agent.warnings ?? []).join(" · ")}</p>
                ) : null}
              </GlassCard>
            ))}
          </div>
        </div>
      )}
    </PanelBody>
  );
}

function Tally({ label, value, tone }: { label: string; value: number; tone: "running" | "blocked" | "waiting" }) {
  const colour = tone === "running" ? "text-running" : tone === "blocked" ? "text-blocked" : "text-waiting";
  return (
    <div className="glass-solid rounded-lg py-2">
      <div className={`text-base font-semibold ${colour}`}>{value}</div>
      <div className="text-ink-muted text-[11.5px]">{label}</div>
    </div>
  );
}

/* ── Usage ────────────────────────────────────────────────────────────────── */

export function UsageScreen({ budget }: { budget: Loaded<Budget> }) {
  return (
    <PanelBody state={budget} empty={(d) => (d.observed ?? []).length || (d.controls ?? []).length ? null : {
      title: "No usage recorded",
      detail: "Figures appear once agents make model calls the runtime records.",
    }}>
      {(data) => {
        const observed = data.observed ?? [];
        const totalTokens = observed.reduce((sum, row) => sum + Number(row.total_tokens ?? 0), 0);
        const totalCalls = observed.reduce((sum, row) => sum + Number(row.api_calls ?? 0), 0);
        const totalCost = observed.reduce((sum, row) => sum + Number(row.estimated_cost_usd ?? 0), 0);
        return (
          <div className="space-y-5">
            <GlassPanel solid className="border-waiting/30 p-4">
              <p className="text-ink-muted text-[12.5px] leading-relaxed">
                <span className="text-waiting font-medium">Usage figures are observed.</span>{" "}
                <Hint text="The runtime cannot refuse a model call, so NOVA enforces budgets where it can: at a budget, new work and tool calls stop. A reply already in flight can still land. Figures are the runtime's estimate, not an invoice.">
                  {data.observed_caveat || "These figures are reported by the runtime."}
                </Hint>
              </p>
            </GlassPanel>

            {data.this_month ? <ThisMonth month={data.this_month} /> : null}

            <div className="grid gap-3 sm:grid-cols-3">
              <Headline label="Tokens" value={totalTokens.toLocaleString()} />
              <Headline label="API calls" value={totalCalls.toLocaleString()} />
              <Headline label="Recorded cost" value={`$${totalCost.toFixed(2)}`} />
            </div>

            {observed.length ? (
              <GlassPanel className="p-5">
                <SectionHeader title="By agent" icon={Gauge} />
                <ul className="divide-glass-border divide-y">
                  {observed.map((row) => (
                    <li key={row.agent_id} className="flex items-center gap-3 py-2.5 first:pt-0 last:pb-0">
                      <span className="text-ink min-w-0 flex-1 truncate text-[13px]">{row.agent_id}</span>
                      {row.available === false ? (
                        <span className="text-ink-faint text-[11.5px]">{row.detail || "nothing recorded"}</span>
                      ) : (
                        <>
                          <span className="text-ink-muted text-[12px]">
                            {Number(row.total_tokens ?? 0).toLocaleString()} tok
                          </span>
                          <span className="text-ink-faint w-16 text-right text-[12px]">
                            ${Number(row.estimated_cost_usd ?? 0).toFixed(2)}
                          </span>
                        </>
                      )}
                    </li>
                  ))}
                </ul>
              </GlassPanel>
            ) : null}

            {(data.controls ?? []).length ? (
              <GlassPanel className="p-5">
                <SectionHeader title="Limits that are actually enforced"
                  detail="Each control below compiles to a runtime key whose enforcement was verified at its call site." icon={ShieldCheck} />
                <ul className="divide-glass-border divide-y">
                  {(data.controls ?? []).map((control: any, i: number) => (
                    <li key={i} className="flex flex-wrap items-center gap-3 py-2.5 first:pt-0 last:pb-0">
                      <span className="text-ink-muted font-mono text-[12px]">{control.key}</span>
                      <span className="text-ink text-[13px] font-semibold">{String(control.value)}</span>
                      <span className="text-ink-faint min-w-0 flex-1 truncate text-[11.5px]">
                        {control.display_name}
                      </span>
                      <StatusPill state="running" dot={false}>{control.enforcement}</StatusPill>
                    </li>
                  ))}
                </ul>
              </GlassPanel>
            ) : null}
          </div>
        );
      }}
    </PanelBody>
  );
}

/** Month-to-date spend against each budget, counted the way the stop counts it. */
function ThisMonth({ month }: { month: NonNullable<Budget["this_month"]> }) {
  const rows = [
    { key: "tenant", label: "Whole deployment", spent: month.tenant.spent_usd, budget: month.tenant.budget_usd },
    ...month.agents.map((a) => ({ key: a.agent_id, label: a.display_name ?? a.agent_id, spent: a.spent_usd, budget: a.budget_usd })),
  ];
  return (
    <GlassPanel className="p-5">
      <SectionHeader title="This month" icon={Gauge}
        detail={month.caveat || "Month-to-date spend against each budget."} />
      <ul className="space-y-3">
        {rows.map((row) => {
          const spent = row.spent ?? null;
          const pct = row.budget && spent !== null ? Math.min(100, (spent / row.budget) * 100) : 0;
          const tone = !row.budget ? "var(--accent)"
            : pct >= 100 ? "var(--blocked)" : pct >= 80 ? "var(--waiting)" : "var(--accent)";
          return (
            <li key={row.key}>
              <div className="mb-1 flex items-baseline gap-3">
                <span className="text-ink min-w-0 flex-1 truncate text-[13px]">{row.label}</span>
                <span className="text-ink text-[12.5px] font-semibold">
                  {spent === null ? "unreadable" : `$${spent.toFixed(2)}`}
                </span>
                <span className="text-ink-faint w-28 text-right text-[11.5px]">
                  {row.budget ? `of $${row.budget.toFixed(2)}${pct >= 100 ? " · stopped" : ""}` : "no budget"}
                </span>
              </div>
              {row.budget ? (
                <div className="bg-glass-1 h-1.5 w-full overflow-hidden rounded-full"
                     role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}
                     aria-label={`${row.label} budget used`}>
                  <div className="h-full rounded-full" style={{ width: `${pct}%`, background: tone }} />
                </div>
              ) : null}
            </li>
          );
        })}
      </ul>
    </GlassPanel>
  );
}

function Headline({ label, value }: { label: string; value: string }) {
  return (
    <GlassCard className="p-4" interactive={false}>
      <div className="text-ink-faint text-[11px] font-medium tracking-wide uppercase">{label}</div>
      <div className="text-ink mt-1.5 text-2xl font-semibold">{value}</div>
    </GlassCard>
  );
}

/** "8 of 10 features native" — a catalogue of twenty-two platforms is scanned, and a wall of
 *  chips on every card hides the one comparison that matters. The detail opens on click. */
function nativeSummary(capabilities: Record<string, { supported: boolean | null }>): string {
  const known = Object.values(capabilities).filter((c) => c.supported !== null);
  if (!known.length) return "capabilities not read";
  const native = known.filter((c) => c.supported).length;
  return `${native} of ${known.length} features native`;
}
