import * as React from "react";

import { SoulEditor } from "@/screens/soul";
import { ArrowLeft, BookOpen, Boxes, ShieldCheck } from "lucide-react";
import { Chip, EmptyState, GlassCard, GlassPanel, SectionHeader, StatusPill } from "@/components/glass";
import { Hint, InfoDot } from "@/components/tooltip";
import { PanelBody } from "@/components/panel";
import { decisionLabel, initials, since, taskLabel, taskState } from "@/lib/state";
import { plural } from "@/lib/api";
import type { Loaded } from "@/lib/api";
import type { Agent, Channel, Decision, Budget, KnowledgeSource, Policy, Task } from "./types";

/** An AI worker, not a database row. What a person wants at a glance is: is it working,
 *  on what, what may it reach, and does anything need me. */
export function AgentCard({
  agent, tasks, channels, onOpen,
}: {
  agent: Agent; tasks: Task[]; channels: Channel[]; onOpen: () => void;
}) {
  const mine = tasks.filter((t) => t.agent_id === agent.id);
  const running = mine.filter((t) => ["running", "ready"].includes(String(t.runtime_status)));
  const attention = mine.filter((t) => t.needs_attention);
  const current = running[0] ?? mine[0];

  const reaching = channels.filter(
    (c) => (c.allowed_agents ?? []).includes(agent.id)
      || (c.derived_agents ?? []).some((d: any) => d.base_agent === agent.id),
  );

  const state = agent.enabled === false ? "neutral"
    : attention.length ? "waiting"
    : running.length ? "running"
    : "neutral";

  const label = agent.enabled === false ? "Not in service"
    : attention.length ? `${attention.length} need${attention.length === 1 ? "s" : ""} a human`
    : running.length ? `Working · ${plural(running.length, "task")}`
    : "Idle";

  return (
    <GlassCard
      className="w-full cursor-pointer p-4 text-left"
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onOpen(); } }}
      aria-label={`Open ${agent.display_name ?? agent.id}`}
    >
      <div className="flex items-start gap-3">
        <div className="glass-solid mt-0.5 grid size-9 shrink-0 place-items-center rounded-lg text-[12px] font-semibold">
          {initials(agent.display_name ?? agent.id)}
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-ink truncate text-[14px] leading-tight font-semibold">
            {agent.display_name ?? agent.id}
          </div>
          <div className="text-ink-faint truncate font-mono text-[11px]">{agent.id}</div>
        </div>
        {agent.in_sync === false ? (
          <Hint text="The runtime holds a different configuration from the one this bundle declares. Re-apply to reconcile.">
            <StatusPill state="blocked">Drifted</StatusPill>
          </Hint>
        ) : null}
      </div>

      <div className="mt-3">
        <StatusPill state={state}>{label}</StatusPill>
      </div>

      {current ? (
        <p className="text-ink-muted mt-2.5 line-clamp-2 text-[12.5px] leading-snug">
          {current.title}
        </p>
      ) : agent.description ? (
        <p className="text-ink-faint mt-2.5 line-clamp-2 text-[12.5px] leading-snug">
          {agent.description}
        </p>
      ) : null}

      <div className="border-glass-border mt-3 flex flex-wrap items-center gap-1.5 border-t pt-3">
        {reaching.map((c) => <Chip key={c.id}>{c.provider_label ?? c.provider}</Chip>)}
        {(agent.knowledge_sources ?? []).length ? (
          <Chip>{plural(agent.knowledge_sources!.length, "corpus", "corpora")}</Chip>
        ) : null}
        {(agent.approval_required_for ?? []).length ? (
          <Chip className="text-waiting border-waiting/25 bg-waiting/10">
            {agent.approval_required_for!.length} needs approval
          </Chip>
        ) : null}
        {!reaching.length && !(agent.knowledge_sources ?? []).length ? (
          <span className="text-ink-faint text-[11px]">No channels or corpora</span>
        ) : null}
      </div>
    </GlassCard>
  );
}

export function AgentsScreen({
  agents, tasks, channels, onOpen, limit,
}: {
  agents: Loaded<{ agents: Agent[] }>; tasks: Task[]; channels: Channel[];
  onOpen: (id: string) => void;
  /** Cap the preview. The Overview shows a sample beside the rest of the dashboard;
   *  the Agents screen shows every one. Without this the Overview simply becomes the
   *  Agents screen with a header, and the column beside it is left empty. */
  limit?: number;
}) {
  return (
    <PanelBody
      state={agents}
      empty={(d) => d.agents.length ? null : {
        title: "No agents materialized yet",
        detail: "An agent is declared in the tenant bundle and applied into the runtime. Nothing has been applied to this deployment.",
        hint: "nova apply <bundle>",
      }}
    >
      {(data) => (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {(limit ? data.agents.slice(0, limit) : data.agents).map((agent) => (
            <AgentCard key={agent.id} agent={agent} tasks={tasks} channels={channels}
                       onOpen={() => onOpen(agent.id)} />
          ))}
        </div>
      )}
    </PanelBody>
  );
}

/** Entering an agent should feel like entering its workspace: identity and state at the
 *  top, then only what is true about THIS agent. */
export function AgentDetail({
  agent, tasks, channels, knowledge, policy, budget, decisions, onBack,
}: {
  agent: Agent; tasks: Task[]; channels: Channel[];
  knowledge: KnowledgeSource[]; policy?: Policy; budget?: Budget;
  decisions: Decision[]; onBack: () => void;
}) {
  const [tab, setTab] = React.useState("work");
  const mine = tasks.filter((t) => t.agent_id === agent.id);
  const attention = mine.filter((t) => t.needs_attention);
  const running = mine.filter((t) => ["running", "ready"].includes(String(t.runtime_status)));
  const reaching = channels.filter(
    (c) => (c.allowed_agents ?? []).includes(agent.id)
      || (c.derived_agents ?? []).some((d: any) => d.base_agent === agent.id),
  );
  const corpora = knowledge.filter((k) => (k.readable_by ?? []).includes(agent.id));
  const governance = (policy?.agents ?? []).find((a) => a.id === agent.id);
  const usage = (budget?.observed ?? []).find((u) => u.agent_id === agent.id);
  const mineDecisions = decisions.filter((d) => d.agent_id === agent.id);

  const tabs = [
    { id: "work", label: "Work", count: mine.length },
    // First after Work: this is the tab the agent profile exists for. Everything else on
    // this screen reports what the agent did; this is where you change what it is.
    { id: "soul", label: "Soul" },
    { id: "knowledge", label: "Knowledge", count: corpora.length },
    { id: "channels", label: "Channels", count: reaching.length },
    { id: "permissions", label: "Permissions" },
    { id: "usage", label: "Usage" },
  ];

  return (
    <div className="space-y-5">
      <button
        type="button" onClick={onBack}
        className="text-ink-faint hover:text-ink inline-flex items-center gap-1.5 text-[12.5px] transition-colors"
      >
        <ArrowLeft className="size-3.5" /> All agents
      </button>

      <GlassPanel elevated className="p-5">
        <div className="flex flex-wrap items-start gap-4">
          <div className="glass-solid grid size-12 shrink-0 place-items-center rounded-xl text-[15px] font-semibold">
            {initials(agent.display_name ?? agent.id)}
          </div>
          <div className="min-w-0 flex-1">
            <h2 className="text-ink text-lg leading-tight font-semibold tracking-tight">
              {agent.display_name ?? agent.id}
            </h2>
            <p className="text-ink-faint mt-0.5 font-mono text-[11.5px]">{agent.id}</p>
            {agent.description ? (
              <p className="text-ink-muted mt-2 max-w-2xl text-[13px] leading-relaxed">
                {agent.description}
              </p>
            ) : null}
          </div>
          <div className="flex flex-col items-end gap-2">
            <StatusPill state={attention.length ? "waiting" : running.length ? "running" : "neutral"}>
              {attention.length ? `${attention.length} awaiting a human`
                : running.length ? `Working · ${plural(running.length, "task")}` : "Idle"}
            </StatusPill>
            {agent.in_sync === false ? <StatusPill state="blocked">Drifted from bundle</StatusPill> : null}
          </div>
        </div>
      </GlassPanel>

      <div role="tablist" aria-label="Agent sections" className="flex flex-wrap gap-1">
        {tabs.map((t) => (
          <button
            key={t.id} role="tab" type="button"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={`rounded-lg px-3 py-1.5 text-[12.5px] font-medium transition-colors ${
              tab === t.id ? "glass-solid text-ink" : "text-ink-muted hover:text-ink"
            }`}
          >
            {t.label}
            {t.count !== undefined ? <span className="text-ink-faint ml-1.5">{t.count}</span> : null}
          </button>
        ))}
      </div>

      {/* The Soul tab brings its own panel: it is an editor, not a list, and nesting it
          inside the shared one would double the border and the padding. */}
      {tab === "soul" ? <SoulEditor agentId={agent.id} /> : (
      <GlassPanel className="p-5">
        {tab === "work" ? (
          mine.length ? (
            <ul className="divide-glass-border divide-y">
              {mine.map((task) => (
                <li key={task.task_id} className="flex items-start gap-3 py-2.5 first:pt-0 last:pb-0">
                  <StatusPill state={taskState(task.runtime_status)} className="mt-0.5 w-[150px] justify-start">
                    {taskLabel(task.runtime_status)}
                  </StatusPill>
                  <div className="min-w-0 flex-1">
                    <p className="text-ink text-[13px] leading-snug">{task.title}</p>
                    {task.last_error ? (
                      <p className="text-blocked mt-0.5 text-[11.5px]">{task.last_error}</p>
                    ) : null}
                  </div>
                  <span className="text-ink-faint shrink-0 text-[11.5px]">{since(task.created_at)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState icon={Boxes} title="No work assigned"
              detail="This agent has nothing on the board. Work arrives when an objective is submitted or a channel routes a conversation here." />
          )
        ) : null}

        {tab === "knowledge" ? (
          corpora.length ? (
            <ul className="divide-glass-border divide-y">
              {corpora.map((source) => (
                <li key={source.id} className="flex items-center gap-3 py-2.5 first:pt-0 last:pb-0">
                  <div className="min-w-0 flex-1">
                    <p className="text-ink text-[13px] font-medium">{source.title ?? source.id}</p>
                    <p className="text-ink-faint truncate text-[11.5px]">{source.description}</p>
                  </div>
                  <Chip>{source.classification}</Chip>
                  {source.indexed
                    ? <Chip>{plural(Number(source.documents ?? 0), "document")}</Chip>
                    : <StatusPill state="waiting">Not indexed</StatusPill>}
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState icon={BookOpen} title="No corpora granted"
              detail="This agent cannot search any documents. Declaring a knowledge source in its spec is what grants the search tool." />
          )
        ) : null}

        {tab === "channels" ? (
          reaching.length ? (
            <ul className="divide-glass-border divide-y">
              {reaching.map((channel) => {
                const derived = (channel.derived_agents ?? []).find((d: any) => d.base_agent === agent.id);
                return (
                  <li key={channel.id} className="py-3 first:pt-0 last:pb-0">
                    <div className="flex items-center gap-3">
                      <span className="text-ink text-[13px] font-medium">
                        {channel.display_name ?? channel.provider_label}
                      </span>
                      <StatusPill state={channel.status === "connected" ? "running" : "waiting"}>
                        {channel.status === "connected" ? "Connected" : "Credential required"}
                      </StatusPill>
                    </div>
                    {derived ? (
                      <p className="text-ink-faint mt-1.5 text-[11.5px]">
                        <Hint text="The runtime's policy hook is never told which channel it is serving, so a tighter posture becomes its own profile with its own compiled policy.">
                          Runs here as
                        </Hint>{" "}
                        <span className="font-mono">{derived.id}</span>, additionally escalating{" "}
                        {(derived.added_approvals ?? []).join(", ")}
                      </p>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          ) : (
            <EmptyState icon={Boxes} title="Not reachable from any channel"
              detail="No connected channel grants this agent. It works only on objectives submitted inside NOVA." />
          )
        ) : null}

        {tab === "permissions" ? (
          governance ? (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-2">
                <StatusPill state={governance.unlisted_tool === "deny" ? "running" : "waiting"}>
                  {governance.unlisted_tool === "deny" ? "Default deny" : "Default allow"}
                </StatusPill>
                <InfoDot text={governance.unlisted_tool === "deny"
                  ? "A tool this agent was not granted is refused before it executes."
                  : "Anything not explicitly denied is permitted for this agent."} />
              </div>
              <PermissionList label="Allowed" tone="running" items={governance.allow ?? []} />
              <PermissionList label="Denied" tone="blocked" items={governance.deny ?? []}
                note="Deny is last-word: it wins over any allowlist or toolset." />
              <PermissionList label="Needs a human" tone="waiting" items={governance.approval_actions ?? []}
                note="Escalated to the same gate that guards dangerous shell commands, which fails closed when nobody is present." />
            </div>
          ) : (
            <EmptyState icon={ShieldCheck} title="No compiled policy"
              detail="Either no policy is declared for this tenant, or your role cannot read governance surfaces." />
          )
        ) : null}

        {tab === "usage" ? (
          usage?.available ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                <Figure label="Tokens" value={Number(usage.total_tokens ?? 0).toLocaleString()} />
                <Figure label="API calls" value={Number(usage.api_calls ?? 0).toLocaleString()} />
                <Figure label="Recorded cost"
                        value={`$${Number(usage.estimated_cost_usd ?? 0).toFixed(2)}`} />
              </div>
              <p className="text-ink-faint border-waiting/40 border-l-2 pl-3 text-[12px] leading-relaxed">
                <Hint text="This runtime's model-boundary hooks discard their return values, so a token or cost ceiling cannot be enforced here. Set the limit in your provider's own console.">
                  Observed, never enforced.
                </Hint>{" "}
                {usage.detail}
              </p>
            </div>
          ) : (
            <EmptyState icon={Boxes} title="No usage recorded"
              detail={usage?.detail || "This agent has not made a model call that the runtime recorded."} />
          )
        ) : null}
      </GlassPanel>
      )}

      {mineDecisions.length ? (
        <GlassPanel className="p-5">
          <SectionHeader title="Recent decisions" detail="What the policy layer allowed, escalated or refused for this agent." />
          <ul className="divide-glass-border divide-y">
            {mineDecisions.slice(0, 8).map((d, i) => (
              <li key={i} className="flex items-center gap-3 py-2 first:pt-0 last:pb-0">
                <span className="text-ink font-mono text-[12px]">{String(d.tool ?? "—")}</span>
                <span className="text-ink-faint ml-auto text-[11.5px]">{decisionLabel(String(d.effect))}</span>
              </li>
            ))}
          </ul>
        </GlassPanel>
      ) : null}
    </div>
  );
}

function PermissionList({
  label, items, tone, note,
}: { label: string; items: string[]; tone: "running" | "blocked" | "waiting"; note?: string }) {
  if (!items.length) return null;
  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <span className="text-ink-faint text-[11px] font-medium tracking-wide uppercase">{label}</span>
        {note ? <InfoDot text={note} /> : null}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {items.map((item) => (
          <span key={item}
            className={`rounded-md border px-2 py-0.5 font-mono text-[11.5px] ${
              tone === "running" ? "text-running border-running/25 bg-running/10"
              : tone === "blocked" ? "text-blocked border-blocked/25 bg-blocked/10"
              : "text-waiting border-waiting/25 bg-waiting/10"}`}>
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}

function Figure({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass-solid rounded-lg p-3">
      <div className="text-ink-faint text-[11px] tracking-wide uppercase">{label}</div>
      <div className="text-ink mt-1 text-lg font-semibold">{value}</div>
    </div>
  );
}
