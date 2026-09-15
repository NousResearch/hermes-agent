import * as React from "react";
import {
  Activity, Blocks, BookOpen, Boxes, CircleCheck, Gauge, LayoutDashboard,
  ListChecks, ShieldCheck, Target, Plus } from "lucide-react";
import { GlassPanel, SectionHeader, StatusPill } from "@/components/glass";
import { MetricCard, Panel, PanelBody } from "@/components/panel";
import { Atmosphere, CommandBar, NAV, Sidebar, TopBar } from "@/components/shell";
import { TooltipProvider } from "@/components/tooltip";
import { AgentDetail, AgentsScreen } from "@/screens/agents";
import { CreateAgent } from "@/screens/agent-create";
import { SettingsScreen } from "@/screens/settings";
import {
  ActivityScreen, ApprovalsScreen, ChannelsScreen, KnowledgeScreen,
  ObjectivesScreen, PoliciesScreen, UsageScreen, WorkScreen,
} from "@/screens/misc";
import type {
  Agent, AutomationsPayload, Budget, Channel, Decision, Health, Identity, KnowledgeSource,
  Objective, Policy, Task,
} from "@/screens/types";
import { AutomationsScreen } from "@/screens/automations";
import { plural } from "@/lib/api";
import { usePanel, useRoute, useTheme } from "@/lib/hooks";

/** How many agents the Overview samples. Six fills two rows of the three-column preview
 *  at desktop width without pushing the panels beside it off the fold. */
const OVERVIEW_AGENTS = 6;

const SCREEN_META: Record<string, { title: string; subtitle: string }> = {
  overview: { title: "Overview", subtitle: "The state of the whole workforce, at a glance." },
  settings: { title: "Settings", subtitle: "Who this deployment serves, and what the workforce is called." },
  agents: { title: "Agents", subtitle: "Every AI worker, what it is doing, and what it may reach." },
  objectives: { title: "Objectives", subtitle: "Repeatable business processes and how far each has got." },
  work: { title: "Work", subtitle: "Everything on the board, newest first." },
  approvals: { title: "Approvals", subtitle: "What is waiting on a person, and what always will be." },
  automations: {
    title: "Automations",
    subtitle: "Recurring work the runtime holds, and whether anything is running it.",
  },
  activity: { title: "Activity", subtitle: "Every refusal and escalation the policy layer recorded. Permitted calls are not logged." },
  knowledge: { title: "Knowledge", subtitle: "The documents the workforce may quote, and who may read each." },
  channels: { title: "Channels", subtitle: "The places customers already talk, wired to the workforce." },
  policies: { title: "Policies", subtitle: "What each agent is permitted to do, and what is actually enforced." },
  usage: { title: "Usage", subtitle: "Model usage as reported by the runtime. Observed, never enforced." },
};

export default function App() {
  const { theme, toggle } = useTheme();
  const [route, go] = useRoute();
  const [commandOpen, setCommandOpen] = React.useState(false);

  // Bumped after any agent write so the list, the sidebar count and the open profile all
  // follow without waiting out the poll interval. Declared before the reads that use it:
  // `const` is not hoisted, and referencing it earlier throws at render.
  const [agentNonce, setAgentNonce] = React.useState(0);
  // Re-reads branding after a settings save, so the shell reflects it at once.
  const [identityNonce, setIdentityNonce] = React.useState(0);
  // Re-reads the corpora after a document is added or removed, so the counts follow.
  const [knowledgeNonce, setKnowledgeNonce] = React.useState(0);
  const identity = usePanel<Identity>("/identity", 60000, identityNonce);
  const health = usePanel<Health>("/health");
  const agents = usePanel<{ agents: Agent[] }>("/agents", 15000, agentNonce);
  const tasks = usePanel<{ tasks: Task[]; counts?: Record<string, number> }>("/tasks?limit=200");
  const objectives = usePanel<{ objectives: Objective[]; detail?: string }>("/objectives");
  const knowledge = usePanel<{
    retrieval_enabled: boolean; sources: KnowledgeSource[]; undeclared_in_index?: string[];
    index_detail?: string; document_extraction?: boolean;
  }>("/knowledge", 15000, knowledgeNonce);
  const channels = usePanel<{ declared: boolean; channel_delivery: boolean; channels: Channel[]; catalogue: any[] }>("/channels");
  const policy = usePanel<Policy>("/policy");
  // Bumped after a pause/resume so the list reflects the runtime at once.
  const [automationNonce, setAutomationNonce] = React.useState(0);
  const automations = usePanel<AutomationsPayload>("/automations", 15000, automationNonce);
  const decisions = usePanel<{ decisions: Decision[]; total?: number; counts?: Record<string, number> }>(
    // total and counts describe the whole log, not this page of it, so the timeline
    // can say how much it is not showing rather than implying 80 is all there is.
    "/decisions?limit=80",
  );
  const budget = usePanel<Budget>("/budget");

  const brand = identity.state === "ok" ? identity.data : null;
  const agentRows = agents.state === "ok" ? agents.data.agents : [];
  const taskRows = tasks.state === "ok" ? tasks.data.tasks : [];
  const channelRows = channels.state === "ok" ? channels.data.channels : [];
  const knowledgeRows = knowledge.state === "ok" ? knowledge.data.sources : [];
  const objectiveRows = objectives.state === "ok" ? objectives.data.objectives : [];
  const decisionRows = decisions.state === "ok" ? decisions.data.decisions : [];

  const attention = taskRows.filter((t) => t.needs_attention);
  const running = taskRows.filter((t) => ["running", "ready"].includes(String(t.runtime_status)));
  const connected = channelRows.filter((c) => c.status === "connected");

  React.useEffect(() => {
    if (brand?.product_name) document.title = `${brand.product_name} — Control Center`;
  }, [brand?.product_name]);

  // The tenant's declared colours, applied to the tokens the whole interface is built from,
  // so branding reaches every surface rather than only the header. Set as inline custom
  // properties on :root: they override the stylesheet's defaults in both themes without a
  // rebuild, and removing them restores NOVA's own palette exactly.
  //
  // A declared colour is written through as-is. NOVA does not validate colour notation —
  // the same string is consumed by a terminal, a web page and an email, which accept
  // different notations — so an unparseable value is ignored by the browser and the default
  // shows through, which is the right failure for a cosmetic field.
  React.useEffect(() => {
    const root = document.documentElement;
    const theme = brand?.theme ?? {};
    const applied: [string, string | undefined][] = [
      ["--accent", theme.accent],
      ["--accent-ink", theme.on_accent],
    ];
    for (const [token, value] of applied) {
      if (value) root.style.setProperty(token, value);
      else root.style.removeProperty(token);
    }
    return () => {
      for (const [token] of applied) root.style.removeProperty(token);
    };
  }, [brand?.theme?.accent, brand?.theme?.on_accent]);

  // The tab icon, served from this origin so `img-src 'self'` needs no change.
  React.useEffect(() => {
    if (!brand?.favicon) return;
    const link = document.querySelector<HTMLLinkElement>("link[rel='icon']")
      ?? document.head.appendChild(Object.assign(document.createElement("link"), { rel: "icon" }));
    link.href = `/platform/v1/branding/favicon?v=${encodeURIComponent(brand.favicon)}`;
  }, [brand?.favicon]);

  // ⌘K / Ctrl-K opens the palette anywhere.
  React.useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setCommandOpen((open) => !open);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  /* The palette searches only what is already loaded. It never queries an endpoint that
     does not exist, and it never lists a record the current role could not read. */
  const commandItems = React.useMemo(() => [
    ...NAV.map((n) => ({ id: n.id, label: n.label, group: "Section" })),
    ...agentRows.map((a) => ({
      id: `agents/${a.id}`, label: a.display_name ?? a.id, group: "Agent", hint: a.role,
    })),
    ...objectiveRows.map((o) => ({
      id: "objectives", label: o.title ?? o.id, group: "Objective", hint: o.owner_display_name,
    })),
    ...channelRows.map((c) => ({
      id: "channels", label: c.display_name ?? c.id, group: "Channel", hint: c.provider_label,
    })),
  ], [agentRows, objectiveRows, channelRows]);

  const nav = NAV.map((item) =>
    item.id === "approvals" ? { ...item, count: attention.length, urgent: attention.length > 0 }
    : item.id === "agents" ? { ...item, count: agentRows.length }
    : item.id === "work" ? { ...item, count: taskRows.length }
    : item);

  const openAgent = route.startsWith("agents/") ? route.slice("agents/".length) : null;
  const creatingAgent = openAgent === "new";
  const activeAgent = openAgent && !creatingAgent ? agentRows.find((a) => a.id === openAgent) : null;
  const meta = SCREEN_META[route.split("/")[0]] ?? SCREEN_META.overview;

  return (
    <TooltipProvider delayDuration={140}>
      <Atmosphere />
      <a
        href="#main"
        className="glass-solid text-ink sr-only rounded-lg px-3 py-2 text-sm focus:not-sr-only focus:absolute focus:top-3 focus:left-3 focus:z-50"
      >
        Skip to content
      </a>

      <div className="flex min-h-screen">
        <aside className="border-glass-border sticky top-0 hidden h-screen w-[212px] shrink-0 border-r backdrop-blur-xl lg:block">
          <Sidebar route={route} go={go} items={nav}
                   tenant={brand?.tenant_id} product={brand?.product_name}
                   logo={brand?.logo ? "/platform/v1/branding/logo" : undefined} />
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          <TopBar
            title={activeAgent ? (activeAgent.display_name ?? activeAgent.id) : meta.title}
            subtitle={activeAgent ? "Agent workspace" : meta.subtitle}
            runtime={health.state === "ok" ? String(health.data.runtime?.runtime ?? "") : undefined}
            healthy={health.state === "ok" ? health.data.runtime?.reachable !== false : undefined}
            theme={theme} onToggleTheme={toggle} onOpenCommand={() => setCommandOpen(true)}
          />

          {/* Narrow viewports get the same sections as a scrollable rail rather than a
              hamburger: an operator on a tablet is still doing the desktop job. */}
          <div className="border-glass-border flex gap-1 overflow-x-auto border-b px-4 py-2 lg:hidden">
            {nav.map((item) => (
              <button
                key={item.id} type="button" onClick={() => go(item.id)}
                aria-current={route.startsWith(item.id) ? "page" : undefined}
                className={`shrink-0 rounded-lg px-2.5 py-1.5 text-[12.5px] font-medium transition-colors ${
                  route.startsWith(item.id) ? "glass-solid text-ink" : "text-ink-muted"}`}
              >
                {item.label}
              </button>
            ))}
          </div>

          <main id="main" className="mx-auto w-full max-w-[1400px] flex-1 px-6 py-6">
            {creatingAgent ? (
              <CreateAgent
                existing={agentRows.map((a) => a.id)}
                onCancel={() => go("agents")}
                onCreated={(id) => { setAgentNonce((n) => n + 1); go(`agents/${id}`); }}
              />
            ) : activeAgent ? (
              <AgentDetail
                agent={activeAgent} tasks={taskRows} channels={channelRows}
                knowledge={knowledgeRows}
                policy={policy.state === "ok" ? policy.data : undefined}
                budget={budget.state === "ok" ? budget.data : undefined}
                decisions={decisionRows} onBack={() => go("agents")}
                onChanged={() => setAgentNonce((n) => n + 1)}
              />
            ) : openAgent ? (
              // A URL naming an agent that is not in the list — deleted, renamed, or never
              // existed. Saying so beats rendering the list as if nothing was asked for.
              <GlassPanel className="p-5">
                <p className="text-ink text-[13px] font-medium">No agent “{openAgent}”.</p>
                <p className="text-ink-muted mt-1 text-[12.5px]">
                  It may have been deleted, or the link may be stale.
                </p>
                <button type="button" onClick={() => go("agents")}
                  className="glass-solid text-ink mt-3 rounded-lg px-3 py-1.5 text-[12.5px] font-medium">
                  All agents
                </button>
              </GlassPanel>
            ) : route === "overview" ? (
              <div className="space-y-6">
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  <MetricCard label="Working now" value={running.length} tone="running"
                    source={tasks.state}
                    caption={running.length ? "items running or ready to start" : "nothing running"}
                    onClick={() => go("work")}
                    hint="Items a worker is running or ready to pick up." />
                  <MetricCard label="Needs a human" value={attention.length}
                    source={tasks.state}
                    tone={attention.length ? "waiting" : "neutral"}
                    caption={attention.length ? "held until someone decides" : "nothing is held"}
                    onClick={() => go("approvals")}
                    hint="Work the runtime stopped and will not resume without a decision." />
                  <MetricCard label="Agents" value={agentRows.length} onClick={() => go("agents")}
                    source={agents.state}
                    caption={`${agentRows.filter((a) => a.in_sync !== false).length} in sync with the bundle`}
                    hint="Declared in the tenant bundle and materialized into the runtime." />
                  <MetricCard label="Channels live" value={connected.length} onClick={() => go("channels")}
                    source={channels.state}
                    caption={channelRows.length ? `of ${plural(channelRows.length, "connection")}` : "none connected"}
                    hint="Connected and holding every credential the provider needs." />
                </div>

                <div className="grid gap-5 xl:grid-cols-3">
                  <div className="xl:col-span-2">
                    {/* h-full: the grid row is as tall as the right-hand column, and a
                        panel that stops short of it reads as a rendering fault rather
                        than a deliberate edge. */}
                    <GlassPanel className="h-full p-5">
                      <SectionHeader title="The workforce" icon={Boxes}
                        detail="Every agent, what it is doing, and what it may reach."
                        action={
                          <button type="button" onClick={() => go("agents")}
                            className="text-ink-faint hover:text-ink text-[12px] transition-colors">
                            {agentRows.length > OVERVIEW_AGENTS
                              ? `View all ${agentRows.length}`
                              : "View all"}
                          </button>
                        } />
                      <AgentsScreen agents={agents} tasks={taskRows} channels={channelRows}
                                    limit={OVERVIEW_AGENTS}
                                    onOpen={(id) => go(`agents/${id}`)} />
                    </GlassPanel>
                  </div>

                  <div className="space-y-5">
                    <Panel title="Platform" icon={LayoutDashboard} state={health}
                           detail="What the platform and the runtime each say.">
                      {(data) => (
                        <dl className="space-y-2.5 text-[13px]">
                          <Row label="Platform" value={`v${data.platform.version}`} />
                          <Row label="Runtime" value={String(data.runtime?.runtime ?? "—")} />
                          <Row label="Agents in runtime" value={String(data.runtime?.agent_count ?? "—")} />
                          <Row label="Bundle" value={data.bundle.digest.replace("sha256:", "").slice(0, 12)} mono />
                        </dl>
                      )}
                    </Panel>

                    <Panel title="Needs a human" icon={CircleCheck} state={tasks}
                           detail="Held until someone decides."
                           empty={(d) => d.tasks.some((t) => t.needs_attention) ? null : {
                             title: "Nothing is waiting",
                             detail: "Your workforce is operating inside its permitted actions.",
                           }}>
                      {(data) => (
                        <ul className="divide-glass-border divide-y">
                          {data.tasks.filter((t) => t.needs_attention).slice(0, 5).map((task) => (
                            <li key={task.task_id} className="py-2 first:pt-0 last:pb-0">
                              <p className="text-ink line-clamp-2 text-[12.5px] leading-snug">{task.title}</p>
                              <p className="text-ink-faint mt-0.5 text-[11px]">{task.agent_id}</p>
                            </li>
                          ))}
                        </ul>
                      )}
                    </Panel>
                  </div>
                </div>

                <GlassPanel className="p-5">
                  <SectionHeader title="Channels" icon={Blocks}
                    detail="Where the workforce can be reached, and by whom." />
                  <ChannelsScreen channels={channels} />
                </GlassPanel>
              </div>
            ) : route === "agents" ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-ink-faint text-[12.5px]">
                    {agentRows.length ? `${agentRows.length} declared` : "None declared yet"}
                  </p>
                  <button
                    type="button" onClick={() => go("agents/new")}
                    className="glass-solid text-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium"
                  >
                    <Plus className="size-3.5" /> New agent
                  </button>
                </div>
                <AgentsScreen agents={agents} tasks={taskRows} channels={channelRows}
                              onOpen={(id) => go(`agents/${id}`)} />
              </div>
            ) : route === "settings" ? (
              // Bumping the identity nonce re-reads the branding the shell is drawn from,
              // so a saved colour or logo appears without a reload.
              <SettingsScreen onChanged={() => setIdentityNonce((n) => n + 1)} />
            ) : route === "objectives" ? <ObjectivesScreen objectives={objectives} />
            : route === "work" ? <WorkScreen tasks={tasks} />
            : route === "approvals" ? (
              <ApprovalsScreen tasks={taskRows} decisions={decisionRows} agents={agentRows}
                               channels={channelRows} canSeeDecisions={decisions.state === "ok"} />
            )
            : route === "automations" ? (
              <AutomationsScreen
                automations={automations}
                onChanged={() => setAutomationNonce((n) => n + 1)}
                onOpenAgent={(id) => go(`agents/${id}`)}
              />
            )
            : route === "activity" ? <ActivityScreen decisions={decisions} />
            : route === "knowledge" ? <KnowledgeScreen knowledge={knowledge} onChanged={() => setKnowledgeNonce((n) => n + 1)} />
            : route === "channels" ? <ChannelsScreen channels={channels} />
            : route === "policies" ? <PoliciesScreen policy={policy} />
            : route === "usage" ? <UsageScreen budget={budget} />
            : <ObjectivesScreen objectives={objectives} />}
          </main>

          <footer className="text-ink-faint mx-auto w-full max-w-[1400px] px-6 pt-2 pb-8 text-[11.5px]">
            <div className="border-glass-border flex flex-wrap items-center gap-x-4 gap-y-1 border-t pt-4">
              <span>{brand?.company_name ?? "NOVA"}</span>
              {brand?.support?.email ? <span>{brand.support.email}</span> : null}
              <span className="ml-auto">Read-only surfaces refresh every 15 seconds.</span>
            </div>
          </footer>
        </div>
      </div>

      <CommandBar open={commandOpen} onClose={() => setCommandOpen(false)}
                  items={commandItems} onPick={go} />
    </TooltipProvider>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-ink-faint">{label}</dt>
      <dd className={`text-ink truncate font-medium ${mono ? "font-mono text-[11.5px]" : ""}`}>{value}</dd>
    </div>
  );
}
