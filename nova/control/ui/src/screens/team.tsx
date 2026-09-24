import * as React from "react";
import { ArrowRight, Network, RadioTower, ShieldCheck, Wallet } from "lucide-react";
import { Chip, EmptyState, GlassPanel, SectionHeader, StatusPill } from "@/components/glass";
import { channelKey, layoutTeam, NODE_H, NODE_W, type ChartEdge, type ChartNode } from "@/lib/orgchart";
import { channelLabel, channelState } from "@/lib/state";
import type { Agent, Channel } from "./types";

/* The Team chart: how work arrives and who may pass it to whom.
 *
 * Every arrow is a rule NOVA enforces, not a diagram someone drew: a channel's route decides
 * which agent answers it, and an agent's hand-offs are its delegation.may_assign_to — refused
 * inside the worker when it is not declared. So the chart is the governance, drawn. The same
 * rules are listed in words underneath, for a screen reader and for a phone. */

export function TeamScreen({
  agents, channels, onOpen,
}: { agents: Agent[]; channels: Channel[]; onOpen: (id: string) => void }) {
  const declared = agents.filter((a) => (a as any).declared !== false);
  const live = channels.filter((c) => c.enabled !== false);
  const layout = React.useMemo(() => layoutTeam(declared, live), [declared, live]);
  const [focus, setFocus] = React.useState<string | null>(null);

  if (!declared.length) {
    return (
      <EmptyState icon={Network} title="No agents yet"
        detail="Agents and the hand-offs between them appear here once the bundle declares them." />
    );
  }

  const byId = new Map(layout.nodes.map((n) => [n.id, n]));
  const agentOf = new Map(declared.map((a) => [a.id, a]));
  const channelOf = new Map(live.map((c) => [channelKey(c.id), c]));
  const name = (id: string) => agentOf.get(id)?.display_name ?? id;
  const touches = (e: ChartEdge) => focus !== null && (e.from === focus || e.to === focus);
  // A two-way hand-off is one sentence, not two: list each pair once.
  const handoffs = layout.edges.filter(
    (e) => e.kind === "handoff" && !(e.mutual && e.from > e.to),
  );

  return (
    <div className="space-y-5">
      <GlassPanel className="p-5">
        <SectionHeader icon={Network} title="How work flows"
          detail="Channels on the left route conversations to agents; arrows between agents are the hand-offs each is allowed to make. Every arrow is enforced." />
        <Legend />
        {layout.width > 640 ? (
          <p className="text-ink-faint mt-3 text-[11.5px] sm:hidden">Scroll sideways to see the whole team.</p>
        ) : null}
        <FitToWidth width={layout.width} height={layout.height}>
          <div className="relative" style={{ width: layout.width, height: layout.height }}
               onMouseLeave={() => setFocus(null)}>
            <svg className="absolute inset-0" width={layout.width} height={layout.height} aria-hidden="true">
              <defs>
                <Arrowhead id="arrow-route" color="var(--accent)" />
                <Arrowhead id="arrow-handoff" color="var(--ink-muted)" />
                <Arrowhead id="arrow-focus" color="var(--accent)" />
              </defs>
              {layout.edges.map((edge, i) => {
                const from = byId.get(edge.from);
                const to = byId.get(edge.to);
                if (!from || !to) return null;
                const lit = touches(edge);
                const dim = focus !== null && !lit;
                const colour = edge.kind === "route" || lit ? "var(--accent)" : "var(--ink-muted)";
                return (
                  <path key={i} d={edgePath(edge, from, to, layout.height)} fill="none"
                    stroke={colour} strokeWidth={lit ? 2.25 : 1.5}
                    strokeDasharray={edge.kind === "route" ? undefined : "5 4"}
                    opacity={dim ? 0.18 : 0.9}
                    markerEnd={`url(#${lit ? "arrow-focus" : edge.kind === "route" ? "arrow-route" : "arrow-handoff"})`}
                    className="transition-opacity duration-200" />
                );
              })}
            </svg>

            {layout.nodes.map((node) => node.kind === "channel" ? (
              <ChannelCard key={node.id} node={node} channel={channelOf.get(node.id)} dim={focus !== null} />
            ) : (
              <AgentCard key={node.id} node={node} agent={agentOf.get(node.id)!}
                dim={focus !== null && focus !== node.id
                  && !layout.edges.some((e) => touches(e) && (e.from === node.id || e.to === node.id))}
                onFocus={() => setFocus(node.id)} onOpen={() => onOpen(node.id)} />
            ))}
          </div>
        </FitToWidth>
      </GlassPanel>

      <GlassPanel className="p-5">
        <SectionHeader icon={ArrowRight} title="Hand-offs in words" />
        {handoffs.length ? (
          <ul className="space-y-1.5">
            {handoffs.map((e) => (
              <li key={`${e.from}>${e.to}`} className="text-ink text-[13px]">
                <span className="font-medium">{name(e.from)}</span>
                <span className="text-ink-muted">{e.mutual ? " and " : " may hand work to "}</span>
                <span className="font-medium">{name(e.to)}</span>
                {e.mutual ? <span className="text-ink-muted"> may hand work to each other</span> : null}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-ink-muted text-[13px]">
            No agent may hand work to another. Each works only what it is given directly — set
            hand-offs in an agent's Capabilities.
          </p>
        )}
        {layout.dangling.length ? (
          <p className="text-blocked mt-3 text-[12.5px]">
            {layout.dangling.map((d) => `${name(d.from)} names ${d.to}, which is not an agent here`).join("; ")}.
          </p>
        ) : null}
      </GlassPanel>
    </div>
  );
}

/** Shrinks the chart to fit its panel — a whole team on one screen is the point of a demo —
 *  down to 70%, below which text would be too small and the chart scrolls sideways instead. */
function FitToWidth({ width, height, children }: { width: number; height: number; children: React.ReactNode }) {
  const ref = React.useRef<HTMLDivElement>(null);
  const [available, setAvailable] = React.useState(width);
  React.useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setAvailable(el.clientWidth);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  const scale = Math.min(1, available / width);
  const fits = scale >= 0.7;
  return (
    <div ref={ref} className={`-mx-1 mt-4 pb-2 ${fits ? "overflow-hidden" : "overflow-x-auto"}`}>
      {fits ? (
        <div style={{ width: width * scale, height: height * scale }}>
          <div style={{ width, height, transform: `scale(${scale})`, transformOrigin: "top left" }}>{children}</div>
        </div>
      ) : children}
    </div>
  );
}

function Arrowhead({ id, color }: { id: string; color: string }) {
  return (
    <marker id={id} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
    </marker>
  );
}

/** Forward edges curve from a card's right edge to the next card's left edge. A hand-off that
 *  points back (the return leg of support ⇄ operations) arcs under the cards; one within a
 *  column loops out to the right, so no line ever runs through a card. */
function edgePath(edge: ChartEdge, from: ChartNode, to: ChartNode, height: number): string {
  const lift = edge.mutual && !edge.back ? -10 : 0;
  if (!edge.back) {
    const x1 = from.x + NODE_W, y1 = from.y + NODE_H / 2 + lift;
    const x2 = to.x - 2, y2 = to.y + NODE_H / 2 + lift;
    const dx = Math.max(40, (x2 - x1) / 2);
    return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
  }
  if (from.column === to.column) {
    const x = from.x + NODE_W, y1 = from.y + NODE_H / 2, y2 = to.y + NODE_H / 2;
    return `M ${x} ${y1} C ${x + 70} ${y1}, ${x + 70} ${y2}, ${x + 2} ${y2}`;
  }
  const x1 = from.x + NODE_W / 2, y1 = from.y + NODE_H;
  const x2 = to.x + NODE_W / 2, y2 = to.y + NODE_H + 2;
  const floor = height - 14;
  return `M ${x1} ${y1} C ${x1} ${floor}, ${x2} ${floor}, ${x2} ${y2}`;
}

function ChannelCard({ node, channel, dim }: { node: ChartNode; channel?: Channel; dim: boolean }) {
  return (
    <div className={`glass-solid absolute flex flex-col justify-between rounded-xl border border-glass-border p-3 transition-opacity duration-200 ${dim ? "opacity-40" : ""}`}
         style={{ left: node.x, top: node.y, width: NODE_W, height: NODE_H }}>
      <div className="flex items-center gap-2">
        <RadioTower className="text-ink-muted size-4 shrink-0" />
        <span className="text-ink truncate text-[13px] font-semibold">{node.label}</span>
      </div>
      <div className="flex items-center justify-between gap-2">
        <span className="text-ink-faint truncate text-[11.5px]">{channel?.provider_label ?? "Channel"}</span>
        {channel ? (
          <StatusPill state={channelState(channel.status, channel.live?.state)}>
            {channelLabel(channel.status, channel.live?.state)}
          </StatusPill>
        ) : null}
      </div>
    </div>
  );
}

function AgentCard({
  node, agent, dim, onFocus, onOpen,
}: { node: ChartNode; agent: Agent; dim: boolean; onFocus: () => void; onOpen: () => void }) {
  const budget = agent.limits?.monthly_budget_usd as number | undefined;
  const approvals = agent.approval_required_for?.length ?? 0;
  const integrations = agent.integrations ?? [];
  return (
    <button type="button" onClick={onOpen} onMouseEnter={onFocus} onFocus={onFocus}
      aria-label={`${node.label}: open agent`}
      className={`glass absolute flex flex-col justify-between rounded-xl border border-glass-border p-3 text-left transition-[opacity,box-shadow] duration-200 hover:shadow-lg focus-visible:ring-2 focus-visible:ring-[var(--accent)] focus-visible:outline-none ${dim ? "opacity-40" : ""}`}
      style={{ left: node.x, top: node.y, width: NODE_W, height: NODE_H }}>
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-ink truncate text-[13.5px] font-semibold">{node.label}</span>
          {agent.enabled === false ? <StatusPill state="neutral" dot={false}>Not in service</StatusPill> : null}
        </div>
        <div className="text-ink-faint truncate text-[11.5px]">{agent.role?.replace(/_/g, " ") ?? agent.id}</div>
      </div>
      <div className="flex flex-nowrap items-center gap-1 overflow-hidden [&>*]:shrink-0 [&>*]:whitespace-nowrap">
        {(() => {
          // Governance first — approval and budget are what a buyer asks about — then the
          // systems the agent reaches, capped so a card never wraps onto a third line.
          const badges: React.ReactNode[] = [];
          if (approvals) badges.push(<Chip key="approval" className="gap-1"><ShieldCheck className="size-3" />approval</Chip>);
          if (budget) badges.push(<Chip key="budget" className="gap-1"><Wallet className="size-3" />${budget}/mo</Chip>);
          const room = Math.max(0, 3 - badges.length);
          integrations.slice(0, room).forEach((m) => badges.push(<Chip key={m}>{m}</Chip>));
          const hidden = integrations.length - Math.min(room, integrations.length);
          if (hidden > 0) badges.push(<Chip key="more">+{hidden}</Chip>);
          return badges;
        })()}
      </div>
    </button>
  );
}

function Legend() {
  return (
    <div className="text-ink-muted flex flex-wrap items-center gap-x-5 gap-y-1.5 text-[11.5px]">
      <span className="inline-flex items-center gap-2">
        <svg width="28" height="8" aria-hidden="true"><line x1="0" y1="4" x2="28" y2="4" stroke="var(--accent)" strokeWidth="1.5" /></svg>
        channel routes to agent
      </span>
      <span className="inline-flex items-center gap-2">
        <svg width="28" height="8" aria-hidden="true"><line x1="0" y1="4" x2="28" y2="4" stroke="var(--ink-muted)" strokeWidth="1.5" strokeDasharray="5 4" /></svg>
        may hand work to
      </span>
      <span className="inline-flex items-center gap-1.5"><ShieldCheck className="size-3" />asks a person before risky actions</span>
      <span className="inline-flex items-center gap-1.5"><Wallet className="size-3" />monthly budget</span>
    </div>
  );
}
