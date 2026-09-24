/* The Team chart's layout, pure: agents and channels in, positioned nodes and edges out.
 *
 * Columns follow the way work arrives. Channels first; then the agents a channel routes to;
 * then, column by column, whoever those agents may hand work to (shortest route wins).
 * Agents nothing routes to and nobody hands work to start their own chains, and anything
 * left — reachable only through a cycle — goes in the last column. An edge that points
 * backwards or within a column (support ⇄ operations) is marked `back` so it is drawn as an
 * arc under the cards instead of through them. Kept free of React so it can be tested alone.
 */

export type ChartAgent = {
  id: string; display_name?: string; may_assign_to?: string[]; enabled?: boolean;
};
export type ChartChannel = {
  id: string; display_name?: string; routes?: { agent?: string }[]; allowed_agents?: string[];
};

export type ChartNode = {
  id: string; kind: "channel" | "agent"; label: string; column: number; row: number;
  x: number; y: number;
};
export type ChartEdge = {
  from: string; to: string; kind: "route" | "handoff"; back: boolean; mutual: boolean;
};
export type ChartLayout = {
  nodes: ChartNode[]; edges: ChartEdge[]; width: number; height: number;
  /** Hand-offs naming an agent this tenant does not have — shown, never dropped silently. */
  dangling: { from: string; to: string }[];
};

export const NODE_W = 232;
export const NODE_H = 104;
const COL_GAP = 96;
const ROW_GAP = 28;
const PAD = 24;

export function channelKey(id: string): string {
  return `channel:${id}`;
}

export function layoutTeam(agents: ChartAgent[], channels: ChartChannel[]): ChartLayout {
  const known = new Set(agents.map((a) => a.id));
  const edges: ChartEdge[] = [];
  const dangling: { from: string; to: string }[] = [];

  // Channel → agent: every agent a route names, falling back to the allowed list.
  const entry = new Set<string>();
  for (const channel of channels) {
    const targets = new Set(
      (channel.routes ?? []).map((r) => r.agent).filter((a): a is string => !!a && known.has(a)),
    );
    if (!targets.size) for (const a of channel.allowed_agents ?? []) if (known.has(a)) targets.add(a);
    for (const agent of targets) {
      entry.add(agent);
      edges.push({ from: channelKey(channel.id), to: agent, kind: "route", back: false, mutual: false });
    }
  }

  // Agent → agent hand-offs.
  const out = new Map<string, string[]>();
  const incoming = new Map<string, number>();
  for (const agent of agents) {
    const targets: string[] = [];
    for (const to of agent.may_assign_to ?? []) {
      if (to === agent.id) continue;
      if (!known.has(to)) { dangling.push({ from: agent.id, to }); continue; }
      targets.push(to);
      incoming.set(to, (incoming.get(to) ?? 0) + 1);
    }
    out.set(agent.id, targets);
  }

  // Columns: breadth-first from channel-facing agents, then from agents nobody reaches.
  const column = new Map<string, number>();
  const offset = channels.length ? 1 : 0;
  const walk = (starts: string[], base: number) => {
    let frontier = starts.filter((s) => !column.has(s));
    for (const s of frontier) column.set(s, base);
    let depth = base;
    while (frontier.length) {
      depth += 1;
      const next: string[] = [];
      for (const id of frontier) {
        for (const to of out.get(id) ?? []) {
          if (!column.has(to)) { column.set(to, depth); next.push(to); }
        }
      }
      frontier = next;
    }
  };
  walk(agents.filter((a) => entry.has(a.id)).map((a) => a.id), offset);
  walk(agents.filter((a) => !column.has(a.id) && !incoming.get(a.id)).map((a) => a.id), offset);
  const deepest = Math.max(offset, ...Array.from(column.values()));
  for (const a of agents) if (!column.has(a.id)) column.set(a.id, deepest + 1);

  // Rows within a column, in the tenant's own agent order.
  const nodes: ChartNode[] = [];
  const rowsIn = new Map<number, number>();
  const place = (id: string, kind: ChartNode["kind"], label: string, col: number) => {
    const row = rowsIn.get(col) ?? 0;
    rowsIn.set(col, row + 1);
    nodes.push({
      id, kind, label, column: col, row,
      x: PAD + col * (NODE_W + COL_GAP), y: PAD + row * (NODE_H + ROW_GAP),
    });
  };
  for (const channel of channels) place(channelKey(channel.id), "channel", channel.display_name ?? channel.id, 0);
  for (const agent of agents) place(agent.id, "agent", agent.display_name ?? agent.id, column.get(agent.id)!);

  const colOf = new Map(nodes.map((n) => [n.id, n.column]));
  const pairs = new Set<string>();
  for (const agent of agents) for (const to of out.get(agent.id) ?? []) pairs.add(`${agent.id}>${to}`);
  for (const agent of agents) {
    for (const to of out.get(agent.id) ?? []) {
      edges.push({
        from: agent.id, to, kind: "handoff",
        back: colOf.get(to)! <= colOf.get(agent.id)!,
        mutual: pairs.has(`${to}>${agent.id}`),
      });
    }
  }

  const columns = Math.max(0, ...nodes.map((n) => n.column)) + 1;
  const rows = Math.max(1, ...Array.from(rowsIn.values()));
  // Room under the cards only when a backward hand-off arcs there.
  const arcs = edges.some((e) => e.back && colOf.get(e.from) !== colOf.get(e.to));
  return {
    nodes, edges, dangling,
    width: PAD * 2 + columns * NODE_W + (columns - 1) * COL_GAP,
    height: PAD * 2 + rows * NODE_H + (rows - 1) * ROW_GAP + (arcs ? 56 : 0),
  };
}
