"use client";

import { useState } from "react";
import { cn } from "../utils/cn";
import { LEARNED_THIS_WEEK, MEMORY_CLUSTERS, MEMORY_EDGES, MEMORY_NODES, type MemoryNode } from "../data/memory";
import { useApp } from "../lib/app";
import { IconBrain, IconExternal, IconPencil, IconPin, IconSpark, IconTrash, IconX } from "./Icons";
import { Badge, Button, EmptyState, IconButton, Ring, Select, Stat } from "./ui";
import { Grid, PageHeader } from "./Titlebar";

/* ====================================================================
   MEMORY GRAPH — what Aro learned this week.
   The agent's self-curated memory is the moat: every session teaches
   it something about the user and the work. This view makes the loop
   visible — digest strip on top, association graph in the middle,
   provenance and controls on the right.
   ==================================================================== */

const PROVIDERS = [
  { value: "builtin", label: "Built-in store" },
  { value: "honcho", label: "Honcho" },
  { value: "mem0", label: "mem0" },
  { value: "supermemory", label: "supermemory" },
];

const MAX_RECALLS = Math.max(...MEMORY_NODES.map((n) => n.recalls));
const LABEL_THRESHOLD = 30; // nodes recalled this often keep their label rendered
const EMERALD = "#34d399"; // brand emerald — marks "learned this week"

const radius = (n: MemoryNode) => 8 + 10 * (n.recalls / MAX_RECALLS);
const clusterById = new Map(MEMORY_CLUSTERS.map((c) => [c.id, c]));
const confTone = (v: number) => (v >= 90 ? "var(--color-mint)" : v >= 80 ? "var(--color-iris)" : "var(--color-amber)");

const edgePath = (a: MemoryNode, b: MemoryNode) => {
  const cx = (a.x + b.x) / 2 - (b.y - a.y) * 0.14;
  const cy = (a.y + b.y) / 2 + (b.x - a.x) * 0.14;
  return `M ${a.x} ${a.y} Q ${cx.toFixed(1)} ${cy.toFixed(1)} ${b.x} ${b.y}`;
};

export function MemoryView() {
  const { toast } = useApp();
  const [nodes, setNodes] = useState(MEMORY_NODES);
  const [sel, setSel] = useState<string | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const [provider, setProvider] = useState("builtin");

  const byId = new Map(nodes.map((n) => [n.id, n]));
  const selected = sel ? (byId.get(sel) ?? null) : null;
  const learned = nodes.filter((n) => n.learnedThisWeek);
  const lowConfidence = nodes.filter((n) => n.confidence < 85).length;
  const digest = LEARNED_THIS_WEEK.filter((l) => byId.has(l.id));
  const connected = selected
    ? MEMORY_EDGES.filter(([a, b]) => a === selected.id || b === selected.id)
        .map(([a, b]) => byId.get(a === selected.id ? b : a))
        .filter((m): m is MemoryNode => !!m)
    : [];

  const togglePin = (id: string) => setNodes((all) => all.map((n) => (n.id === id ? { ...n, pinned: !n.pinned } : n)));
  const forget = (id: string) => {
    setNodes((all) => all.filter((n) => n.id !== id));
    setSel(null);
    toast("Memory forgotten — Aro won't recall it", "rose");
  };

  return (
    <div className="scroll-thin flex flex-1 flex-col overflow-y-auto">
      <PageHeader eyebrow="memory" title="What Aro knows" sub="Agent-curated memory with user modeling — every session teaches it. Nothing leaves the workspace."
        right={<div className="flex flex-wrap items-center gap-2">
          <Select value={provider} onChange={(v) => { setProvider(v); toast(`Memory provider switched to ${PROVIDERS.find((p) => p.value === v)?.label ?? v}`, "iris"); }} options={PROVIDERS} />
          <Button variant="secondary" icon={IconExternal} onClick={() => toast(`Exported ${nodes.length} memories to aro-memory.jsonl`, "mint")}>Export</Button>
          <Button variant="primary" icon={IconSpark} onClick={() => toast(`Curator queued ${lowConfidence} low-confidence memories for review`, "amber")}>Curate now</Button>
        </div>} />

      {/* ---- digest strip ---- */}
      <div className="space-y-3 border-b border-line-soft p-4">
        <Grid className="grid-cols-2 lg:grid-cols-4">
          <Stat label="learned this week" value={learned.length} sub={`${lowConfidence} queued for curation`} tone="mint" />
          <Stat label="memories total" value={nodes.length} sub={`across ${MEMORY_CLUSTERS.length} clusters`} />
          <Stat label="recall rate" value="34%" sub="turns that pulled a memory" />
          <Stat label="sessions contributing" value="41" sub="last 30 days" />
        </Grid>
        <div className="scroll-thin flex items-center gap-2 overflow-x-auto pb-0.5">
          <span className="shrink-0 font-mono text-[9px] font-semibold tracking-[.14em] text-ink-4 uppercase">this week</span>
          {digest.map((l) => (
            <button key={l.id} title={l.detail} onClick={() => setSel(l.id)} aria-pressed={sel === l.id}
              className={cn("flex shrink-0 cursor-pointer items-center gap-2 rounded-[10px] border px-2.5 py-[7px] transition-colors", sel === l.id ? "border-mint/50 bg-mint-tint/70" : "border-mint/20 bg-mint-tint/40 hover:border-mint/40 hover:bg-mint-tint/70")}>
              <i className="size-[5px] shrink-0 rounded-full" style={{ background: EMERALD }} />
              <span className="text-[11.5px] font-medium whitespace-nowrap text-ink">{l.label}</span>
              <span className="font-mono text-[9.5px] whitespace-nowrap text-ink-4">{l.when}</span>
            </button>
          ))}
          {digest.length === 0 && <span className="text-[11px] text-ink-4">nothing new this week</span>}
        </div>
      </div>

      {/* ---- graph + detail ---- */}
      <div className="flex min-h-[320px] flex-1 flex-col lg:flex-row">
        <div className="flex min-h-[240px] min-w-0 flex-1 flex-col">
          <div className="relative min-h-0 flex-1 bg-code">
            <div className="grid-bg pointer-events-none absolute inset-0" />
            <svg viewBox="0 0 900 560" role="group" aria-label="Memory graph — what Aro knows" className="relative h-full w-full select-none">
              {/* cluster hulls */}
              {MEMORY_CLUSTERS.map((c) => {
                const pts = nodes.filter((n) => n.clusterId === c.id);
                if (pts.length === 0) return null;
                const x1 = Math.min(...pts.map((n) => n.x - radius(n))) - 26;
                const y1 = Math.min(...pts.map((n) => n.y - radius(n))) - 30;
                const x2 = Math.max(...pts.map((n) => n.x + radius(n))) + 26;
                const y2 = Math.max(...pts.map((n) => n.y + radius(n))) + 36;
                return (
                  <g key={c.id}>
                    <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} rx={22} fill={`var(--color-${c.tone})`} fillOpacity={0.08} stroke={`var(--color-${c.tone})`} strokeOpacity={0.32} strokeDasharray="5 6" />
                    <text x={x1 + 13} y={y1 + 18} fontSize={9.5} letterSpacing="0.14em" className="font-mono" fill={`var(--color-${c.tone})`}>{c.label.toUpperCase()}</text>
                  </g>
                );
              })}

              {/* associations */}
              {MEMORY_EDGES.map(([a, b]) => {
                const na = byId.get(a), nb = byId.get(b);
                if (!na || !nb) return null;
                const hot = sel === a || sel === b;
                return <path key={`${a}~${b}`} d={edgePath(na, nb)} fill="none" stroke={hot ? "var(--color-iris)" : "var(--color-line-strong)"} strokeOpacity={hot ? 0.85 : 0.35} strokeWidth={hot ? 1.6 : 1} strokeLinecap="round" />;
              })}

              {/* memories */}
              {nodes.map((n) => {
                const r = radius(n);
                const on = sel === n.id;
                const hot = hover === n.id || on;
                const c = clusterById.get(n.clusterId)!;
                return (
                  <g key={n.id} role="button" tabIndex={0} aria-pressed={on} aria-label={`${n.label} — ${n.detail}`}
                    transform={`translate(${n.x} ${n.y})`}
                    className="group/node cursor-pointer outline-none"
                    onClick={() => setSel(on ? null : n.id)}
                    onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setSel(on ? null : n.id); } }}
                    onMouseEnter={() => setHover(n.id)} onMouseLeave={() => setHover(null)}>
                    <title>{`${n.label} — ${n.detail}`}</title>
                    <g className="transition-transform duration-150" style={hot ? { transform: "scale(1.12)" } : undefined}>
                      {on && <circle r={r + 8} fill="none" stroke="var(--color-iris)" strokeWidth={7} strokeOpacity={0.16} />}
                      <circle r={r} fill={`var(--color-${c.tone})`} stroke="var(--color-base)" strokeWidth={2} />
                      {n.learnedThisWeek && <circle r={r + 4.5} fill="none" stroke={EMERALD} strokeWidth={2} strokeDasharray="3.5 3.5" />}
                      {n.pinned && <circle cx={r * 0.72} cy={-r * 0.72} r={2.4} fill="var(--color-ink)" stroke="var(--color-base)" strokeWidth={1} />}
                    </g>
                    <circle r={r + 8} fill="none" stroke="var(--color-iris)" strokeWidth={1.5} className="opacity-0 transition-opacity duration-150 group-focus-visible/node:opacity-70" />
                    <text y={r + 14} textAnchor="middle" fontSize={11} fill="var(--color-ink-2)"
                      className={cn("pointer-events-none transition-opacity duration-150", n.recalls >= LABEL_THRESHOLD || hot ? "opacity-100" : "opacity-0")}>
                      {n.label}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* legend */}
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-line-soft px-4 py-2">
            {MEMORY_CLUSTERS.map((c) => (
              <span key={c.id} className="flex items-center gap-1.5 font-mono text-[9.5px] text-ink-3"><i className="size-[6px] rounded-full" style={{ background: `var(--color-${c.tone})` }} />{c.label}</span>
            ))}
            <span className="flex items-center gap-1.5 font-mono text-[9.5px] text-ink-3">
              <svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><circle cx="6.5" cy="6.5" r="4" fill="none" stroke={EMERALD} strokeWidth="1.6" strokeDasharray="2.5 2.5" /></svg>
              emerald ring = learned this week
            </span>
            <span className="ml-auto flex items-center gap-1.5 font-mono text-[9.5px] text-ink-4">
              <svg width="22" height="10" viewBox="0 0 22 10" aria-hidden="true"><circle cx="4" cy="5" r="2.4" fill="var(--color-ink-4)" /><circle cx="15" cy="5" r="4.6" fill="var(--color-ink-4)" /></svg>
              size = recall count
            </span>
          </div>
        </div>

        {/* ---- detail panel ---- */}
        <aside className="flex w-full shrink-0 flex-col border-t border-line-soft bg-sunken lg:w-[340px] lg:border-t-0 lg:border-l">
          {selected ? (
            <>
              <div className="flex items-center gap-2 border-b border-line-soft px-3 py-2.5">
                <Badge tone={clusterById.get(selected.clusterId)!.tone} mono className="text-[9px]">{clusterById.get(selected.clusterId)!.label}</Badge>
                {selected.learnedThisWeek && <Badge tone="mint" mono className="text-[9px]">new this week</Badge>}
                <IconButton icon={IconX} label="Close" size={24} className="ml-auto" onClick={() => setSel(null)} />
              </div>
              <div className="scroll-thin flex-1 space-y-3 overflow-y-auto p-3">
                <div>
                  <div className="flex items-start gap-2">
                    {selected.pinned && <IconPin size={12} className="mt-[3px] shrink-0 text-iris-soft" />}
                    <h3 className="text-[14.5px] leading-[1.35] font-semibold text-ink">{selected.label}</h3>
                  </div>
                  <p className="mt-1.5 text-[12px] leading-[1.65] text-ink-3">{selected.detail}</p>
                </div>

                <div className="rounded-[9px] border border-line-soft bg-raise p-2.5">
                  <div className="font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">recall</div>
                  <div className="mt-2 flex items-center gap-3">
                    <div className="min-w-0 flex-1 space-y-1.5">
                      <div className="flex items-baseline justify-between"><span className="text-[11px] text-ink-3">times recalled</span><span className="font-mono text-[12px] text-ink">{selected.recalls}</span></div>
                      <div className="flex items-baseline justify-between"><span className="text-[11px] text-ink-3">last recall</span><span className="font-mono text-[12px] text-ink">{selected.lastRecall}</span></div>
                      <div className="flex items-baseline justify-between"><span className="text-[11px] text-ink-3">pinned</span><span className="font-mono text-[12px] text-ink">{selected.pinned ? "yes" : "no"}</span></div>
                    </div>
                    <Ring value={selected.confidence} size={48} stroke={4} tone={confTone(selected.confidence)}><span className="font-mono text-[10px] font-semibold text-ink-2">{selected.confidence}</span></Ring>
                  </div>
                </div>

                <div className="rounded-[9px] border border-line-soft bg-raise p-2.5">
                  <div className="font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">source session</div>
                  <code className="mt-1.5 block font-mono text-[11px] text-iris-soft">{selected.source}</code>
                  <div className="mt-1 font-mono text-[9.5px] text-ink-4">learned {selected.learnedThisWeek ? "this week" : "earlier"} · {selected.confidence < 85 ? "curator watching" : "corroborated"}</div>
                </div>

                <div className="rounded-[9px] border border-line-soft bg-raise p-2.5">
                  <div className="font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">associated with</div>
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    {connected.map((m) => <button key={m.id} onClick={() => setSel(m.id)} className="cursor-pointer rounded-[5px] border border-line-soft bg-well px-1.5 py-[2px] text-[11px] text-ink-2 transition-colors hover:border-line-strong hover:text-ink">{m.label}</button>)}
                    {connected.length === 0 && <span className="text-[11px] text-ink-4">No associations yet.</span>}
                  </div>
                </div>
              </div>
              <div className="space-y-1.5 border-t border-line-soft p-3">
                <Button variant="primary" className="w-full" icon={IconPin} onClick={() => { togglePin(selected.id); toast(selected.pinned ? "Memory unpinned — Aro may age it out" : "Memory pinned — kept in every session's context", selected.pinned ? "amber" : "mint"); }}>{selected.pinned ? "Unpin memory" : "Pin memory"}</Button>
                <div className="flex gap-1.5">
                  <Button variant="outline" size="xs" className="flex-1" icon={IconPencil} onClick={() => toast("Memory editor opened — the curator re-verifies after edits", "iris")}>Edit</Button>
                  <Button variant="ghost" size="xs" className="flex-1 text-rose" icon={IconTrash} onClick={() => forget(selected.id)}>Forget</Button>
                </div>
              </div>
            </>
          ) : (
            <div className="flex flex-1 flex-col justify-center">
              <EmptyState icon={IconBrain} title="Select a memory" body="Click any node to inspect what Aro remembers, where it came from, and how often it's recalled." />
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
