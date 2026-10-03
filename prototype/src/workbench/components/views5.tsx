"use client";

import { useState } from "react";
import { cn } from "../utils/cn";
import { AGENTS } from "../data/catalog";
import { ARTIFACTS, type Artifact, type ArtifactKind } from "../data/artifacts";
import { useApp } from "../lib/app";
import { AgentMark, IconArchive, IconExternal, IconPin, IconPlus, IconRefresh, IconSearch, IconSpark } from "./Icons";
import { Badge, Button, EmptyState, IconButton, Input, Segmented, Stat, Tip, type Tone } from "./ui";
import { Grid, PageHeader } from "./Titlebar";

/* ====================================================================
   ARTIFACTS — pinned outputs from agent sessions (P-F)
   Docs, decks, data, diagrams, sites. Anything an agent produces can
   be pinned; every card keeps its session provenance and regenerating
   is one prompt away.
   ==================================================================== */
type KindFilter = "all" | ArtifactKind;
const KIND_TONE: Record<ArtifactKind, Tone> = { doc: "neutral", deck: "iris", data: "cyan", diagram: "plum", site: "sky" };

/* ------------------------- CSS-drawn previews ------------------------- */
function ArtifactPreview({ kind }: { kind: ArtifactKind }) {
  return (
    <div className="relative h-[110px] overflow-hidden rounded-[8px] border border-line-soft bg-well">
      {kind === "doc" && (
        <div className="flex h-full items-center justify-center">
          <div className="w-[78px] space-y-[5px] rounded-[5px] border border-black/10 bg-[#eceae4] p-2.5 shadow-e2">
            <div className="mb-1.5 h-[5px] w-[64%] rounded-full bg-[#4d4b45]" />
            <div className="h-[3.5px] w-[95%] rounded-full bg-[#bab7ac]" />
            <div className="h-[3.5px] w-full rounded-full bg-[#bab7ac]" />
            <div className="h-[3.5px] w-full rounded-full bg-[#bab7ac]" />
            <div className="h-[3.5px] w-[72%] rounded-full bg-[#bab7ac]" />
            <div className="h-[3.5px] w-[45%] rounded-full bg-[#8f8d83]" />
          </div>
        </div>
      )}
      {kind === "deck" && (
        <div className="flex h-full items-center justify-center gap-1.5">
          {[0, 1, 2].map((i) => (
            <div key={i} className={cn("space-y-[4px] rounded-[5px] border bg-base p-2", i === 1 ? "h-[64px] w-[54px] border-iris/30 shadow-e2" : "h-[48px] w-[40px] border-line-soft opacity-50")}>
              <div className={cn("rounded-[2px]", i === 1 ? "h-[9px] bg-gradient-to-r from-[#6EE7B7] to-[#10B981]" : "h-[7px] bg-line-strong")} />
              <div className="h-[3px] w-[85%] rounded-full bg-line" />
              <div className="h-[3px] w-[60%] rounded-full bg-line" />
              {i === 1 && <div className="h-[3px] w-[75%] rounded-full bg-line" />}
            </div>
          ))}
        </div>
      )}
      {kind === "data" && (
        <div className="relative flex h-full items-end justify-center gap-[7px] px-8 pb-6">
          {[38, 62, 46, 80, 56].map((h, i) => (
            <div key={i} style={{ height: `${h}%` }} className={cn("w-[12px] rounded-t-[3px]", i % 2 ? "bg-cyan/50" : "bg-mint/55")} />
          ))}
          <div className="absolute inset-x-7 bottom-5 h-px bg-line-soft" />
        </div>
      )}
      {kind === "diagram" && (
        <svg viewBox="0 0 240 110" preserveAspectRatio="xMidYMid meet" className="h-full w-full">
          <path d="M76 55 C 102 55 104 30 130 30" fill="none" stroke="var(--color-line-strong)" strokeWidth="1.5" />
          <path d="M76 55 C 102 55 104 80 130 80" fill="none" stroke="var(--color-line-strong)" strokeWidth="1.5" />
          <rect x="28" y="37" width="48" height="36" rx="7" fill="var(--color-iris-tint)" stroke="var(--color-iris)" strokeOpacity="0.45" />
          <rect x="130" y="14" width="42" height="32" rx="6" fill="var(--color-raise)" stroke="var(--color-line-strong)" />
          <rect x="130" y="64" width="42" height="32" rx="6" fill="var(--color-raise)" stroke="var(--color-line-strong)" />
          <rect x="40" y="49" width="24" height="3" rx="1.5" fill="var(--color-iris)" opacity="0.65" />
          <rect x="40" y="57" width="16" height="3" rx="1.5" fill="var(--color-ink-4)" opacity="0.5" />
          <rect x="140" y="26" width="20" height="3" rx="1.5" fill="var(--color-ink-4)" opacity="0.5" />
          <rect x="140" y="76" width="20" height="3" rx="1.5" fill="var(--color-ink-4)" opacity="0.5" />
        </svg>
      )}
      {kind === "site" && (
        <div className="flex h-full flex-col">
          <div className="flex items-center gap-[5px] border-b border-line-soft bg-sunken px-2.5 py-[7px]">
            <i className="size-[5px] rounded-full bg-rose/70" />
            <i className="size-[5px] rounded-full bg-amber/70" />
            <i className="size-[5px] rounded-full bg-mint/70" />
            <div className="ml-2 h-[9px] flex-1 rounded-[3px] bg-hover" />
          </div>
          <div className="flex flex-1 flex-col items-center justify-center gap-[7px] px-6">
            <div className="h-[7px] w-[58%] rounded-full bg-ink/20" />
            <div className="h-[4px] w-[38%] rounded-full bg-line-strong/80" />
            <div className="mt-1.5 h-[13px] w-[54px] rounded-[4px] bg-gradient-to-r from-[#6EE7B7] to-[#10B981]" />
          </div>
        </div>
      )}
    </div>
  );
}

/* ------------------------------ card ------------------------------ */
function ArtifactCard({ art, pinned, onTogglePin, onRegen }: { art: Artifact; pinned: boolean; onTogglePin: () => void; onRegen: () => void }) {
  const { toast } = useApp();
  const agent = AGENTS.find((a) => a.id === art.session.agentId);
  return (
    <article className={cn("group flex flex-col rounded-[12px] border border-line-soft bg-raise p-2.5 transition-all duration-200 hover:-translate-y-px hover:border-line-strong hover:shadow-e3", !pinned && "opacity-[0.82] hover:opacity-100")}>
      <ArtifactPreview kind={art.kind} />
      <div className="flex min-w-0 flex-1 flex-col px-0.5 pt-2.5">
        <div className="flex items-center gap-1.5">
          <Badge tone={KIND_TONE[art.kind]} mono className="text-[9px]">{art.kind}</Badge>
          {pinned && <Tip label="Pinned — surfaces in the session header"><IconPin size={11} className="text-amber" /></Tip>}
        </div>
        <h3 className="mt-1.5 truncate text-[13px] font-medium text-ink">{art.title}</h3>
        <p className="mt-1 line-clamp-2 min-h-[32px] text-[11.5px] leading-[1.4] text-ink-3">{art.desc}</p>
        <div className="mt-2 flex items-center gap-1.5 font-mono text-[9.5px] text-ink-4">
          {agent && (
            <Tip label={art.session.title}>
              <span className="flex min-w-0 items-center gap-1">
                <AgentMark glyph={agent.glyph} from={agent.from} to={agent.to} size={13} />
                <span className="truncate text-ink-3">{agent.name}</span>
              </span>
            </Tip>
          )}
          <span className="shrink-0">·</span>
          <span className="shrink-0">{art.updated}</span>
          <span className="shrink-0">·</span>
          <span className="shrink-0">{art.size}</span>
        </div>
      </div>
      <div className="mt-2 flex items-center gap-1 border-t border-line-soft px-0.5 pt-2">
        <IconButton icon={IconExternal} label={`Open ${art.title}`} size={24} onClick={() => toast(`Opening ${art.title}`, "iris")} />
        <Button variant="ghost" size="xs" icon={IconRefresh} onClick={onRegen}>Regenerate</Button>
        <IconButton icon={IconPin} label={pinned ? `Unpin ${art.title}` : `Pin ${art.title}`} size={24} active={pinned} className={cn("ml-auto", pinned && "text-amber")} onClick={() => { toast(pinned ? `${art.title} unpinned` : `${art.title} pinned`, pinned ? "iris" : "amber"); onTogglePin(); }} />
      </div>
    </article>
  );
}

/* ------------------------------ view ------------------------------ */
export function ArtifactsView() {
  const { toast } = useApp();
  const [q, setQ] = useState("");
  const [kind, setKind] = useState<KindFilter>("all");
  const [pinned, setPinned] = useState<Set<string>>(() => new Set(ARTIFACTS.filter((a) => a.pinned).map((a) => a.id)));
  const [regen, setRegen] = useState(0);

  const list = ARTIFACTS.filter((a) =>
    (kind === "all" || a.kind === kind) &&
    (a.title + a.desc + a.session.title).toLowerCase().includes(q.toLowerCase()),
  );
  const pinnedCount = ARTIFACTS.filter((a) => pinned.has(a.id)).length;
  const thisWeek = ARTIFACTS.filter((a) => !a.updated.includes("w")).length;

  const togglePin = (a: Artifact) => setPinned((s) => {
    const next = new Set(s);
    if (next.has(a.id)) next.delete(a.id); else next.add(a.id);
    return next;
  });

  return (
    <div className="scroll-thin flex-1 overflow-y-auto">
      <PageHeader eyebrow="outputs" title="Artifacts" sub="Pinned outputs from agent sessions — docs, decks, data, diagrams. Anything an agent produces can be pinned here and regenerating is one prompt away."
        right={<div className="flex flex-wrap items-center gap-2">
          <div className="relative"><IconSearch size={12} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-ink-4" /><Input placeholder="Search artifacts…" value={q} onChange={(e) => setQ(e.target.value)} className="w-[180px] pl-7" /></div>
          <Segmented value={kind} onChange={setKind} items={[{ value: "all", label: `All ${ARTIFACTS.length}` }, { value: "doc", label: "Docs" }, { value: "deck", label: "Decks" }, { value: "data", label: "Data" }, { value: "diagram", label: "Diagrams" }, { value: "site", label: "Sites" }]} />
          <Button variant="primary" icon={IconPlus} onClick={() => toast("Pin outputs from any thread — hover a result and click Pin", "iris")}>New from chat</Button>
        </div>} />

      <div className="space-y-4 p-5">
        <Grid className="grid-cols-2 lg:grid-cols-4">
          <Stat label="pinned" value={pinnedCount} sub="surface in session headers" tone="amber" />
          <Stat label="this week" value={thisWeek} sub="from 5 agent sessions" />
          <Stat label="storage" value="38 MB" sub="of 5 GB · kept local" tone="cyan" />
          <Stat label="regenerated" value={7 + regen} sub={`this week · $${(0.42 + regen * 0.06).toFixed(2)}`} tone="iris" />
        </Grid>

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {list.map((a) => (
            <ArtifactCard key={a.id} art={a} pinned={pinned.has(a.id)}
              onTogglePin={() => togglePin(a)}
              onRegen={() => { setRegen((r) => r + 1); toast("Regenerating from session context — $0.06", "iris"); }} />
          ))}
        </div>

        {list.length === 0 && (
          <div className="rounded-[12px] border border-line-soft bg-raise">
            <EmptyState icon={IconArchive} title="No artifacts match" body="Try another kind or clear the search — anything an agent outputs can be pinned here." />
          </div>
        )}

        <div className="flex items-start gap-2.5 rounded-[10px] border border-dashed border-line px-4 py-3">
          <IconSpark size={13} className="mt-0.5 shrink-0 text-iris-soft" />
          <p className="text-[12px] leading-[1.55] text-ink-3">Ask any agent to <span className="font-mono text-[11px] text-ink-2">“pin this as an artifact”</span> — it lands here with full session provenance.</p>
        </div>
      </div>
    </div>
  );
}
