"use client";

import { useState, type ComponentType } from "react";
import { cn } from "../utils/cn";
import { MCP_SERVERS, PINNED_FILES, PLAN_TASKS, RULES, RUNS, agentById, type PlanTask, type Run } from "../data/catalog";
import { CHANGED_FILES } from "../data/extra";
import { useApp } from "../lib/app";
import { AgentMark, IconAt, IconBolt, IconBranch, IconCheck, IconChevronDown, IconEye, IconFile, IconFolder, IconGit, IconGrid, IconHistory, IconLayers, IconList, IconMcp, IconMonitor, IconPanelRight, IconPlug, IconPlus, IconRefresh, IconSearch, IconShield, IconSpark, IconTerminal, IconWarning, IconX } from "./Icons";
import { Badge, Bar, Button, IconButton, Ring, Toggle, Tip, toneText, type Tone } from "./ui";
import { DiffView } from "./Transcript";
import { REVIEW_FILES } from "./views";
import type { ViewId } from "./Sidebar";

/* ============================================================
   CONTEXT PANE — single Codex-style side surface
   Everything that used to be its own view (plan, changes,
   context, runs, bridges, browser preview, git summary, terminal
   output) collapses into one icon rail + body.
   ============================================================ */

export type ContextSectionId = "plan" | "changes" | "context" | "runs" | "git" | "browser" | "editors" | "terminal";

const SECTIONS: { id: ContextSectionId; label: string; icon: ComponentType<{ size?: number; className?: string }>; keywordView?: ViewId; count?: (ctx: { changed: number; plan: number; runs: number }) => number | undefined }[] = [
  { id: "plan", label: "Plan", icon: IconList, count: (c) => c.plan },
  { id: "changes", label: "Changes", icon: IconGrid, keywordView: "review", count: (c) => c.changed },
  { id: "context", label: "Context", icon: IconFolder },
  { id: "runs", label: "Fleet", icon: IconLayers, keywordView: "runs", count: (c) => c.runs },
  { id: "git", label: "Git", icon: IconGit, keywordView: "git" },
  { id: "browser", label: "Browser", icon: IconEye, keywordView: "browser" },
  { id: "editors", label: "Editors", icon: IconMonitor, keywordView: "bridges" },
  { id: "terminal", label: "Terminal", icon: IconTerminal },
];

/* ---------- plan ---------- */
const stateMeta: Record<PlanTask["state"], { tone: Tone; icon: ComponentType<{ size?: number }>; label: string }> = {
  done: { tone: "mint", icon: IconCheck, label: "done" }, active: { tone: "cyan", icon: IconRefresh, label: "active" }, todo: { tone: "neutral", icon: IconAt, label: "queued" }, blocked: { tone: "rose", icon: IconWarning, label: "blocked" },
};

function PlanSection() {
  const [openTask, setOpenTask] = useState<string | null>("p3");
  const done = PLAN_TASKS.filter((t) => t.state === "done").length;
  return (
    <div className="space-y-3 p-3">
      <div className="rounded-[10px] border border-line-soft bg-raise p-3">
        <div className="flex items-center justify-between"><span className="font-display text-[12.5px] font-semibold text-ink">Implementation plan</span><Badge tone="iris" mono className="text-[9.5px]">v3 · editable</Badge></div>
        <p className="mt-1 text-[11px] text-ink-3">Agent-maintained outline. Edit, reorder or hand steps to another agent.</p>
        <div className="mt-2.5 flex items-center gap-2"><Bar value={(done / PLAN_TASKS.length) * 100} /><span className="shrink-0 font-mono text-[10px] text-ink-4">{done}/{PLAN_TASKS.length}</span></div>
        <div className="mt-2.5 flex gap-1.5"><Button variant="primary" size="xs" icon={IconBolt} className="flex-1">Build plan</Button><Button variant="outline" size="xs">Export .md</Button></div>
      </div>
      <div className="space-y-1">
        {PLAN_TASKS.map((t) => {
          const meta = stateMeta[t.state]; const open = openTask === t.id; const a = t.agentId ? agentById(t.agentId) : null;
          return (
            <div key={t.id} className={cn("overflow-hidden rounded-[9px] border transition-colors", open ? "border-line-strong bg-raise" : "border-line-soft bg-raise/50 hover:border-line", t.state === "active" && "border-cyan/25")}>
              <button onClick={() => setOpenTask(open ? null : t.id)} className="flex w-full cursor-pointer items-start gap-2 px-2.5 py-2 text-left">
                <span className={cn("mt-[1px] flex size-[16px] shrink-0 items-center justify-center rounded-[4px] border", t.state === "done" ? "border-mint/40 bg-mint-tint text-mint" : t.state === "active" ? "border-cyan/40 bg-cyan-tint text-cyan" : t.state === "blocked" ? "border-rose/40 bg-rose-tint text-rose" : "border-line-strong text-ink-4")}><meta.icon size={9} /></span>
                <span className="min-w-0 flex-1">
                  <span className={cn("block text-[12px] leading-[1.45]", t.state === "done" ? "text-ink-4 line-through" : "text-ink-2")}>{t.title}</span>
                  <span className="mt-1 flex items-center gap-1.5">{a && <AgentMark glyph={a.glyph} from={a.from} to={a.to} size={12} />}<span className="font-mono text-[9.5px] text-ink-4">{a?.name ?? "unassigned"}</span><span className={cn("font-mono text-[9.5px]", toneText[meta.tone])}>{meta.label}</span></span>
                </span>
                <IconChevronDown size={11} className={cn("mt-0.5 shrink-0 text-ink-4 transition-transform", open && "rotate-180")} />
              </button>
              {open && (
                <div className="animate-slide-down space-y-2 border-t border-line-soft px-2.5 py-2">
                  {t.note && <p className="rounded-[6px] border border-amber/20 bg-amber-tint px-2 py-1.5 text-[11px] leading-[1.5] text-amber">{t.note}</p>}
                  {t.files.map((f) => <div key={f} className="flex items-center gap-1.5 font-mono text-[10.5px] text-ink-3"><IconFile size={9} className="shrink-0 text-ink-4" /><span className="truncate">{f}</span></div>)}
                  <div className="flex gap-1.5"><Button variant="ghost" size="xs">Reassign</Button><Button variant="ghost" size="xs">Edit</Button></div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ---------- changes ---------- */
function ChangesSection({ onFullView }: { onFullView: () => void }) {
  const [sel, setSel] = useState(0);
  const adds = CHANGED_FILES.reduce((s, f) => s + f.adds, 0), dels = CHANGED_FILES.reduce((s, f) => s + f.dels, 0);
  const file = REVIEW_FILES[Math.min(sel, REVIEW_FILES.length - 1)];
  return (
    <div className="space-y-2.5 p-3">
      <div className="flex items-center gap-2 rounded-[10px] border border-line-soft bg-raise px-3 py-2.5">
        <IconGit size={13} className="text-ink-3" />
        <span className="text-[12.5px] font-semibold text-ink">{CHANGED_FILES.length} files</span>
        <span className="font-mono text-[10.5px] text-mint">+{adds}</span><span className="font-mono text-[10.5px] text-rose">−{dels}</span>
        <Button variant="primary" size="xs" className="ml-auto" onClick={onFullView}>Open review</Button>
      </div>
      <div className="overflow-hidden rounded-[10px] border border-line-soft bg-raise">
        {CHANGED_FILES.map((f, i) => (
          <button key={f.path} onClick={() => setSel(i)} className={cn("flex w-full cursor-pointer items-center gap-2 border-b border-line-soft/70 px-2.5 py-[7px] text-left last:border-0", i === sel ? "bg-iris-tint/60" : "hover:bg-hover/60")}>
            <span className={cn("w-[14px] shrink-0 text-center font-mono text-[10px] font-bold", f.status === "A" ? "text-mint" : f.status === "D" ? "text-rose" : "text-amber")}>{f.status}</span>
            <span className="min-w-0 flex-1 truncate font-mono text-[10.5px] text-ink-2">{f.path}</span>
            <span className={cn("shrink-0 rounded-[3px] px-1 font-mono text-[8.5px] uppercase", f.origin === "agent" ? "bg-iris-tint text-iris-soft" : f.origin === "editor" ? "bg-sky-tint text-sky" : "bg-hover text-ink-3")}>{f.origin}</span>
            <span className="shrink-0 font-mono text-[9.5px] text-mint">+{f.adds}</span><span className="shrink-0 font-mono text-[9.5px] text-rose">−{f.dels}</span>
          </button>
        ))}
      </div>
      <DiffView file={file} maxLines={14} />
      <div className="flex gap-1.5"><Button variant="success" size="xs" icon={IconCheck} className="flex-1">Accept file</Button><Button variant="outline" size="xs">Revert</Button><Button variant="ghost" size="xs">Ask to fix</Button></div>
    </div>
  );
}

/* ---------- context ---------- */
function ContextSection() {
  const { product } = useApp();
  return (
    <div className="space-y-3 p-3">
      <div className="rounded-[10px] border border-line-soft bg-raise p-3">
        <div className="flex items-center gap-3">
          <Ring value={62} size={56} stroke={5}><span className="font-mono text-[12px] font-semibold text-ink">62%</span></Ring>
          <div className="min-w-0 flex-1 space-y-1">
            <div className="flex items-baseline justify-between"><span className="text-[12px] font-medium text-ink-2">Context window</span><span className="font-mono text-[10px] text-ink-4">124k / 200k</span></div>
            {[{ l: "System + tools", v: 14, t: "neutral" as const }, { l: product === "code" ? "Repo snapshot" : "Memory", v: 31, t: "iris" as const }, { l: "Pinned", v: 11, t: "cyan" as const }, { l: "Transcript", v: 44, t: "sky" as const }].map((r) => (
              <div key={r.l} className="flex items-center gap-2"><span className="w-[86px] shrink-0 truncate text-[10.5px] text-ink-3">{r.l}</span><Bar value={r.v} tone={r.t} height={3} /><span className="w-7 text-right font-mono text-[9.5px] text-ink-4">{r.v}%</span></div>
            ))}
          </div>
        </div>
        <div className="mt-2.5 flex gap-1.5"><Button variant="secondary" size="xs" className="flex-1">Compact</Button><Button variant="ghost" size="xs">Pin selection</Button></div>
      </div>
      <div className="rounded-[10px] border border-line-soft bg-raise">
        <div className="flex items-center justify-between border-b border-line-soft px-2.5 py-2"><span className="font-mono text-[9.5px] font-semibold tracking-[.14em] text-ink-4 uppercase">pinned</span><span className="font-mono text-[9.5px] text-ink-4">11.6k tok</span></div>
        {PINNED_FILES.map((f) => <div key={f.path} className="group flex items-center gap-2 border-b border-line-soft/70 px-2.5 py-[7px] last:border-0 hover:bg-hover/60"><IconFile size={11} className="shrink-0 text-ink-4" /><span className="min-w-0 flex-1 truncate font-mono text-[10.5px] text-ink-2">{f.path}</span><span className="font-mono text-[9.5px] text-ink-4">{(f.tokens / 1000).toFixed(1)}k</span></div>)}
      </div>
      <div className="rounded-[10px] border border-line-soft bg-raise">
        <div className="flex items-center justify-between border-b border-line-soft px-2.5 py-2"><span className="flex items-center gap-1.5 font-mono text-[9.5px] font-semibold tracking-[.14em] text-ink-4 uppercase"><IconMcp size={10} /> connectors</span><Badge tone="mint" mono className="text-[9px]">4 live</Badge></div>
        <div className="grid grid-cols-2 gap-1.5 p-2">
          {MCP_SERVERS.map((m) => <div key={m.name} className="rounded-[7px] border border-line-soft bg-well px-2 py-1.5"><div className="flex items-center gap-1.5"><i className={cn("size-[5px] rounded-full", m.status === "live" ? "bg-mint" : "animate-breathe bg-amber")} /><span className="font-mono text-[10.5px] text-ink-2">{m.name}</span><span className="ml-auto font-mono text-[9px] text-ink-4">{m.tools}t</span></div></div>)}
        </div>
      </div>
      <div className="rounded-[10px] border border-line-soft bg-raise">
        <div className="flex items-center justify-between border-b border-line-soft px-2.5 py-2"><span className="flex items-center gap-1.5 font-mono text-[9.5px] font-semibold tracking-[.14em] text-ink-4 uppercase"><IconShield size={10} /> rules</span><button className="cursor-pointer font-mono text-[9.5px] text-iris-soft hover:underline">edit</button></div>
        {RULES.map((r) => <div key={r.id} className="flex items-start gap-2 border-b border-line-soft/70 px-2.5 py-[7px] last:border-0"><div className="min-w-0 flex-1"><p className="text-[11.5px] leading-[1.45] text-ink-2">{r.text}</p><span className="font-mono text-[9.5px] text-ink-4">{r.scope}</span></div><Toggle checked={r.on} onChange={() => { /* persist via settings */ }} size="sm" /></div>)}
      </div>
    </div>
  );
}

/* ---------- runs ---------- */
const runMeta: Record<Run["status"], { tone: Tone; label: string }> = { running: { tone: "cyan", label: "running" }, queued: { tone: "neutral", label: "queued" }, review: { tone: "amber", label: "in review" }, done: { tone: "mint", label: "done" }, failed: { tone: "rose", label: "halted" } };
function RunsSection({ onFullView }: { onFullView: () => void }) {
  return (
    <div className="space-y-2.5 p-3">
      <div className="flex items-center gap-1.5"><Button variant="primary" size="xs" icon={IconBolt} className="flex-1">New parallel run</Button><Button variant="outline" size="xs" onClick={onFullView}>Fleet</Button></div>
      {RUNS.map((r) => { const m = runMeta[r.status]; const a = agentById(r.agentId); return (
        <div key={r.id} className={cn("space-y-2 rounded-[10px] border bg-raise px-2.5 py-2.5", r.status === "running" ? "border-cyan/20" : r.status === "review" ? "border-amber/25" : r.status === "failed" ? "border-rose/25" : "border-line-soft")}>
          <div className="flex items-start gap-2"><AgentMark glyph={a.glyph} from={a.from} to={a.to} size={18} /><div className="min-w-0 flex-1"><p className="truncate text-[12px] font-medium text-ink">{r.title}</p><div className="mt-0.5 font-mono text-[9.5px] text-ink-4">{r.worktree} · {r.cost}</div></div><Badge tone={m.tone} mono className="text-[9px]">{m.label}</Badge></div>
          {r.status !== "queued" && <div className="flex items-center gap-2"><Bar value={r.progress} tone={m.tone} height={3} /><span className="font-mono text-[9.5px] text-ink-4">{r.eta}</span></div>}
        </div>); })}
    </div>
  );
}

/* ---------- git ---------- */
function GitSection({ onFullView }: { onFullView: () => void }) {
  const prs = [{ num: 482, title: "IDE bridge: live sync for external editors", status: "open", checks: "11/12", reviews: "1 approval needed" }, { num: 480, title: "Payments: ledger idempotency keys", status: "changes", checks: "9 passed · 1 failed", reviews: "changes requested" }, { num: 478, title: "Dead code sweep", status: "ready", checks: "all green", reviews: "approved · 2" }];
  return (
    <div className="space-y-3 p-3">
      <div className="rounded-[10px] border border-line-soft bg-raise p-3">
        <div className="flex items-center gap-2"><IconBranch size={12} className="text-iris-soft" /><span className="font-mono text-[11.5px] text-iris-soft">feat/ide-bridge</span><span className="ml-auto font-mono text-[10px] text-ink-4">↑3 · ↓0 · 2 dirty</span></div>
        <div className="mt-2 grid grid-cols-3 gap-1.5"><Button size="xs" variant="secondary">Push 3</Button><Button size="xs" variant="secondary">Pull</Button><Button size="xs" variant="primary">Open PR</Button></div>
      </div>
      <div className="rounded-[10px] border border-line-soft bg-raise">
        <div className="flex items-center justify-between border-b border-line-soft px-2.5 py-2"><span className="font-mono text-[9.5px] font-semibold tracking-[.14em] text-ink-4 uppercase">open PRs</span><Button size="xs" variant="ghost" onClick={onFullView}>View all</Button></div>
        {prs.map((p) => <div key={p.num} className={cn("border-b border-line-soft/70 px-2.5 py-2 last:border-0", p.status === "ready" && "bg-mint-tint/20")}>
          <div className="flex items-start gap-2"><span className="font-mono text-[10px] text-ink-4">#{p.num}</span><p className="min-w-0 flex-1 text-[11.5px] text-ink">{p.title}</p>{p.status === "ready" && <Badge tone="mint" mono className="text-[9px]">ready</Badge>}{p.status === "changes" && <Badge tone="rose" mono className="text-[9px]">changes</Badge>}</div>
          <div className="mt-1 flex items-center gap-2 font-mono text-[9.5px] text-ink-4"><span className={p.status === "changes" ? "text-rose" : "text-mint"}>{p.checks}</span><span>·</span><span>{p.reviews}</span></div>
        </div>)}
      </div>
    </div>
  );
}

/* ---------- browser ---------- */
function BrowserSection({ onFullView, onAskAgent }: { onFullView: () => void; onAskAgent: (text: string) => void }) {
  return (
    <div className="space-y-3 p-3">
      <div className="flex items-center gap-2 rounded-[10px] border border-line bg-sunken px-2.5 py-2"><IconEye size={12} className="shrink-0 text-mint" /><span className="min-w-0 flex-1 truncate font-mono text-[11px] text-ink-2">localhost:5173/review</span><Badge tone="mint" mono className="text-[9px]">200</Badge></div>
      <div className="dot-bg flex min-h-[180px] items-center justify-center rounded-[12px] border border-line-strong bg-base p-3">
        <div className="w-full rounded-[8px] border border-line-soft bg-raise p-3">
          <div className="skeleton h-4 w-2/3 rounded-[4px]" />
          <div className="mt-2 space-y-1.5"><div className="skeleton h-2.5 w-full rounded-[4px]" /><div className="skeleton h-2.5 w-5/6 rounded-[4px]" /></div>
          <div className="mt-3 rounded-[6px] border border-line-soft bg-sunken p-2 font-mono text-[10px] leading-[1.6]"><div className="diff-del px-1.5 text-rose">- socket.send(store.snapshot())</div><div className="diff-add px-1.5 text-mint">+ attachEditor(socket, channel)</div></div>
        </div>
      </div>
      <div className="rounded-[10px] border border-amber/20 bg-amber-tint/40 p-2.5"><div className="flex items-center gap-2"><IconWarning size={11} className="text-amber" /><span className="text-[11px] font-medium text-amber">1 error · 2 warnings</span></div><button onClick={() => onAskAgent("Fix: ResizeObserver loop at DiffView.tsx:118")} className="mt-1.5 cursor-pointer text-[10.5px] text-iris-soft hover:underline">Fix all with agent →</button></div>
      <div className="flex gap-1.5"><Button variant="secondary" size="xs" icon={IconSearch} className="flex-1">Pick element</Button><Button variant="outline" size="xs" onClick={onFullView}>Full browser</Button></div>
    </div>
  );
}

/* ---------- editors ---------- */
function EditorsSection({ onFullView }: { onFullView: () => void }) {
  const items = [{ id: "cursor", name: "Cursor", status: "live", latency: "9ms", caps: "Diff · Agent tabs" }, { id: "vscode", name: "VS Code", status: "live", latency: "11ms", caps: "Diff · Terminal · Worktree" }, { id: "zed", name: "Zed", status: "ready", latency: "—", caps: "Multi-buffer" }, { id: "jetbrains", name: "IntelliJ", status: "offline", latency: "—", caps: "—" }];
  return (
    <div className="space-y-2.5 p-3">
      <div className="rounded-[10px] border border-iris/25 bg-iris-tint/60 p-3"><div className="flex items-center gap-1.5"><IconPlug size={12} className="text-iris-soft" /><span className="text-[12px] font-semibold text-ink">Editor bridges</span></div><p className="mt-1 text-[11px] leading-[1.5] text-ink-3">Attach an editor and keep it in lockstep. Edits made in Cursor flow through the same review queue.</p></div>
      {items.map((e) => <div key={e.id} className={cn("flex items-center gap-2 rounded-[10px] border bg-raise px-2.5 py-2", e.status === "live" ? "border-mint/25" : "border-line-soft")}>
        <span className="flex size-[22px] items-center justify-center rounded-[6px] border border-line-soft bg-well font-mono text-[9.5px] font-bold text-ink-2">{e.id.slice(0, 2).toUpperCase()}</span>
        <div className="min-w-0 flex-1"><div className="flex items-center gap-1.5"><span className="text-[12px] text-ink">{e.name}</span><Badge tone={e.status === "live" ? "mint" : e.status === "ready" ? "sky" : "neutral"} mono className="text-[9px]" dot>{e.status}</Badge></div><div className="font-mono text-[9.5px] text-ink-4">{e.caps}</div></div>
        {e.status === "live" && <span className="shrink-0 font-mono text-[10px] text-mint">{e.latency}</span>}
      </div>)}
      <Button variant="ghost" size="xs" onClick={onFullView} className="w-full">Manage bridges →</Button>
    </div>
  );
}

/* ---------- terminal snippet ---------- */
function TerminalSection({ onOpenTerminal }: { onOpenTerminal: () => void }) {
  return (
    <div className="space-y-3 p-3">
      <div className="rounded-[10px] border border-line-soft bg-raise p-3"><div className="flex items-center gap-1.5"><IconTerminal size={12} className="text-cyan" /><span className="text-[12px] font-semibold text-ink">Latest agent commands</span><Button variant="ghost" size="xs" className="ml-auto" onClick={onOpenTerminal}>Open terminal</Button></div><p className="mt-1 text-[11px] text-ink-3">Full output and interactive shells live in the terminal panel (⌘J).</p></div>
      <div className="overflow-hidden rounded-[10px] border border-line-soft bg-code font-mono text-[11px] leading-[1.65]">
        <div className="border-b border-line-soft bg-well px-2.5 py-1.5 text-[9.5px] tracking-[.14em] text-ink-4 uppercase">agent · tail</div>
        <pre className="scroll-thin max-h-[280px] overflow-auto px-2.5 py-2 text-ink-3">
          <div><span className="text-iris-soft">❯</span> pnpm --filter @aro/bridge test</div>
          <div className="text-mint"> ✓ session-channel.test.ts (14 tests) 284ms</div>
          <div className="text-mint"> ✓ attach-editor.test.ts (9 tests) 121ms</div>
          <div className="text-mint"> ✓ reconnect.test.ts (12 tests) 612ms</div>
          <div className="text-mint">Test Files 3 passed · Tests 35 passed</div>
          <div className="mt-2 text-ink-4">[hmr] updated packages/review/src/FileList.tsx</div>
        </pre>
      </div>
    </div>
  );
}

/* ============================================================
   PANE SHELL
   ============================================================ */
export function ContextPane({ section, setSection, onNavigate, onAskAgent, onCollapse, onOpenTerminal }: {
  section: ContextSectionId; setSection: (s: ContextSectionId) => void;
  onNavigate: (v: ViewId) => void; onAskAgent: (text: string) => void; onCollapse: () => void; onOpenTerminal: () => void;
}) {
  const counts = { changed: CHANGED_FILES.length, plan: PLAN_TASKS.filter((t) => t.state !== "done").length, runs: RUNS.filter((r) => r.status === "running" || r.status === "review").length };
  const current = SECTIONS.find((s) => s.id === section) ?? SECTIONS[0];
  return (
    <aside className="flex h-full min-w-0 flex-col border-l border-line-soft bg-sunken">
      {/* header */}
      <div className="flex h-[44px] shrink-0 items-center gap-2 border-b border-line-soft px-3">
        <current.icon size={13} className="text-iris-soft" />
        <h2 className="font-display text-[12.5px] font-semibold text-ink">{current.label}</h2>
        {typeof current.count === "function" && current.count(counts) !== undefined && <span className="rounded-full bg-hover px-1.5 font-mono text-[9.5px] text-ink-3">{current.count(counts)}</span>}
        <div className="ml-auto flex items-center">
          {current.keywordView && <Tip label="Open as full page"><IconButton icon={IconHistory} label="Open as page" size={26} onClick={() => current.keywordView && onNavigate(current.keywordView)} /></Tip>}
          <Tip label="Hide panel · ⌘L"><IconButton icon={IconX} label="Collapse" size={26} onClick={onCollapse} /></Tip>
        </div>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* icon rail */}
        <nav className="flex w-[44px] shrink-0 flex-col items-center gap-0.5 border-r border-line-soft bg-void py-1.5">
          {SECTIONS.map((item) => {
            const active = section === item.id;
            const count = typeof item.count === "function" ? item.count(counts) : undefined;
            return (
              <Tip key={item.id} label={item.label}>
                <button onClick={() => setSection(item.id)} className={cn("group relative flex size-[34px] cursor-pointer items-center justify-center rounded-[9px] transition-all duration-150 active:scale-[.94]", active ? "bg-raise text-iris-soft shadow-e1" : "text-ink-3 hover:bg-raise/60 hover:text-ink-2")}>
                  {active && <span className="absolute top-1.5 bottom-1.5 -left-[3px] w-[2px] rounded-full bg-iris" />}
                  <item.icon size={14} />
                  {count !== undefined && count > 0 && <span className="absolute -top-0.5 -right-0.5 flex h-[13px] min-w-[13px] items-center justify-center rounded-full bg-iris px-0.5 font-mono text-[8px] font-bold text-on-iris">{count}</span>}
                </button>
              </Tip>
            );
          })}
          <div className="mt-auto flex flex-col items-center gap-0.5 pt-1.5">
            <Tip label="New section layout"><IconButton icon={IconPlus} label="Add section" size={30} onClick={() => { /* reserved — future custom pins */ }} /></Tip>
            <Tip label="Pop out panel"><IconButton icon={IconPanelRight} label="Pop out" size={30} /></Tip>
          </div>
        </nav>

        <div key={section} className="scroll-thin flex-1 animate-fade overflow-y-auto">
          {section === "plan" && <PlanSection />}
          {section === "changes" && <ChangesSection onFullView={() => onNavigate("review")} />}
          {section === "context" && <ContextSection />}
          {section === "runs" && <RunsSection onFullView={() => onNavigate("runs")} />}
          {section === "git" && <GitSection onFullView={() => onNavigate("git")} />}
          {section === "browser" && <BrowserSection onFullView={() => onNavigate("browser")} onAskAgent={onAskAgent} />}
          {section === "editors" && <EditorsSection onFullView={() => onNavigate("bridges")} />}
          {section === "terminal" && <TerminalSection onOpenTerminal={onOpenTerminal} />}
        </div>
      </div>

      {/* footer with quick actions */}
      <div className="shrink-0 border-t border-line-soft bg-base/50 px-2 py-2">
        <div className="flex items-center justify-between gap-1 font-mono text-[9.5px] text-ink-4">
          <span className="flex items-center gap-1"><IconSpark size={10} />agent sync · live</span>
          <span className="flex items-center gap-1"><IconFile size={10} />{counts.changed} files · {counts.runs} runs</span>
        </div>
      </div>
    </aside>
  );
}
