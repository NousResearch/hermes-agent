"use client";

import { useEffect, useState } from "react";
import { cn } from "../utils/cn";
import { AGENTS, BRIDGES, RUNS, agentById, type Agent, type AgentKind, type DiffFile, type DiffLine } from "../data/catalog";
import { useApp } from "../lib/app";
import { AgentMark, IconBolt, IconCheck, IconChevronDown, IconFile, IconGit, IconPencil, IconPlug, IconRefresh, IconSpark, IconTerminal, IconUndo, IconX, IdeMark } from "./Icons";
import { Badge, Bar, Button, IconButton, Input, Ring, Segmented, Stat, type Tone } from "./ui";
import { DiffView } from "./Transcript";
import { Grid, PageHeader } from "./Titlebar";

/* ================================ AGENTS ================================ */
const statusTone: Record<Agent["status"], Tone> = { connected: "mint", "auth-required": "amber", "not-installed": "neutral", degraded: "rose" };
export function AgentsView({ onStart }: { onStart: (agentId: string) => void }) {
  const { product } = useApp();
  const [kind, setKind] = useState<"all" | AgentKind>("all");
  const list = AGENTS.filter((a) => kind === "all" || a.kind === kind);
  return (
    <div className="scroll-thin flex-1 overflow-y-auto">
      <PageHeader eyebrow={product === "code" ? "registry" : "models"} title={product === "code" ? "Coding agents" : "Models behind the assistant"} sub={product === "code" ? "One contract for every agent: same transcript, same approvals, same review queue. Bring your own CLI." : "Agent mode routes each step to the cheapest capable model. These are the engines it can pick from."}
        right={<div className="flex items-center gap-2"><Segmented value={kind} onChange={setKind} items={[{ value: "all", label: "All" }, { value: "terminal", label: "Terminal" }, { value: "ide", label: "Editor" }, { value: "cloud", label: "Cloud" }, { value: "hybrid", label: "Hybrid" }]} /><Button variant="secondary" icon={IconPlug}>Add custom</Button></div>} />
      <div className="space-y-4 p-4">
        <div className="relative overflow-hidden rounded-[12px] border border-line bg-raise p-4">
          <div className="grid-bg pointer-events-none absolute inset-0" />
          <div className="relative flex flex-wrap items-start gap-6">
            <div className="min-w-[280px] flex-1">
              <div className="flex items-center gap-2"><IconBolt size={13} className="text-iris-soft" /><h3 className="text-[13.5px] font-semibold text-ink">Routing policy</h3><Badge tone="iris" mono className="text-[9.5px]">active</Badge></div>
              <p className="mt-1 max-w-[560px] text-[12.5px] leading-[1.6] text-ink-3">Work is dispatched by intent. Aro scores each agent on the shape of the task and hands off mid-session if a better fit appears.</p>
              <div className="mt-3 grid gap-2 sm:grid-cols-2">
                {[{ m: "Multi-file backend refactor", to: "codex" }, { m: "Ambiguous front-end work", to: "claude-code" }, { m: "Bulk mechanical migration", to: "glm-code" }, { m: "Pixel-level inline edit", to: "cursor" }, { m: "Overnight unattended sweep", to: "darwin" }, { m: "Private / air-gapped code", to: "aro" }].map((r) => (
                  <div key={r.m} className="flex items-center gap-2 rounded-[8px] border border-line-soft bg-well px-2.5 py-2"><span className="min-w-0 flex-1 truncate text-[12px] text-ink-2">{r.m}</span><IconChevronDown size={11} className="-rotate-90 text-ink-4" /><span className="font-mono text-[11px] text-iris-soft">{r.to}</span></div>
                ))}
              </div>
            </div>
            <div className="grid w-full grid-cols-3 gap-2 lg:w-[320px] lg:grid-cols-1">
              {[{ l: "tasks routed", v: "1,448", s: "30 days" }, { l: "handoff rate", v: "12%", s: "mid-session switch" }, { l: "spend avoided", v: "$612", s: "vs single agent" }].map((s) => <div key={s.l} className="rounded-[9px] border border-line-soft bg-well px-3 py-2.5"><div className="font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">{s.l}</div><div className="mt-1 text-[19px] leading-none font-semibold text-ink">{s.v}</div><div className="mt-1 font-mono text-[9.5px] text-ink-4">{s.s}</div></div>)}
            </div>
          </div>
        </div>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          {list.map((a) => (
            <div key={a.id} className={cn("flex flex-col overflow-hidden rounded-[12px] border bg-raise transition-all hover:shadow-e3", a.status === "connected" ? "border-line-soft hover:border-line-strong" : "border-dashed border-line")}>
              <div className="h-[2px]" style={{ background: `linear-gradient(90deg, ${a.from}, ${a.to})`, opacity: a.status === "connected" ? 0.9 : 0.25 }} />
              <div className="flex items-start gap-2.5 px-3 pt-3">
                <AgentMark glyph={a.glyph} from={a.from} to={a.to} size={32} />
                <div className="min-w-0 flex-1"><div className="flex items-center gap-1.5"><h4 className="truncate text-[13.5px] font-semibold text-ink">{a.name}</h4><Badge tone={statusTone[a.status]} mono className="text-[9px]">{a.status}</Badge></div><div className="mt-0.5 font-mono text-[10px] text-ink-4">{a.vendor} · {a.version}</div></div>
                <Ring value={a.successRate} size={34} stroke={3} tone={a.to}><span className="font-mono text-[8.5px] text-ink-2">{a.successRate}</span></Ring>
              </div>
              <div className="mt-3 grid grid-cols-3 gap-px border-y border-line-soft bg-line-soft/60">
                {[{ l: "runs", v: a.invocations }, { l: "p50", v: a.p50 }, { l: "ctx", v: a.models[0].ctx }].map((s) => <div key={s.l} className="bg-raise px-2.5 py-2"><div className="font-mono text-[8.5px] tracking-[.1em] text-ink-4 uppercase">{s.l}</div><div className="mt-0.5 font-mono text-[12px] text-ink">{s.v}</div></div>)}
              </div>
              <div className="flex-1 space-y-2 px-3 py-2.5">
                <div className="flex flex-wrap gap-1">{a.strengths.map((s) => <span key={s} className="rounded-[4px] bg-mint-tint px-1.5 py-[1px] font-mono text-[9.5px] text-mint">{s}</span>)}</div>
                <div className="flex flex-wrap gap-1">{a.weakSpots.map((s) => <span key={s} className="rounded-[4px] bg-amber-tint px-1.5 py-[1px] font-mono text-[9.5px] text-amber">{s}</span>)}</div>
              </div>
              <div className="flex items-center gap-1.5 border-t border-line-soft px-2.5 py-2"><Button variant="primary" size="xs" className="flex-1" disabled={a.status !== "connected"} onClick={() => onStart(a.id)}>New session</Button><Button variant="ghost" size="xs">Configure</Button><IconButton icon={IconRefresh} label="Reconnect" size={24} /></div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ================================ FLEET ================================ */
const COLUMNS: { id: string; label: string; tone: Tone }[] = [{ id: "queued", label: "Queued", tone: "neutral" }, { id: "running", label: "Running", tone: "cyan" }, { id: "review", label: "Awaiting review", tone: "amber" }, { id: "done", label: "Landed", tone: "mint" }];
export function FleetView() {
  const { product } = useApp();
  return (
    <div className="scroll-thin flex-1 overflow-y-auto">
      <PageHeader eyebrow={product === "code" ? "fleet" : "automations"} title={product === "code" ? "Parallel runs" : "Automations & background work"} sub={product === "code" ? "Every run is an isolated worktree with its own budget, checkpoints and review queue." : "Recurring jobs and long tasks the assistant runs without you watching. Each has a spend cap."}
        right={<div className="flex gap-2"><Button variant="secondary" icon={IconGit}>Reconcile all</Button><Button variant="primary" icon={IconBolt}>Launch</Button></div>} />
      <div className="space-y-4 p-4">
        <Grid className="grid-cols-2 lg:grid-cols-4">
          <Stat label="active" value="2" sub="1 cloud · 1 local" chart={[3, 4, 2, 5, 4, 6, 3, 5]} /><Stat label="spend" value="$4.29" sub="cap $25 · 17%" tone="iris" chart={[1, 2, 2, 3, 3, 4, 4, 4.3]} /><Stat label="queued" value="1" sub="waiting on lock" /><Stat label="needs review" value="1" sub="best-of-4 · variant B" tone="amber" />
        </Grid>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
          {COLUMNS.map((col) => {
            const items = RUNS.filter((r) => col.id === "done" ? r.status === "done" || r.status === "failed" : r.status === col.id);
            return (
              <div key={col.id} className="flex flex-col rounded-[11px] border border-line-soft bg-sunken">
                <div className="flex items-center gap-2 border-b border-line-soft px-3 py-2.5"><span className="text-[12.5px] font-semibold text-ink">{col.label}</span><span className="rounded-full bg-raise px-1.5 font-mono text-[9.5px] text-ink-4">{items.length}</span></div>
                <div className="space-y-2 p-2">
                  {items.length === 0 && <div className="rounded-[9px] border border-dashed border-line px-3 py-6 text-center text-[11px] text-ink-4">empty</div>}
                  {items.map((r) => { const a = agentById(r.agentId); const failed = r.status === "failed"; return (
                    <div key={r.id} className={cn("space-y-2 rounded-[10px] border bg-raise p-2.5 transition-all hover:shadow-e2", failed ? "border-rose/25" : "border-line-soft hover:border-line-strong")}>
                      <div className="flex items-start gap-2"><AgentMark glyph={a.glyph} from={a.from} to={a.to} size={18} /><p className="min-w-0 flex-1 text-[12px] leading-[1.45] font-medium text-ink">{r.title}</p></div>
                      <div className="font-mono text-[9.5px] text-ink-4">{r.worktree}</div>
                      {r.status !== "queued" && <div className="flex items-center gap-2"><Bar value={r.progress} tone={failed ? "rose" : col.tone} height={3} /><span className="font-mono text-[9px] text-ink-4">{r.eta}</span></div>}
                      <div className="flex justify-between font-mono text-[9.5px] text-ink-4"><span>{r.tokens.toLocaleString()} tok</span><span>{r.cost}</span></div>
                      {r.variant && <p className="rounded-[6px] bg-mint-tint px-2 py-1 font-mono text-[9.5px] text-mint">▲ {r.variant}</p>}
                    </div>); })}
                </div>
              </div>);
          })}
        </div>
      </div>
    </div>
  );
}

/* ================================ BRIDGES ================================ */
const MATRIX = [
  { cap: "Inline diff review", vscode: 2, cursor: 2, zed: 2, windsurf: 2, jetbrains: 1, neovim: 1 }, { cap: "Selection → context", vscode: 2, cursor: 2, zed: 2, windsurf: 2, jetbrains: 1, neovim: 2 },
  { cap: "Terminal passthrough", vscode: 2, cursor: 2, zed: 1, windsurf: 2, jetbrains: 1, neovim: 2 }, { cap: "Worktree handoff", vscode: 2, cursor: 2, zed: 0, windsurf: 0, jetbrains: 0, neovim: 1 },
  { cap: "Rules sync", vscode: 2, cursor: 2, zed: 0, windsurf: 1, jetbrains: 0, neovim: 1 }, { cap: "Design mode canvas", vscode: 0, cursor: 2, zed: 0, windsurf: 0, jetbrains: 0, neovim: 0 },
];
export function BridgesView() {
  const { toast } = useApp();
  return (
    <div className="scroll-thin flex-1 overflow-y-auto">
      <PageHeader eyebrow="interoperability" title="External editors" sub="Keep the agent where it's strong and your editor where you're strong. Attach Cursor, VS Code, Windsurf, Zed, JetBrains or Neovim to any running session." right={<div className="flex items-center gap-2"><Badge tone="mint" mono dot>2 attached</Badge><Button variant="secondary" icon={IconTerminal}>Daemon logs</Button></div>} />
      <div className="space-y-4 p-4">
        <div className="grid gap-4 rounded-[14px] border border-line bg-raise p-5 lg:grid-cols-[1.1fr_1fr]">
          <div>
            <h3 className="text-[15px] font-semibold tracking-[-.01em] text-ink">One session, every surface</h3>
            <p className="mt-1.5 text-[12.5px] leading-[1.65] text-ink-3">The daemon owns state. Editors subscribe to resumable frames and can push edits back — every editor-originated patch is routed through the same reviewer.</p>
            <div className="mt-4 space-y-2">
              {[["1", "Editor requests attach", "capability handshake on :4733"], ["2", "Frames stream out", "resumable cursor, back-pressure aware"], ["3", "Edits stream back", "patch → reviewer → checkpoint"], ["4", "Disconnect is safe", "resumes from last acked frame"]].map(([n, t, d]) => (
                <div key={n} className="flex items-start gap-3"><span className="flex size-[18px] shrink-0 items-center justify-center rounded-[5px] border border-iris/30 bg-iris-tint font-mono text-[9.5px] text-iris-soft">{n}</span><div><p className="text-[12.5px] font-medium text-ink">{t}</p><p className="font-mono text-[10px] text-ink-4">{d}</p></div></div>
              ))}
            </div>
          </div>
          <div className="overflow-hidden rounded-[10px] border border-line-strong bg-code">
            <div className="flex items-center gap-1.5 border-b border-line-soft px-2.5 py-1.5">{["#ff5f57", "#febc2e", "#28c840"].map((c) => <i key={c} className="size-[7px] rounded-full" style={{ background: c }} />)}<span className="ml-1.5 font-mono text-[10px] text-ink-4">aro — bridge</span></div>
            <pre className="scroll-thin overflow-x-auto px-3 py-2.5 font-mono text-[11px] leading-[1.75] text-ink-2"><span className="text-ink-4">$ </span>aro bridge attach --session s1{"\n"}<span className="text-mint">✓</span> handshake vscode@1.104.2 · caps: diff,terminal,worktree{"\n"}<span className="text-mint">✓</span> frames resumed from cursor 4821{"\n"}<span className="text-cyan">▶</span> ws://127.0.0.1:4733/session/s1{"\n"}<span className="text-ink-4">$ </span><span className="animate-caret text-iris-soft">▊</span></pre>
          </div>
        </div>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          {BRIDGES.map((b) => (
            <div key={b.id} className={cn("flex flex-col rounded-[12px] border bg-raise p-3 transition-all hover:shadow-e2", b.status === "live" ? "border-mint/20" : "border-line-soft hover:border-line-strong")}>
              <div className="flex items-start gap-2.5"><IdeMark kind={b.ide} size={30} /><div className="min-w-0 flex-1"><div className="flex items-center gap-1.5"><h4 className="text-[13.5px] font-semibold text-ink capitalize">{b.ide}</h4><Badge tone={b.status === "live" ? "mint" : b.status === "ready" ? "sky" : b.status === "update" ? "amber" : "neutral"} mono className="text-[9px]">{b.status}</Badge></div><p className="font-mono text-[9.5px] text-ink-4">ext {b.version}</p></div>{b.status === "live" && <span className="font-mono text-[10px] text-mint">{b.latency}</span>}</div>
              <p className="mt-2.5 min-h-[34px] text-[12px] leading-[1.55] text-ink-3">{b.note}</p>
              <div className="mt-2 flex flex-wrap gap-1">{b.capabilities.map((c) => <span key={c} className="rounded-[4px] border border-line-soft bg-well px-1.5 py-[1px] font-mono text-[9.5px] text-ink-3">{c}</span>)}</div>
              <div className="mt-3 flex gap-1.5 border-t border-line-soft pt-2.5"><Button variant={b.status === "live" ? "secondary" : "primary"} size="xs" className="flex-1" onClick={() => toast(b.status === "live" ? `Detached ${b.ide}` : `Attached ${b.ide}`, "mint")}>{b.status === "live" ? "Detach" : b.status === "offline" ? "Install extension" : "Attach session"}</Button><Button variant="ghost" size="xs">Docs</Button></div>
            </div>
          ))}
        </div>
        <div className="overflow-hidden rounded-[12px] border border-line-soft bg-raise">
          <div className="border-b border-line-soft px-3 py-2.5 text-[12.5px] font-semibold text-ink">Capability matrix</div>
          <div className="scroll-thin overflow-x-auto"><table className="w-full min-w-[640px] border-collapse">
            <thead><tr className="border-b border-line-soft"><th className="px-3 py-2 text-left font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">capability</th>{["vscode", "cursor", "zed", "windsurf", "jetbrains", "neovim"].map((h) => <th key={h} className="px-3 py-2"><span className="flex flex-col items-center gap-1"><IdeMark kind={h} size={16} /><span className="font-mono text-[9.5px] text-ink-3 capitalize">{h}</span></span></th>)}</tr></thead>
            <tbody>{MATRIX.map((row) => <tr key={row.cap} className="border-b border-line-soft/60 last:border-0 hover:bg-hover/40"><td className="px-3 py-2 text-[12px] text-ink-2">{row.cap}</td>{(["vscode", "cursor", "zed", "windsurf", "jetbrains", "neovim"] as const).map((k) => <td key={k} className="px-3 py-2 text-center"><i className={cn("inline-block size-[8px] rounded-[2px]", row[k] === 2 ? "bg-mint" : row[k] === 1 ? "bg-amber" : "bg-line-strong")} /></td>)}</tr>)}</tbody>
          </table></div>
        </div>
      </div>
    </div>
  );
}

/* ================================ REVIEW ================================ */
type Hunk = { header: string; lines: DiffLine[]; adds: number; dels: number };
function splitHunks(file: DiffFile): Hunk[] {
  const hunks: Hunk[] = [];
  for (const ln of file.lines) {
    let cur = hunks[hunks.length - 1];
    if (ln.t === "hunk" || !cur) {
      cur = { header: ln.t === "hunk" ? ln.text : "@@", lines: [], adds: 0, dels: 0 };
      hunks.push(cur);
    }
    cur.lines.push(ln);
    if (ln.t === "add") cur.adds++;
    else if (ln.t === "del") cur.dels++;
  }
  return hunks;
}
export const REVIEW_FILES: DiffFile[] = [
  { path: "packages/bridge/src/server.ts", adds: 20, dels: 6, lang: "ts", lines: [
    { t: "hunk", text: "@@ -4,9 +4,19 @@ export async function startBridge", o: 4, n: 4 }, { t: "ctx", text: "  const store = new SessionStore(opts.root)", o: 5, n: 5 }, { t: "del", text: "  const server = createServer({ port: opts.port })", o: 6 },
    { t: "add", text: "  const channel = new SessionChannel(store, {", n: 6 }, { t: "add", text: "    flushEveryMs: 12,", n: 7 }, { t: "add", text: "    resume: 'cursor',", n: 8 }, { t: "add", text: "  })", n: 9 },
    { t: "ctx", text: "  server.on('connection', (socket) => {", o: 8, n: 12 }, { t: "del", text: "    socket.send(store.snapshot())", o: 9 }, { t: "add", text: "    const client = attachEditor(socket, channel, {", n: 13 }, { t: "add", text: "      capabilities: socket.requested ?? [],", n: 14 }, { t: "add", text: "      onEdit: (patch) => reviewer.enqueue(patch),", n: 15 }, { t: "add", text: "    })", n: 16 }, { t: "ctx", text: "  })", o: 10, n: 18 },
    { t: "hunk", text: "@@ -38,7 +48,10 @@ export async function stopBridge", o: 38, n: 48 }, { t: "ctx", text: "  server.on('close', () => {", o: 39, n: 49 }, { t: "del", text: "    store.dispose()", o: 40 },
    { t: "add", text: "    void channel.drain().then(() => {", n: 50 }, { t: "add", text: "      store.dispose()", n: 51 }, { t: "add", text: "      for (const s of sockets) s.close(1000, 'server-stop')", n: 52 }, { t: "add", text: "    })", n: 53 }, { t: "ctx", text: "  })", o: 42, n: 55 },
    { t: "hunk", text: "@@ -60,4 +73,9 @@ export function describeBridge", o: 60, n: 73 }, { t: "ctx", text: "export function describeBridge(b: Bridge) {", o: 61, n: 74 }, { t: "del", text: "  return `bridge:${b.port}`", o: 62 },
    { t: "add", text: "  const caps = b.editors.map((e) => e.caps.join('+')).sort()", n: 75 }, { t: "add", text: "  return `bridge:${b.port} editors=${caps.length}`", n: 76 }, { t: "add", text: "    + ` caps=${caps.join(',')}`", n: 77 }, { t: "add", text: "    + ` queue=${b.reviewer.size()}`", n: 78 }, { t: "ctx", text: "}", o: 63, n: 80 } ] },
  { path: "packages/editor-kit/src/attach.ts", adds: 47, dels: 9, lang: "ts", lines: [
    { t: "hunk", text: "@@ -12,6 +12,47 @@ export function attachEditor", o: 12, n: 12 }, { t: "add", text: "export type EditorCaps = 'diff' | 'terminal' | 'worktree' | 'rules'", n: 12 }, { t: "add", text: "const BACKOFF = [0, 250, 500, 1_000, 2_500, 5_000]", n: 14 },
    { t: "ctx", text: "export function attachEditor(socket, channel, opts) {", o: 13, n: 16 }, { t: "del", text: "  socket.on('message', (m) => channel.push(m))", o: 14 }, { t: "add", text: "  const caps = negotiate(socket, opts.capabilities)", n: 17 }, { t: "add", text: "  let cursor = channel.head", n: 18 }, { t: "add", text: "  socket.on('ack', (n) => { cursor = n })", n: 19 }, { t: "add", text: "  socket.on('edit', (patch) => reviewer.enqueue({ ...patch, origin: 'editor' }))", n: 24 }, { t: "ctx", text: "  return { dispose }", o: 18, n: 30 },
    { t: "hunk", text: "@@ -41,6 +76,13 @@ function dispose", o: 41, n: 76 }, { t: "ctx", text: "  function dispose() {", o: 42, n: 77 }, { t: "del", text: "    socket.close()", o: 43 },
    { t: "add", text: "    clearTimeout(retryTimer)", n: 78 }, { t: "add", text: "    for (const t of backoffTimers) clearTimeout(t)", n: 79 }, { t: "add", text: "    socket.close(1000, 'client-dispose')", n: 80 }, { t: "add", text: "    channel.release(cursor)", n: 81 }, { t: "ctx", text: "  }", o: 45, n: 83 } ] },
  { path: "packages/review/src/reviewer.ts", adds: 9, dels: 3, lang: "ts", lines: [
    { t: "hunk", text: "@@ -30,4 +30,12 @@ class DiffReviewer", o: 30, n: 30 }, { t: "del", text: "  enqueue(patch: AgentPatch) {", o: 31 }, { t: "add", text: "  enqueue(patch: Patch) {   // agent- or editor-originated", n: 31 }, { t: "add", text: "    const entry = normalise(patch, patch.origin ?? 'agent')", n: 32 }, { t: "add", text: "    if (entry.origin === 'editor') this.markHuman(entry)", n: 33 }, { t: "ctx", text: "    this.queue.push(entry)", o: 33, n: 34 }, { t: "add", text: "    checkpoints.snapshot([entry.path])", n: 36 } ] },
];
const REVIEW_HUNKS: Record<string, Hunk[]> = Object.fromEntries(REVIEW_FILES.map((f) => [f.path, splitHunks(f)]));
const REVIEW_TOTAL_HUNKS = REVIEW_FILES.reduce((n, f) => n + REVIEW_HUNKS[f.path].length, 0);
type HunkVerdict = "accepted" | "rejected";
type HunkThreadData = { id: string; hunkKey: string; text: string };

function HunkThread({ thread }: { thread: HunkThreadData }) {
  const [queued, setQueued] = useState(false);
  const [resolved, setResolved] = useState(false);
  useEffect(() => {
    if (queued) return;
    const t = setTimeout(() => setQueued(true), 1200);
    return () => clearTimeout(t);
  }, [queued]);
  return (
    <div className="animate-slide-down">
      <div className={cn("border-t border-line-soft bg-amber-tint/50 px-2.5 py-2 transition-opacity", resolved && "opacity-60")}>
        <div className="flex items-start gap-2">
          <IconPencil size={11} className="mt-[3px] shrink-0 text-amber" />
          <div className="min-w-0 flex-1">
            <p className={cn("text-[12px] leading-[1.55] text-ink-2", resolved && "text-ink-4 line-through")}>{thread.text}</p>
            <div className="mt-1 font-mono text-[9.5px] text-ink-4">you · just now{resolved && <span className="text-mint"> · resolved</span>}</div>
          </div>
          <Button variant="ghost" size="xs" onClick={() => setResolved((r) => !r)} aria-label={resolved ? "Reopen thread" : "Resolve thread"}>{resolved ? "Reopen" : "Resolve"}</Button>
        </div>
        <div className="mt-1.5 flex items-center gap-1.5 border-t border-amber/20 pt-1.5">
          {queued ? (<>
            <IconCheck size={10} className="shrink-0 text-mint" /><span className="font-mono text-[10px] text-mint">Aro queued fix · patch incoming</span>
          </>) : (<>
            <i className="size-[5px] shrink-0 animate-breathe rounded-full bg-amber" /><span className="font-mono text-[10px] text-ink-4">Aro will address this before the next checkpoint</span>
          </>)}
        </div>
      </div>
    </div>
  );
}

export function ReviewView() {
  const { toast } = useApp();
  const [active, setActive] = useState(0);
  const [staged, setStaged] = useState<string[]>([REVIEW_FILES[0].path]);
  const [layout, setLayout] = useState<"unified" | "split">("split");
  const [verdicts, setVerdicts] = useState<Record<string, HunkVerdict>>({});
  const [commenting, setCommenting] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [threads, setThreads] = useState<HunkThreadData[]>([]);
  const file = REVIEW_FILES[active];
  const hunks = REVIEW_HUNKS[file.path];
  const hkey = (i: number) => `${file.path}#${i}`;
  const fAcc = hunks.reduce((n, _, i) => n + (verdicts[hkey(i)] === "accepted" ? 1 : 0), 0);
  const fRej = hunks.reduce((n, _, i) => n + (verdicts[hkey(i)] === "rejected" ? 1 : 0), 0);
  const gAcc = Object.values(verdicts).filter((v) => v === "accepted").length;
  const gRej = Object.values(verdicts).filter((v) => v === "rejected").length;
  const fileThreads = threads.filter((t) => t.hunkKey.startsWith(`${file.path}#`));
  const setVerdict = (key: string, v: HunkVerdict) => setVerdicts((s) => {
    if (s[key] === v) { const next = { ...s }; delete next[key]; return next; }
    return { ...s, [key]: v };
  });
  const resetVerdict = (key: string) => setVerdicts((s) => {
    if (!(key in s)) return s;
    const next = { ...s }; delete next[key]; return next;
  });
  const toggleComposer = (key: string) => { setDraft(""); setCommenting((c) => (c === key ? null : key)); };
  const sendComment = (key: string) => {
    const text = draft.trim();
    if (!text) return;
    setThreads((ts) => [...ts, { id: `t${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, hunkKey: key, text }]);
    setDraft("");
    setCommenting(null);
  };
  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <PageHeader eyebrow="changes" title="Review changes" sub="Agent edits, editor edits and cloud runs all land here as one reviewable diff — with a checkpoint behind every change."
        right={<div className="flex flex-wrap items-center gap-2"><Segmented value={layout} onChange={setLayout} items={[{ value: "unified", label: "Unified" }, { value: "split", label: "Side by side" }]} /><Button variant="secondary" icon={IconUndo}>Restore checkpoint</Button><Button variant="danger" onClick={() => toast(`Changes requested — Aro is rewriting ${gRej} hunk${gRej === 1 ? "" : "s"}`, "amber")}>Request changes</Button><Button variant="primary" icon={IconCheck} onClick={() => toast(`Accepted ${gAcc}/${REVIEW_TOTAL_HUNKS} hunks · ${gRej} rejected · ${threads.length} commented`, "mint")}>Accept {gAcc}/{REVIEW_TOTAL_HUNKS} hunks</Button></div>} />
      <div className="flex min-h-0 flex-1">
        <div className="flex w-[300px] shrink-0 flex-col border-r border-line-soft bg-sunken">
          <div className="flex items-center gap-2 border-b border-line-soft px-3 py-2.5"><span className="font-mono text-[9.5px] tracking-[.14em] text-ink-4 uppercase">changed</span><span className="font-mono text-[10px] text-mint">+{REVIEW_FILES.reduce((a, f) => a + f.adds, 0)}</span><span className="font-mono text-[10px] text-rose">−{REVIEW_FILES.reduce((a, f) => a + f.dels, 0)}</span><span className="ml-auto font-mono text-[9.5px] text-ink-4">{REVIEW_FILES.length} files</span></div>
          <div className="scroll-thin flex-1 overflow-y-auto p-1.5">
            {REVIEW_FILES.map((f, i) => { const on = i === active; const s = staged.includes(f.path); return (
              <div key={f.path} onClick={() => setActive(i)} className={cn("flex cursor-pointer items-start gap-2 rounded-[8px] px-2 py-2 transition-colors", on ? "bg-raise shadow-e1" : "hover:bg-raise/60")}>
                <button onClick={(e) => { e.stopPropagation(); setStaged((x) => x.includes(f.path) ? x.filter((p) => p !== f.path) : [...x, f.path]); }} aria-label={s ? `Unstage ${f.path}` : `Stage ${f.path}`} className={cn("mt-[2px] flex size-[15px] shrink-0 cursor-pointer items-center justify-center rounded-[3px] border", s ? "border-iris bg-iris text-on-iris" : "border-line-strong")}>{s && <IconCheck size={9} />}</button>
                <div className="min-w-0 flex-1"><p className="truncate font-mono text-[11px] text-ink-2">{f.path.split("/").pop()}</p><p className="truncate font-mono text-[9.5px] text-ink-4">{f.path.split("/").slice(0, -1).join("/")}</p></div>
                <div className="text-right font-mono text-[9.5px]"><div className="text-mint">+{f.adds}</div><div className="text-rose">−{f.dels}</div></div>
              </div>); })}
          </div>
          <div className="border-t border-line-soft p-2.5">
            <div className="font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">checkpoints</div>
            {[["a91f3c2", "before bridge rewrite", "09:47"], ["77bd0e4", "channel + flush added", "09:44"], ["1c3a9f8", "session start", "09:41"]].map(([h, l, t], i) => (
              <div key={h} className="group mt-2 flex items-center gap-2"><i className={cn("size-[6px] rounded-full", i === 0 ? "bg-iris" : "bg-line-strong")} /><div className="min-w-0 flex-1"><p className="truncate text-[11.5px] text-ink-2">{l}</p><p className="font-mono text-[9.5px] text-ink-4">{h} · {t}</p></div><button className="cursor-pointer font-mono text-[9.5px] text-iris-soft opacity-0 group-hover:opacity-100">restore</button></div>
            ))}
          </div>
        </div>
        <div className="scroll-thin flex-1 overflow-y-auto bg-base">
          <div className="sticky top-0 z-20 flex items-center gap-2 border-b border-line-soft bg-base/90 px-4 py-2.5 backdrop-blur-xl"><IconFile size={12} className="text-ink-4" /><span className="font-mono text-[11.5px] text-ink">{file.path}</span><span className="font-mono text-[10px] text-mint">+{file.adds}</span><span className="font-mono text-[10px] text-rose">−{file.dels}</span><div className="ml-auto flex gap-1.5"><Button variant="ghost" size="xs" onClick={() => hunks.length > 0 && toggleComposer(hkey(0))}>Comment</Button><Button variant="ghost" size="xs" onClick={() => { const r = hunks.findIndex((_, i) => verdicts[hkey(i)] === "rejected"); toggleComposer(hkey(r === -1 ? 0 : r)); }}>Ask agent to fix</Button><Button variant="ghost" size="xs">Open in editor</Button></div></div>
          <div className="p-4">
            <div className="mb-3 flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[10px]">
              <span className="text-ink-3">{hunks.length} hunk{hunks.length === 1 ? "" : "s"}</span><span className="text-ink-4">·</span>
              <span className="text-mint">{fAcc} accepted</span><span className="text-ink-4">·</span>
              <span className="text-rose">{fRej} rejected</span><span className="text-ink-4">·</span>
              <span className="text-ink-4">{hunks.length - fAcc - fRej} pending</span>
              {fileThreads.length > 0 && (<><span className="text-ink-4">·</span><span className="text-amber">{fileThreads.length} thread{fileThreads.length === 1 ? "" : "s"}</span></>)}
            </div>
            {hunks.map((h, i) => { const key = hkey(i); const v = verdicts[key]; return (
              <div key={key} className={cn("group/hunk mb-2 overflow-clip rounded-[8px] border bg-well transition-all", v === "accepted" ? "border-mint/30" : v === "rejected" ? "border-rose/25 opacity-60" : "border-line-soft hover:border-line-strong")}>
                <div className="sticky top-[45px] z-10 flex items-center gap-2 border-b border-line-soft bg-well/80 px-2.5 py-[5px] backdrop-blur">
                  <span className="shrink-0 font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">hunk {i + 1}</span>
                  <span className="shrink-0 font-mono text-[9.5px]"><span className="text-mint">+{h.adds}</span> <span className="text-rose">−{h.dels}</span></span>
                  <span className="min-w-0 flex-1 truncate font-mono text-[10.5px] text-ink-3">{h.header}</span>
                  {v && <button onDoubleClick={() => resetVerdict(key)} title="Double-click to reset" aria-label={`Reset hunk ${i + 1}`} className={cn("inline-flex shrink-0 cursor-pointer items-center rounded-full px-1.5 py-[1px] font-mono text-[9.5px] transition-opacity hover:opacity-75", v === "accepted" ? "bg-mint-tint text-mint" : "bg-rose-tint text-rose")}>{v === "accepted" ? "accepted ✓" : "rejected"}</button>}
                  <div className="ml-auto flex shrink-0 items-center gap-1.5 opacity-0 transition-opacity duration-150 group-hover/hunk:opacity-100 group-focus-within/hunk:opacity-100">
                    <Button variant="success" size="xs" icon={IconCheck} aria-label={`Accept hunk ${i + 1}`} onClick={() => setVerdict(key, "accepted")}>Accept</Button>
                    <Button variant="danger" size="xs" icon={IconX} aria-label={`Reject hunk ${i + 1}`} onClick={() => setVerdict(key, "rejected")}>Reject</Button>
                    <Button variant="ghost" size="xs" icon={IconPencil} aria-label={`Comment on hunk ${i + 1}`} onClick={() => toggleComposer(key)}>Comment</Button>
                  </div>
                </div>
                <div className="[&>div]:rounded-none [&>div]:border-0 [&>div>div:first-child]:hidden">
                  <DiffView key={layout + key} file={{ path: file.path, adds: h.adds, dels: h.dels, lang: file.lang, lines: h.lines }} maxLines={40} mode={layout} />
                </div>
                {commenting === key && (
                  <div className="flex items-center gap-1.5 border-t border-line-soft bg-well px-2.5 py-2">
                    <Input autoFocus value={draft} onChange={(e) => setDraft(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); sendComment(key); } else if (e.key === "Escape") setCommenting(null); }} placeholder="Comment on this hunk — ⏎ to send" aria-label={`Comment on hunk ${i + 1}`} className="h-[26px] flex-1 text-[12px]" />
                    <Button variant="primary" size="xs" onClick={() => sendComment(key)}>Ask agent to fix</Button>
                  </div>
                )}
                {threads.filter((t) => t.hunkKey === key).map((t) => <HunkThread key={t.id} thread={t} />)}
              </div>); })}
            <div className="mt-3 rounded-[10px] border border-line-soft bg-raise p-3"><div className="flex items-center gap-2"><IconSpark size={12} className="text-iris-soft" /><span className="text-[12.5px] font-medium text-ink">Why this change</span><Badge tone="iris" mono className="text-[9.5px]">claude-code</Badge></div><p className="mt-1.5 text-[12px] leading-[1.6] text-ink-3">Replaces the one-shot snapshot with a streaming <code className="font-mono text-cyan">SessionChannel</code>. Editor sockets negotiate capabilities and every editor-originated patch enters the same reviewer, so review, checkpoint and rewind behave identically regardless of who made the edit.</p></div>
          </div>
        </div>
      </div>
    </div>
  );
}
