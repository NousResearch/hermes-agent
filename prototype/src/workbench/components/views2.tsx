"use client";

import { useState } from "react";
import { cn } from "../utils/cn";
import { agentById } from "../data/catalog";
import { CHANGED_FILES, COMMITS, PRS, REPOS, TASKS, type Task } from "../data/extra";
import { useApp } from "../lib/app";
import { AgentMark, IconBolt, IconBranch, IconCheck, IconChevronDown, IconEye, IconFile, IconFolder, IconGit, IconList, IconPlus, IconRefresh, IconSearch, IconSpark, IconWarning, IconX, IconGrid, IconArrowUp } from "./Icons";
import { Badge, Bar, Button, IconButton, Input, Segmented, Stat, Tip, toneBg, toneDot, toneText, type Tone } from "./ui";
import { Grid, PageHeader } from "./Titlebar";

/* ====================================================================
   TASKS — a work graph, not a ticket list.
   Gap vs. Linear/Jira/Cursor plan: tasks here know who (agent or human)
   does them, what "done" means (acceptance criteria), what they cost,
   what blocks them, and which live session is executing them.
   ==================================================================== */
const COLS: { id: Task["status"]; label: string; tone: Tone }[] = [{ id: "backlog", label: "Backlog", tone: "neutral" }, { id: "ready", label: "Ready", tone: "sky" }, { id: "running", label: "In progress", tone: "cyan" }, { id: "review", label: "Review", tone: "amber" }, { id: "done", label: "Done", tone: "mint" }];
const prioTone: Record<Task["priority"], Tone> = { p0: "rose", p1: "amber", p2: "sky", p3: "neutral" };
const srcLabel: Record<Task["source"], string> = { linear: "LIN", github: "GH", manual: "ME", agent: "AGT" };

function TaskCard({ t, onOpen, active }: { t: Task; onOpen: () => void; active: boolean }) {
  const a = t.assignee ? agentById(t.assignee) : null;
  const blocked = t.deps?.some((d) => TASKS.find((x) => x.id === d)?.status !== "done");
  return (
    <button onClick={onOpen} className={cn("w-full cursor-pointer space-y-2 rounded-[10px] border bg-raise p-2.5 text-left transition-all hover:shadow-e2", active ? "border-iris/50 shadow-glow-iris" : "border-line-soft hover:border-line-strong")}>
      <div className="flex items-center gap-1.5"><span className={cn("rounded-[3px] px-1 font-mono text-[8.5px] font-bold uppercase", toneBg[prioTone[t.priority]])}>{t.priority}</span><span className="font-mono text-[9.5px] text-ink-4">{t.key}</span><span className="ml-auto rounded-[3px] border border-line-soft px-1 font-mono text-[8.5px] text-ink-4">{srcLabel[t.source]}</span></div>
      <p className="text-[12.5px] leading-[1.45] font-medium text-ink">{t.title}</p>
      <div className="flex items-center gap-1.5">
        {t.human ? <span className="flex size-[16px] items-center justify-center rounded-[4px] bg-hover font-mono text-[8px] font-bold text-ink-2">DV</span> : a ? <AgentMark glyph={a.glyph} from={a.from} to={a.to} size={16} /> : <span className="flex size-[16px] items-center justify-center rounded-[4px] border border-dashed border-line-strong text-ink-4"><IconPlus size={9} /></span>}
        <span className="truncate font-mono text-[9.5px] text-ink-4">{t.human ? "you" : a?.name ?? "auto-assign"}</span>
        {blocked && t.status !== "done" && <Tip label="Blocked by an unfinished dependency"><span className="flex items-center gap-1 font-mono text-[9px] text-rose"><IconWarning size={9} /> blocked</span></Tip>}
        <span className="ml-auto font-mono text-[9.5px] text-ink-4">{t.estimate}</span>
        {t.cost && <span className="font-mono text-[9.5px] text-ink-3">{t.cost}</span>}
      </div>
      {t.status === "running" && <Bar indeterminate tone="cyan" height={2} />}
    </button>
  );
}

export function TasksView({ onOpenSession }: { onOpenSession: (id: string) => void }) {
  const { toast, setTodos } = useApp();
  const [layout, setLayout] = useState<"board" | "list">("board");
  const [sel, setSel] = useState<string | null>("k1");
  const [q, setQ] = useState("");
  const [tasks, setTasks] = useState(TASKS);
  const task = tasks.find((t) => t.id === sel);
  const filtered = tasks.filter((t) => (t.title + t.key + t.tags.join()).toLowerCase().includes(q.toLowerCase()));
  const move = (id: string, status: Task["status"]) => setTasks((ts) => ts.map((t) => (t.id === id ? { ...t, status } : t)));
  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <PageHeader eyebrow="work graph" title="Tasks" sub="Every task knows who does it (agent or you), what done means, what it costs, and what blocks it. Import from Linear/GitHub or let the agent decompose."
        right={<div className="flex flex-wrap items-center gap-2"><div className="relative"><IconSearch size={12} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-ink-4" /><Input placeholder="Filter tasks…" value={q} onChange={(e) => setQ(e.target.value)} className="w-[200px] pl-7" /></div><Segmented value={layout} onChange={setLayout} items={[{ value: "board", label: "Board", icon: IconGrid }, { value: "list", label: "List", icon: IconList }]} /><Button variant="secondary" icon={IconRefresh} onClick={() => toast("Synced 14 issues from Linear · 6 from GitHub", "mint")}>Sync</Button><Button variant="secondary" icon={IconSpark} onClick={() => toast("Agent decomposed 1 epic into 4 tasks", "iris")}>Decompose</Button><Button variant="primary" icon={IconPlus}>New task</Button></div>} />
      <div className="flex min-h-0 flex-1">
        <div className="scroll-thin flex-1 overflow-auto p-4">
          <Grid className="mb-4 grid-cols-2 lg:grid-cols-4">
            <Stat label="in progress" value={tasks.filter((t) => t.status === "running").length} sub="2 agents working" tone="cyan" /><Stat label="needs you" value={tasks.filter((t) => t.human && t.status !== "done").length} sub="human-only tasks" tone="amber" /><Stat label="blocked" value="2" sub="waiting on PLAT-482" tone="rose" /><Stat label="spent this week" value="$12.10" sub="across 6 tasks" chart={[1, 2, 3, 5, 7, 9, 12]} />
          </Grid>
          {layout === "board" ? (
            <div className="grid min-w-[900px] grid-cols-5 gap-3">
              {COLS.map((c) => { const items = filtered.filter((t) => t.status === c.id); return (
                <div key={c.id} className="flex flex-col rounded-[11px] border border-line-soft bg-sunken">
                  <div className="flex items-center gap-2 border-b border-line-soft px-3 py-2.5"><i className={cn("size-[6px] rounded-full", toneDot[c.tone])} /><span className="text-[12.5px] font-semibold text-ink">{c.label}</span><span className="rounded-full bg-raise px-1.5 font-mono text-[9.5px] text-ink-4">{items.length}</span></div>
                  <div className="flex-1 space-y-2 p-2">{items.map((t) => <TaskCard key={t.id} t={t} onOpen={() => setSel(t.id)} active={sel === t.id} />)}{items.length === 0 && <div className="rounded-[9px] border border-dashed border-line px-3 py-5 text-center text-[11px] text-ink-4">drop here</div>}</div>
                </div>); })}
            </div>
          ) : (
            <div className="overflow-hidden rounded-[11px] border border-line-soft bg-raise">
              {filtered.map((t) => { const a = t.assignee ? agentById(t.assignee) : null; return (
                <button key={t.id} onClick={() => setSel(t.id)} className={cn("flex w-full cursor-pointer items-center gap-3 border-b border-line-soft/70 px-3 py-2.5 text-left last:border-0 hover:bg-hover/60", sel === t.id && "bg-iris-tint/50")}>
                  <span className={cn("w-6 rounded-[3px] text-center font-mono text-[9px] font-bold uppercase", toneText[prioTone[t.priority]])}>{t.priority}</span><span className="w-[76px] font-mono text-[10px] text-ink-4">{t.key}</span>
                  <span className="min-w-0 flex-1 truncate text-[12.5px] text-ink">{t.title}</span>
                  <Badge tone={COLS.find((c) => c.id === t.status)!.tone} mono className="text-[9px]">{t.status}</Badge>
                  <span className="flex w-[110px] items-center gap-1.5 font-mono text-[10px] text-ink-3">{a && <AgentMark glyph={a.glyph} from={a.from} to={a.to} size={14} />}{t.human ? "you" : a?.name ?? "—"}</span>
                  <span className="w-10 text-right font-mono text-[10px] text-ink-4">{t.estimate}</span>
                </button>); })}
            </div>
          )}
        </div>
        {task && (
          <aside className="flex w-[340px] shrink-0 flex-col border-l border-line-soft bg-sunken">
            <div className="flex items-center gap-2 border-b border-line-soft px-3 py-2.5"><span className="font-mono text-[10px] text-ink-4">{task.key}</span><Badge tone={prioTone[task.priority]} mono className="text-[9px]">{task.priority}</Badge><IconButton icon={IconX} label="Close" size={24} className="ml-auto" onClick={() => setSel(null)} /></div>
            <div className="scroll-thin flex-1 space-y-3 overflow-y-auto p-3">
              <h3 className="text-[14px] leading-[1.4] font-semibold text-ink">{task.title}</h3>
              <div className="flex flex-wrap gap-1">{task.tags.map((t) => <span key={t} className="rounded-[4px] bg-hover px-1.5 py-[1px] font-mono text-[9.5px] text-ink-3">#{t}</span>)}</div>
              <div className="rounded-[9px] border border-line-soft bg-raise p-2.5">
                <div className="font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">status</div>
                <div className="mt-1.5 flex flex-wrap gap-1">{COLS.map((c) => <button key={c.id} onClick={() => move(task.id, c.id)} className={cn("cursor-pointer rounded-full border px-2 py-[2px] text-[10.5px]", task.status === c.id ? "border-iris/50 bg-iris-tint text-iris-soft" : "border-line text-ink-3 hover:text-ink")}>{c.label}</button>)}</div>
              </div>
              <div className="rounded-[9px] border border-line-soft bg-raise p-2.5">
                <div className="font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">done means</div>
                {task.criteria.map((c) => <label key={c} className="mt-1.5 flex cursor-pointer items-start gap-2 text-[12px] text-ink-2"><span className="mt-[2px] flex size-[14px] shrink-0 items-center justify-center rounded-[3px] border border-line-strong">{task.status === "done" && <IconCheck size={9} className="text-mint" />}</span>{c}</label>)}
              </div>
              {task.deps && <div className="rounded-[9px] border border-line-soft bg-raise p-2.5"><div className="font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">blocked by</div>{task.deps.map((d) => { const dt = tasks.find((x) => x.id === d)!; return <button key={d} onClick={() => setSel(d)} className="mt-1.5 flex w-full cursor-pointer items-center gap-2 text-left text-[12px] text-ink-2 hover:text-ink"><i className={cn("size-[6px] rounded-full", dt.status === "done" ? "bg-mint" : "bg-rose")} /><span className="font-mono text-[10px] text-ink-4">{dt.key}</span><span className="truncate">{dt.title}</span></button>; })}</div>}
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-[9px] border border-line-soft bg-raise p-2.5"><div className="font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">estimate</div><div className="mt-1 text-[14px] font-semibold text-ink">{task.estimate}</div></div>
                <div className="rounded-[9px] border border-line-soft bg-raise p-2.5"><div className="font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">spent</div><div className="mt-1 text-[14px] font-semibold text-ink">{task.cost ?? "$0"}</div></div>
              </div>
            </div>
            <div className="space-y-1.5 border-t border-line-soft p-3">
              {task.sessionId ? <Button variant="primary" className="w-full" icon={IconEye} onClick={() => onOpenSession(task.sessionId!)}>Open live session</Button>
                : <Button variant="primary" className="w-full" icon={IconBolt} onClick={() => { move(task.id, "running"); toast(`Dispatched ${task.key} to ${task.assignee ? agentById(task.assignee).name : "best agent"}`, "iris"); }}>{task.human ? "Start" : "Run with agent"}</Button>}
              <div className="flex gap-1.5"><Button variant="outline" size="xs" className="flex-1" onClick={() => { setTodos((t) => [{ id: `tt${Date.now()}`, text: task.title, state: "todo", by: "you" }, ...t]); toast("Added to your todo list", "mint"); }}>Add to my todos</Button><Button variant="outline" size="xs" className="flex-1">Split</Button></div>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}

/* ================================ GIT ================================ */
export function GitView({ onReview }: { onReview: () => void }) {
  const { toast } = useApp();
  const [repoId, setRepoId] = useState("g1");
  const [tab, setTab] = useState<"changes" | "commits" | "prs" | "branches">("changes");
  const [connect, setConnect] = useState(false);
  const [url, setUrl] = useState("");
  const repo = REPOS.find((r) => r.id === repoId)!;
  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <PageHeader eyebrow="source control" title="Git" sub="Attach local folders or clone remotes. Agents commit on branches, open PRs, and you merge — or let a rule do it when checks are green."
        right={<div className="flex items-center gap-2"><Button variant="secondary" icon={IconRefresh} onClick={() => toast(`Fetched ${repo.name} · ↑${repo.ahead} ↓${repo.behind}`, "mint")}>Sync</Button><Button variant="primary" icon={IconPlus} onClick={() => setConnect((c) => !c)}>Connect repo</Button></div>} />
      <div className="flex min-h-0 flex-1">
        <div className="flex w-[280px] shrink-0 flex-col border-r border-line-soft bg-sunken">
          <div className="border-b border-line-soft px-3 py-2.5 font-mono text-[9.5px] tracking-[.14em] text-ink-4 uppercase">repositories</div>
          <div className="space-y-1 p-1.5">
            {REPOS.map((r) => (
              <button key={r.id} onClick={() => setRepoId(r.id)} className={cn("w-full cursor-pointer rounded-[8px] px-2.5 py-2 text-left transition-colors", repoId === r.id ? "bg-raise shadow-e1" : "hover:bg-raise/60")}>
                <div className="flex items-center gap-2"><IconGit size={12} className="text-ink-4" /><span className="truncate text-[12.5px] font-medium text-ink">{r.name}</span><span className="ml-auto rounded-[3px] border border-line-soft px-1 font-mono text-[8.5px] text-ink-4 uppercase">{r.provider}</span></div>
                <div className="mt-1 flex items-center gap-2 font-mono text-[9.5px] text-ink-4"><IconBranch size={9} /><span className="truncate text-iris-soft">{r.branch}</span><span className="ml-auto">↑{r.ahead} ↓{r.behind}</span>{r.dirty > 0 && <span className="text-amber">●{r.dirty}</span>}</div>
                <div className="mt-0.5 truncate font-mono text-[9.5px] text-ink-4">{r.path}</div>
              </button>
            ))}
          </div>
          {connect && (
            <div className="animate-slide-down m-2 space-y-2 rounded-[10px] border border-iris/30 bg-raise p-3">
              <div className="text-[12.5px] font-semibold text-ink">Connect a repository</div>
              <Button variant="secondary" className="w-full justify-start" icon={IconFolder} onClick={() => { toast("Attached ~/code/new-project", "mint"); setConnect(false); }}>Attach local folder…</Button>
              <div className="flex items-center gap-2 py-0.5"><span className="h-px flex-1 bg-line" /><span className="font-mono text-[9px] text-ink-4">or clone</span><span className="h-px flex-1 bg-line" /></div>
              <Input placeholder="github.com/org/repo or git@…" value={url} onChange={(e) => setUrl(e.target.value)} />
              <div className="flex gap-1.5"><Button variant="primary" size="xs" className="flex-1" disabled={!url} onClick={() => { toast(`Cloning ${url}…`, "iris"); setConnect(false); setUrl(""); }}>Clone</Button><Button variant="ghost" size="xs" onClick={() => setConnect(false)}>Cancel</Button></div>
              <div className="flex gap-1">{["github", "gitlab", "bitbucket"].map((p) => <span key={p} className="rounded-[4px] bg-hover px-1.5 py-[1px] font-mono text-[9px] text-ink-3">{p}</span>)}</div>
            </div>
          )}
          <div className="mt-auto border-t border-line-soft p-3">
            <div className="font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">auto-merge rule</div>
            <p className="mt-1 text-[11.5px] leading-[1.5] text-ink-3">Merge agent PRs when all checks pass <em>and</em> one human approved.</p>
          </div>
        </div>

        <div className="flex min-w-0 flex-1 flex-col bg-base">
          <div className="flex items-center gap-3 border-b border-line-soft px-4 py-2.5">
            <IconGit size={14} className="text-ink-3" /><span className="font-mono text-[12px] text-ink">{repo.name}</span><span className="h-3 w-px bg-line" /><IconBranch size={12} className="text-ink-4" /><span className="font-mono text-[12px] text-iris-soft">{repo.branch}</span>
            <Badge tone={repo.dirty ? "amber" : "mint"} mono dot className="text-[9px]">{repo.dirty ? `${repo.dirty} uncommitted` : "clean"}</Badge>
            <div className="ml-auto flex gap-1.5"><Button variant="secondary" size="xs" icon={IconArrowUp} onClick={() => toast(`Pushed ${repo.ahead} commits`, "mint")}>Push {repo.ahead}</Button><Button variant="secondary" size="xs" icon={IconChevronDown} onClick={() => toast(`Pulled ${repo.behind} commits`, "mint")}>Pull {repo.behind}</Button><Button variant="primary" size="xs" icon={IconGit} onClick={() => toast("PR #483 opened from feat/ide-bridge", "iris")}>Open PR</Button></div>
          </div>
          <div className="flex items-center gap-0.5 border-b border-line-soft px-3">
            {([["changes", `Changes · ${CHANGED_FILES.length}`], ["commits", "Commits"], ["prs", `Pull requests · ${PRS.filter((p) => p.status !== "merged").length}`], ["branches", "Branches"]] as const).map(([v, l]) => <button key={v} onClick={() => setTab(v)} className={cn("relative cursor-pointer px-3 py-2.5 text-[12.5px] font-medium", tab === v ? "text-ink" : "text-ink-3 hover:text-ink-2")}>{l}{tab === v && <span className="absolute inset-x-2 bottom-0 h-[2px] rounded-full bg-iris" />}</button>)}
          </div>
          <div className="scroll-thin flex-1 overflow-y-auto p-4">
            {tab === "changes" && (
              <div className="space-y-3">
                <div className="overflow-hidden rounded-[11px] border border-line-soft bg-raise">
                  {CHANGED_FILES.map((f) => <div key={f.path} className="flex items-center gap-2.5 border-b border-line-soft/70 px-3 py-2 last:border-0 hover:bg-hover/50"><span className={cn("w-4 text-center font-mono text-[11px] font-bold", f.status === "A" ? "text-mint" : "text-amber")}>{f.status}</span><span className="min-w-0 flex-1 truncate font-mono text-[11.5px] text-ink-2">{f.path}</span><span className={cn("rounded-[3px] px-1 font-mono text-[8.5px] uppercase", f.origin === "agent" ? "bg-iris-tint text-iris-soft" : f.origin === "editor" ? "bg-sky-tint text-sky" : "bg-hover text-ink-3")}>{f.origin}</span><span className="font-mono text-[10px] text-mint">+{f.adds}</span><span className="font-mono text-[10px] text-rose">−{f.dels}</span></div>)}
                </div>
                <div className="flex items-center gap-2 rounded-[11px] border border-line-soft bg-raise p-3">
                  <Input placeholder="Commit message — leave blank to let the agent write it" className="flex-1" />
                  <Button variant="secondary" icon={IconEye} onClick={onReview}>Review diff</Button>
                  <Button variant="primary" icon={IconCheck} onClick={() => toast("Committed 5 files · message drafted by agent", "mint")}>Commit all</Button>
                </div>
              </div>
            )}
            {tab === "commits" && (
              <div className="overflow-hidden rounded-[11px] border border-line-soft bg-raise">
                {COMMITS.map((c) => <div key={c.hash} className="flex items-center gap-3 border-b border-line-soft/70 px-3 py-2.5 last:border-0 hover:bg-hover/50"><span className="font-mono text-[10.5px] text-iris-soft">{c.hash}</span><span className="min-w-0 flex-1 truncate text-[12.5px] text-ink">{c.msg}</span><span className={cn("rounded-[3px] px-1.5 py-[1px] font-mono text-[9px]", c.agent ? "bg-iris-tint text-iris-soft" : "bg-hover text-ink-3")}>{c.by}</span><span className="w-12 text-right font-mono text-[10px] text-ink-4">{c.files} files</span><span className="w-8 text-right font-mono text-[10px] text-ink-4">{c.when}</span></div>)}
              </div>
            )}
            {tab === "prs" && (
              <div className="space-y-2">
                {PRS.map((p) => (
                  <div key={p.num} className={cn("rounded-[11px] border bg-raise p-3", p.mergeable ? "border-mint/30" : "border-line-soft")}>
                    <div className="flex items-start gap-3">
                      <span className={cn("mt-0.5 flex size-[22px] shrink-0 items-center justify-center rounded-[6px]", p.status === "merged" ? "bg-plum-tint text-plum" : p.mergeable ? "bg-mint-tint text-mint" : "bg-hover text-ink-3")}><IconGit size={12} /></span>
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2"><span className="text-[13px] font-semibold text-ink">{p.title}</span><span className="font-mono text-[10.5px] text-ink-4">#{p.num}</span>{p.status === "merged" && <Badge tone="plum" mono className="text-[9px]">merged</Badge>}</div>
                        <div className="mt-1 flex flex-wrap items-center gap-3 font-mono text-[10px] text-ink-4"><span className="text-iris-soft">{p.branch}</span><span>by {p.author}</span><span className="text-mint">+{p.adds}</span><span className="text-rose">−{p.dels}</span></div>
                        <div className="mt-2 flex flex-wrap items-center gap-2">
                          <span className="flex items-center gap-1.5 rounded-[6px] border border-line-soft bg-well px-2 py-1 font-mono text-[10px]"><span className="text-mint">✓ {p.checks.pass}</span>{p.checks.fail > 0 && <span className="text-rose">✗ {p.checks.fail}</span>}{p.checks.pending > 0 && <span className="text-amber">• {p.checks.pending}</span>}<span className="text-ink-4">checks</span></span>
                          <span className={cn("rounded-[6px] border border-line-soft bg-well px-2 py-1 font-mono text-[10px]", p.reviews.includes("approved") ? "text-mint" : p.reviews.includes("changes") ? "text-rose" : "text-amber")}>{p.reviews}</span>
                        </div>
                      </div>
                      {p.status !== "merged" && (
                        <div className="flex shrink-0 flex-col gap-1.5">
                          <Button variant={p.mergeable ? "primary" : "secondary"} size="xs" disabled={!p.mergeable} icon={IconCheck} onClick={() => toast(`Merged #${p.num} (squash)`, "mint")}>Merge</Button>
                          <Button variant="ghost" size="xs" onClick={onReview}>Review</Button>
                          {p.checks.fail > 0 && <Button variant="ghost" size="xs" onClick={() => toast("Agent is fixing the failing check", "iris")}>Ask agent to fix</Button>}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {tab === "branches" && (
              <div className="overflow-hidden rounded-[11px] border border-line-soft bg-raise">
                {[["feat/ide-bridge", "claude-code", "9m", true], ["refactor/ledger", "codex", "18m", false], ["refactor/webhooks", "claude-code", "25m", false], ["chore/dead-code", "aro", "1h", false], ["main", "—", "3h", false]].map(([b, by, t, cur]) => <div key={b as string} className="flex items-center gap-3 border-b border-line-soft/70 px-3 py-2.5 last:border-0 hover:bg-hover/50"><IconBranch size={12} className={cur ? "text-iris-soft" : "text-ink-4"} /><span className={cn("min-w-0 flex-1 truncate font-mono text-[12px]", cur ? "text-iris-soft" : "text-ink-2")}>{b as string}</span><span className="font-mono text-[10px] text-ink-4">{by as string}</span><span className="w-8 text-right font-mono text-[10px] text-ink-4">{t as string}</span><Button variant="ghost" size="xs">{cur ? "current" : "checkout"}</Button></div>)}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ================================ BROWSER ================================ */
export function BrowserView({ onAskAgent }: { onAskAgent: (t: string) => void }) {
  const { toast } = useApp();
  const [url, setUrl] = useState("http://localhost:5173/review");
  const [device, setDevice] = useState<"desktop" | "tablet" | "mobile">("desktop");
  const [pick, setPick] = useState(false);
  const [console_, setConsole] = useState(true);
  const width = device === "desktop" ? "100%" : device === "tablet" ? 820 : 390;
  return (
    <div className="flex flex-1 flex-col overflow-hidden bg-base">
      <div className="flex items-center gap-2 border-b border-line-soft px-3 py-2">
        <div className="flex items-center gap-0.5"><IconButton icon={() => <IconChevronDown size={14} className="rotate-90" />} label="Back" size={28} /><IconButton icon={() => <IconChevronDown size={14} className="-rotate-90" />} label="Forward" size={28} /><IconButton icon={IconRefresh} label="Reload" size={28} onClick={() => toast("Reloaded", "iris")} /></div>
        <div className="flex h-[30px] min-w-0 flex-1 items-center gap-2 rounded-[8px] border border-line bg-sunken px-2.5 focus-within:border-iris/50"><span className="size-[6px] rounded-full bg-mint" /><input value={url} onChange={(e) => setUrl(e.target.value)} className="min-w-0 flex-1 bg-transparent font-mono text-[12px] text-ink focus:outline-none" /><span className="font-mono text-[9.5px] text-ink-4">dev server · hmr</span></div>
        <Segmented value={device} onChange={setDevice} items={[{ value: "desktop", label: "Desktop" }, { value: "tablet", label: "Tablet" }, { value: "mobile", label: "Mobile" }]} />
        <Tip label="Pick an element → send to agent as context"><Button variant={pick ? "primary" : "secondary"} icon={IconSearch} onClick={() => setPick((p) => !p)}>{pick ? "Picking…" : "Pick element"}</Button></Tip>
        <Button variant="secondary" icon={IconEye} onClick={() => { toast("Screenshot attached to session", "mint"); }}>Screenshot</Button>
        <Button variant="primary" icon={IconSpark} onClick={() => onAskAgent("Open localhost:5173/review, reproduce the dropdown clipping in the file list, and fix the CSS.")}>Ask agent about this page</Button>
      </div>
      <div className="scroll-thin flex min-h-0 flex-1 flex-col">
        <div className="dot-bg flex flex-1 justify-center overflow-auto p-4">
          <div style={{ width }} className={cn("flex max-w-full flex-col overflow-hidden rounded-[12px] border border-line-strong bg-base shadow-e4 transition-all", pick && "cursor-crosshair ring-2 ring-iris/60")}>
            {/* mocked rendered page (the product's own review page) */}
            <div className="flex items-center gap-2 border-b border-line-soft bg-raise px-3 py-2"><span className="font-mono text-[10px] text-ink-4">aro / review · preview</span><span className="ml-auto rounded-[4px] bg-mint-tint px-1.5 font-mono text-[9px] text-mint">200 · 84ms</span></div>
            <div className="grid flex-1 gap-3 p-4 md:grid-cols-[220px_1fr]">
              <div className="space-y-1.5">
                {["server.ts", "attach.ts", "reviewer.ts"].map((f, i) => <div key={f} onClick={() => pick && (setPick(false), onAskAgent(`The file row "${f}" in the review list clips its dropdown on hover. Fix overflow so the menu is visible.`))} className={cn("flex items-center gap-2 rounded-[7px] border px-2.5 py-2 font-mono text-[11px]", i === 0 ? "border-iris/40 bg-iris-tint text-ink" : "border-line-soft bg-raise text-ink-2", pick && "hover:ring-2 hover:ring-iris")}><IconFile size={10} className="text-ink-4" />{f}<span className="ml-auto text-mint">+{[14, 41, 9][i]}</span></div>)}
              </div>
              <div className="space-y-2">
                <div className="skeleton h-5 w-2/3 rounded-[5px]" /><div className="skeleton h-3 w-full rounded-[5px]" /><div className="skeleton h-3 w-5/6 rounded-[5px]" />
                <div className="mt-3 rounded-[8px] border border-line-soft bg-sunken p-3 font-mono text-[11px] leading-[1.7]"><div className="diff-del px-2 text-rose">- socket.send(store.snapshot())</div><div className="diff-add px-2 text-mint">+ const client = attachEditor(socket, channel, {"{"}</div><div className="diff-add px-2 text-mint">+   onEdit: (patch) =&gt; reviewer.enqueue(patch),</div></div>
              </div>
            </div>
          </div>
        </div>
        <div className={cn("shrink-0 border-t border-line-soft bg-code transition-all", console_ ? "h-[150px]" : "h-[30px]")}>
          <button onClick={() => setConsole((c) => !c)} className="flex h-[30px] w-full cursor-pointer items-center gap-2 px-3 text-left"><span className="font-mono text-[9.5px] tracking-[.14em] text-ink-4 uppercase">console</span><Badge tone="rose" mono className="text-[9px]">1 error</Badge><Badge tone="amber" mono className="text-[9px]">2 warn</Badge><IconChevronDown size={12} className={cn("ml-auto text-ink-4 transition-transform", !console_ && "rotate-180")} /></button>
          {console_ && <pre className="scroll-thin h-[120px] overflow-auto px-3 pb-2 font-mono text-[11px] leading-[1.7] text-ink-3">
            <div><span className="text-ink-4">[hmr]</span> updated packages/review/src/FileList.tsx</div>
            <div className="text-amber">⚠ Each child in a list should have a unique "key" prop. <span className="text-ink-4">FileList.tsx:42</span></div>
            <div className="text-rose">✗ ResizeObserver loop completed with undelivered notifications. <span className="text-ink-4">DiffView.tsx:118</span> <button onClick={() => onAskAgent("Fix: ResizeObserver loop completed with undelivered notifications at DiffView.tsx:118")} className="ml-2 cursor-pointer rounded bg-iris-tint px-1.5 text-iris-soft">fix with agent</button></div>
            <div className="text-amber">⚠ Dropdown overflow clipped by parent (overflow: hidden) <span className="text-ink-4">FileRow.tsx:17</span></div>
          </pre>}
        </div>
      </div>
    </div>
  );
}
