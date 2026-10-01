"use client";

import { useState } from "react";
import { cn } from "../utils/cn";
import { AGENTS, agentById, type ModeId, type Session } from "../data/catalog";
import { MCP_SERVERS, RULES } from "../data/catalog";
import type { Project } from "../data/providers";
import { useApp } from "../lib/app";
import { IconBolt, IconCheck, IconChevronDown, IconFolder, IconGlobe, IconLayers, IconMcp, IconPlus, IconPulse, IconRefresh, IconSearch, IconSettings, IconShield, IconSpark, IconTerminal, IconTrash, IconX, IconZap } from "./Icons";
import { Badge, Bar, Button, Input, Segmented, Select, Toggle, Tip, type Tone } from "./ui";
import { Grid, PageHeader } from "./Titlebar";

/* ====================================================================
   SKILLS — bundled · hub · agent-authored (Aro-style skill system (agentskills.io))
   ==================================================================== */
type Skill = { id: string; name: string; cmd: string; desc: string; origin: "bundled" | "hub" | "agent"; on: boolean; runs: number; version: string; scope: string; tone: Tone };
const SKILLS: Skill[] = [
  { id: "sk1", name: "release-notes", cmd: "/release-notes", desc: "Draft release notes from merged PRs since the last tag.", origin: "bundled", on: true, runs: 42, version: "2.1.0", scope: "code", tone: "iris" },
  { id: "sk2", name: "migrate-vitest", cmd: "/migrate-vitest", desc: "Convert jest specs to vitest, fixing matchers and timers.", origin: "bundled", on: true, runs: 240, version: "1.4.2", scope: "code", tone: "cyan" },
  { id: "sk3", name: "sentry-triage", cmd: "/sentry-triage", desc: "Open tasks for new P0 issues with a reproduction outline.", origin: "hub", on: true, runs: 118, version: "0.9.1", scope: "both", tone: "mint" },
  { id: "sk4", name: "design", cmd: "/design", desc: "Generate editable UI artboards before writing implementation.", origin: "hub", on: false, runs: 17, version: "0.3.0", scope: "both", tone: "plum" },
  { id: "sk5", name: "db-migration", cmd: "/db-migration", desc: "Write a SQL migration, dry-run it, and produce a rollback.", origin: "agent", on: true, runs: 9, version: "0.1.0", scope: "code", tone: "amber" },
  { id: "sk6", name: "weekly-digest", cmd: "/weekly-digest", desc: "Summarise velocity, spend and incidents into a Slack post.", origin: "agent", on: true, runs: 26, version: "0.2.3", scope: "agent", tone: "sky" },
];

export function SkillsView() {
  const { toast } = useApp();
  const [q, setQ] = useState("");
  const [origin, setOrigin] = useState<"all" | Skill["origin"]>("all");
  const [skills, setSkills] = useState(SKILLS);
  const list = skills.filter((s) => (origin === "all" || s.origin === origin) && (s.name + s.desc + s.cmd).toLowerCase().includes(q.toLowerCase()));
  const toggle = (id: string) => setSkills((all) => all.map((s) => (s.id === id ? { ...s, on: !s.on } : s)));

  return (
    <div className="scroll-thin flex-1 overflow-y-auto">
      <PageHeader eyebrow="capability" title="Skills" sub="Versioned procedures any agent can run with a slash command or pick on its own. Bundled with Aro, installed from the hub, or authored by an agent during a session."
        right={<div className="flex items-center gap-2"><div className="relative"><IconSearch size={12} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-ink-4" /><Input placeholder="Filter skills…" value={q} onChange={(e) => setQ(e.target.value)} className="w-[190px] pl-7" /></div><Segmented value={origin} onChange={setOrigin} items={[{ value: "all", label: `All ${skills.length}` }, { value: "bundled", label: "Bundled" }, { value: "hub", label: "Hub" }, { value: "agent", label: "Agent" }]} /><Button variant="primary" icon={IconPlus} onClick={() => toast("Skill scaffold created at .aro/skills/new-skill.md", "mint")}>New skill</Button></div>} />

      <div className="space-y-4 p-5">
        <Grid className="grid-cols-2 lg:grid-cols-4">
          {[["Enabled", skills.filter((s) => s.on).length, `${skills.length} installed`], ["Runs this month", "452", "across 6 agents"], ["Agent-authored", skills.filter((s) => s.origin === "agent").length, "review before sharing"], ["Hub updates", "2", "sentry-triage · design"]].map(([l, v, s]) => (
            <div key={l as string} className="rounded-[11px] border border-line-soft bg-raise p-3">
              <div className="font-mono text-[9.5px] tracking-[.12em] text-ink-4 uppercase">{l}</div>
              <div className="mt-1 font-display text-[22px] leading-none font-semibold tracking-[-.02em] text-ink">{v}</div>
              <div className="mt-1 text-[11px] text-ink-3">{s}</div>
            </div>
          ))}
        </Grid>

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {list.map((s) => (
            <article key={s.id} className={cn("group relative flex flex-col overflow-hidden rounded-[12px] border bg-raise transition-all duration-200 hover:-translate-y-px hover:shadow-e3", s.on ? "border-line-soft hover:border-line-strong" : "border-dashed border-line opacity-80")}>
              <span className={cn("absolute inset-x-0 top-0 h-[2px] transition-opacity", s.on ? "opacity-100" : "opacity-25")} style={{ background: `linear-gradient(90deg, var(--color-${s.tone}), transparent)` }} />
              <div className="flex items-start gap-2.5 p-3">
                <span className={cn("flex size-[30px] shrink-0 items-center justify-center rounded-[9px] border border-line-soft bg-well", `text-${s.tone}`)}><IconSpark size={14} /></span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5"><h3 className="truncate font-display text-[13.5px] font-semibold text-ink">{s.name}</h3><Badge tone={s.origin === "agent" ? "amber" : s.origin === "hub" ? "sky" : "neutral"} mono className="text-[9px]">{s.origin}</Badge></div>
                  <code className="mt-0.5 block truncate font-mono text-[10.5px] text-iris-soft">{s.cmd}</code>
                </div>
                <Toggle checked={s.on} onChange={() => toggle(s.id)} size="sm" />
              </div>
              <p className="px-3 text-[11.5px] leading-[1.55] text-ink-3">{s.desc}</p>
              <div className="mt-2.5 flex items-center gap-2 px-3 font-mono text-[9.5px] text-ink-4">
                <span>v{s.version}</span><span>·</span><span>{s.runs} runs</span><span>·</span><span>{s.scope}</span>
              </div>
              <div className="mt-2.5 flex gap-1.5 border-t border-line-soft p-2.5">
                <Button variant="secondary" size="xs" icon={IconBolt} className="flex-1" onClick={() => toast(`Running ${s.cmd} in the active thread`, "iris")}>Run</Button>
                <Button variant="ghost" size="xs" onClick={() => toast("Opened skill manifest", "iris")}>Edit</Button>
                <Button variant="ghost" size="xs" icon={IconRefresh} onClick={() => toast(`${s.name} checked for updates`, "mint")} />
              </div>
            </article>
          ))}
        </div>

        <div className="grid gap-3 lg:grid-cols-[1.2fr_1fr]">
          <div className="overflow-hidden rounded-[12px] border border-line-soft bg-code">
            <div className="flex items-center gap-1.5 border-b border-line-soft bg-well px-3 py-2"><IconTerminal size={11} className="text-ink-4" /><span className="font-mono text-[10.5px] text-ink-3">.aro/skills/sentry-triage/SKILL.md</span><Badge tone="mint" mono className="ml-auto text-[9px]">valid</Badge></div>
            <pre className="scroll-thin overflow-x-auto px-3.5 py-3 font-mono text-[11px] leading-[1.75] text-ink-3">{`---
name: sentry-triage
description: Open tasks for new P0 issues
version: 0.9.1
tools: [sentry, linear, github]
triggers:
  - cron: "0 9 * * 1-5"
  - on: "issue.created"
permissions:
  write: ask        # never creates PRs unattended
  network: allow
---

1. Fetch unresolved P0 issues from the last 24h.
2. Group by root signature, drop duplicates.
3. For each group: open a Linear task with a repro
   outline, the failing release, and the owner.
4. Post a digest to #platform. Ask before paging.`}</pre>
          </div>
          <div className="rounded-[12px] border border-line-soft bg-raise p-3.5">
            <h3 className="font-display text-[13.5px] font-semibold text-ink">How skills resolve</h3>
            <div className="mt-2.5 space-y-2.5">
              {[["1", "Project scope first", ".aro/skills/ in the repo wins"], ["2", "Then user scope", "~/.aro/skills/ across projects"], ["3", "Then bundled + hub", "shipped with the app, updatable"], ["4", "Conflicts", "narrowest scope wins; ties show a picker"]].map(([n, t, d]) => (
                <div key={n} className="flex items-start gap-2.5">
                  <span className="flex size-[18px] shrink-0 items-center justify-center rounded-[5px] border border-iris/30 bg-iris-tint font-mono text-[9.5px] text-iris-soft">{n}</span>
                  <div><p className="text-[12px] font-medium text-ink">{t}</p><p className="font-mono text-[10px] text-ink-4">{d}</p></div>
                </div>
              ))}
            </div>
            <div className="mt-3 flex items-start gap-2 rounded-[9px] border border-amber/20 bg-amber-tint/50 px-2.5 py-2 text-[11px] leading-[1.5] text-amber"><IconShield size={11} className="mt-0.5 shrink-0" />Agent-authored skills are quarantined until you approve them. They can't request permissions their author didn't have.</div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ====================================================================
   CONNECTORS — MCP servers, tools, scopes
   ==================================================================== */
type Server = { name: string; tools: number; status: string; tone: Tone; transport: "stdio" | "sse" | "http"; command: string; scopes: ("code" | "agent")[] };
const SERVERS: Server[] = MCP_SERVERS.map((m, i) => ({
  name: m.name, tools: m.tools, status: m.status === "live" ? "connected" : "degraded",
  tone: m.status === "live" ? "mint" : "amber",
  transport: (["stdio", "sse", "http", "stdio", "sse"] as const)[i % 5],
  command: (["gh-mcp --token $GITHUB_TOKEN", "sentry-mcp --org acme", "linear-mcp", "pg-mcp --dsn $DATABASE_URL", "figma-mcp"])[i % 5],
  scopes: i % 3 === 0 ? ["code"] : i % 3 === 1 ? ["agent"] : ["code", "agent"],
}));

export function ConnectorsView() {
  const { toast } = useApp();
  const [servers, setServers] = useState(SERVERS);
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState("");
  const [cmd, setCmd] = useState("");
  const totalTools = servers.reduce((s, x) => s + x.tools, 0);

  return (
    <div className="scroll-thin flex-1 overflow-y-auto">
      <PageHeader eyebrow="integrations" title="Connectors" sub="MCP servers expose tools to every agent. Scope each one to Code, Agent or both — an agent can only call what its mode and the project allow."
        right={<div className="flex items-center gap-2"><Badge tone="mint" mono dot>{servers.filter((s) => s.status === "connected").length} connected</Badge><Badge tone="neutral" mono>{totalTools} tools</Badge><Button variant="primary" icon={IconPlus} onClick={() => setAdding((a) => !a)}>Add server</Button></div>} />

      <div className="space-y-4 p-5">
        {adding && (
          <div className="animate-slide-down rounded-[12px] border border-iris/30 bg-iris-tint/40 p-3.5">
            <div className="flex items-center gap-2"><IconMcp size={13} className="text-iris-soft" /><h3 className="font-display text-[13.5px] font-semibold text-ink">Add an MCP server</h3><span className="ml-auto font-mono text-[10px] text-ink-4">stdio · sse · streamable-http</span></div>
            <div className="mt-3 grid gap-2.5 sm:grid-cols-2">
              <div><label className="mb-1 block font-mono text-[9.5px] tracking-[.12em] text-ink-4 uppercase">name</label><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="github" /></div>
              <div><label className="mb-1 block font-mono text-[9.5px] tracking-[.12em] text-ink-4 uppercase">command</label><Input value={cmd} onChange={(e) => setCmd(e.target.value)} placeholder="gh-mcp --token $GITHUB_TOKEN" className="font-mono text-[11.5px]" /></div>
            </div>
            <div className="mt-3 flex items-center gap-2">
              <span className="text-[11.5px] text-ink-3">Scope</span>
              <Segmented value="both" onChange={() => {}} items={[{ value: "code", label: "Code" }, { value: "agent", label: "Agent" }, { value: "both", label: "Both" }]} />
              <div className="ml-auto flex gap-2"><Button variant="ghost" onClick={() => setAdding(false)}>Cancel</Button><Button variant="primary" icon={IconCheck} disabled={!name.trim()} onClick={() => { setServers((s) => [{ name: name.trim(), tools: 4, status: "connected", tone: "mint", transport: "stdio", command: cmd || "npx -y mcp", scopes: ["code", "agent"] }, ...s]); setAdding(false); setName(""); setCmd(""); toast(`${name} connected · 4 tools discovered`, "mint"); }}>Connect</Button></div>
            </div>
          </div>
        )}

        <div className="grid gap-3 md:grid-cols-2">
          {servers.map((s) => (
            <article key={s.name} className="group overflow-hidden rounded-[12px] border border-line-soft bg-raise transition-all hover:border-line-strong hover:shadow-e2">
              <div className="flex items-start gap-3 p-3.5">
                <span className={cn("flex size-[32px] shrink-0 items-center justify-center rounded-[9px] border border-line-soft bg-well", `text-${s.tone}`)}><IconGlobe size={14} /></span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-1.5"><h3 className="font-display text-[13.5px] font-semibold text-ink">{s.name}</h3><Badge tone={s.tone} mono dot className="text-[9px]">{s.status}</Badge><Badge tone="neutral" mono className="text-[9px]">{s.transport}</Badge></div>
                  <code className="mt-1 block truncate font-mono text-[10.5px] text-ink-3">{s.command}</code>
                  <div className="mt-1.5 flex items-center gap-1.5">
                    {s.scopes.map((sc) => <span key={sc} className={cn("rounded-[4px] px-1.5 py-[1px] font-mono text-[9px]", sc === "code" ? "bg-iris-tint text-iris-soft" : "bg-cyan-tint text-cyan")}>{sc}</span>)}
                    <span className="font-mono text-[9.5px] text-ink-4">{s.tools} tools</span>
                  </div>
                </div>
                <Toggle checked={s.status === "connected"} onChange={(v) => { setServers((all) => all.map((x) => x.name === s.name ? { ...x, status: v ? "connected" : "degraded", tone: v ? "mint" : "amber" } : x)); toast(`${s.name} ${v ? "enabled" : "disabled"}`, v ? "mint" : "amber"); }} size="sm" />
              </div>
              <div className="flex items-center gap-1.5 border-t border-line-soft px-3 py-2.5">
                <Button variant="ghost" size="xs" icon={IconRefresh} onClick={() => toast(`Re-handshaked ${s.name}`, "mint")}>Reconnect</Button>
                <Button variant="ghost" size="xs" onClick={() => toast(`${s.tools} tools listed in the tool inspector`, "iris")}>Inspect tools</Button>
                <Button variant="ghost" size="xs" className="ml-auto text-rose" icon={IconTrash} onClick={() => setServers((all) => all.filter((x) => x.name !== s.name))}>Remove</Button>
              </div>
            </article>
          ))}
        </div>

        <div className="overflow-hidden rounded-[12px] border border-line-soft bg-raise">
          <div className="flex items-center justify-between border-b border-line-soft px-3.5 py-2.5"><h3 className="font-display text-[13px] font-semibold text-ink">Tool permissions</h3><span className="font-mono text-[10px] text-ink-4">evaluated per call · mode ∧ project ∧ connector scope</span></div>
          <div className="scroll-thin overflow-x-auto">
            <table className="w-full min-w-[560px] border-collapse">
              <thead><tr className="border-b border-line-soft">{["Tool call", "Read only", "Plan", "Agent", "Full access"].map((h) => <th key={h} className="px-3.5 py-2 text-left font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">{h}</th>)}</tr></thead>
              <tbody>
                {[["github.search_issues", 1, 1, 1, 1], ["linear.create_task", 0, 0, 2, 1], ["postgres.query", 1, 1, 2, 1], ["postgres.migrate", 0, 0, 0, 2], ["figma.export_frame", 1, 1, 1, 1], ["sentry.resolve_issue", 0, 0, 2, 1]].map(([t, a, b, c, d]) => (
                  <tr key={t as string} className="border-b border-line-soft/60 last:border-0 hover:bg-hover/40">
                    <td className="px-3.5 py-2 font-mono text-[11px] text-ink-2">{t as string}</td>
                    {[a, b, c, d].map((v, i) => <td key={i} className="px-3.5 py-2">
                      <span className={cn("inline-flex h-[18px] min-w-[46px] items-center justify-center rounded-[4px] px-1.5 font-mono text-[9.5px]", v === 1 ? "bg-mint-tint text-mint" : v === 2 ? "bg-amber-tint text-amber" : "bg-hover text-ink-4")}>{v === 1 ? "allow" : v === 2 ? "ask" : "deny"}</span>
                    </td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}


/* ====================================================================
   PROJECTS — defaults inherited by threads
   ==================================================================== */
export function ProjectsView({ projects, sessions, onUpdate, onNew, onOpenThread }: { projects: Project[]; sessions: Session[]; onUpdate: (p: Project) => void; onNew: (id: string) => void; onOpenThread: (id: string) => void }) {
  const { toast } = useApp();
  const [sel, setSel] = useState<string | null>(projects[0]?.id ?? null);
  const project = projects.find((p) => p.id === sel) ?? null;
  const threads = project ? sessions.filter((s) => s.projectId === project.id) : [];
  const attention = threads.filter((s) => s.status === "waiting" || s.status === "failed").length;
  const spend = threads.reduce((sum, s) => sum + Number(s.cost.replace(/[^0-9.]/g, "") || 0), 0);

  return (
    <div className="scroll-thin flex-1 overflow-y-auto">
      <PageHeader eyebrow="workspace" title="Projects" sub="A project carries the defaults its threads inherit — agent, model, permission mode, rules and budget. Change them once, every new thread follows."
        right={<Button variant="primary" icon={IconPlus} onClick={() => toast("Create projects from the sidebar's ⋯ menu", "iris")}>New project</Button>} />

      <div className="grid gap-4 p-5 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
        <div className="space-y-2">
          {projects.map((pr) => {
            const n = sessions.filter((s) => s.projectId === pr.id).length;
            const att = sessions.filter((s) => s.projectId === pr.id && (s.status === "waiting" || s.status === "failed")).length;
            const on = sel === pr.id;
            return (
              <button key={pr.id} onClick={() => setSel(pr.id)} className={cn("group flex w-full cursor-pointer items-start gap-2.5 rounded-[11px] border p-3 text-left transition-all", on ? "border-iris/50 bg-iris-tint/50 shadow-glow-iris" : "border-line-soft bg-raise hover:border-line-strong")}>
                <span className="mt-0.5 flex size-[26px] shrink-0 items-center justify-center rounded-[7px] font-mono text-[10px] font-bold" style={{ color: pr.color, background: `color-mix(in srgb, ${pr.color} 14%, transparent)`, boxShadow: `inset 0 0 0 1px ${pr.color}40` }}>{pr.name.slice(0, 2).toUpperCase()}</span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-1.5"><span className="truncate text-[12.5px] font-medium text-ink">{pr.name}</span>{att > 0 && <span className="flex h-[15px] min-w-[15px] items-center justify-center rounded-full bg-amber px-1 font-mono text-[8.5px] font-bold text-on-iris">{att}</span>}</span>
                  <span className="mt-0.5 block truncate font-mono text-[10px] text-ink-4">{pr.path ?? pr.kind}</span>
                  <span className="mt-1 flex items-center gap-2 font-mono text-[9.5px] text-ink-4"><span>{n} threads</span><span>·</span><span>{pr.defaults?.agentId ? agentById(pr.defaults.agentId).name : "no default"}</span></span>
                </span>
                <IconChevronDown size={12} className={cn("mt-1 shrink-0 text-ink-4 transition-transform", on ? "rotate-90 text-iris-soft" : "group-hover:translate-x-0.5")} />
              </button>
            );
          })}
        </div>

        {project && (
          <div className="space-y-3">
            <div className="relative overflow-hidden rounded-[13px] border border-line-soft bg-raise p-4">
              <div className="ambient pointer-events-none absolute inset-0 opacity-70" />
              <div className="relative flex flex-wrap items-start gap-3">
                <span className="flex size-[40px] items-center justify-center rounded-[11px] font-mono text-[13px] font-bold" style={{ color: project.color, background: `color-mix(in srgb, ${project.color} 15%, transparent)`, boxShadow: `inset 0 0 0 1px ${project.color}44` }}>{project.name.slice(0, 2).toUpperCase()}</span>
                <div className="min-w-0 flex-1">
                  <h2 className="font-display text-[18px] leading-tight font-semibold tracking-[-.02em] text-ink">{project.name}</h2>
                  <p className="mt-0.5 font-mono text-[10.5px] text-ink-4">{project.path ?? "workspace"} · {project.kind}</p>
                </div>
                <div className="flex gap-2"><Button variant="secondary" icon={IconFolder} onClick={() => onNew(project.id)}>New thread</Button><Button variant="ghost" icon={IconSettings} onClick={() => toast("Edit defaults below", "iris")}>Defaults</Button></div>
              </div>
              <div className="relative mt-4 grid grid-cols-2 gap-2.5 sm:grid-cols-4">
                {[["Threads", threads.length, "neutral"], ["Needs you", attention, attention ? "amber" : "neutral"], ["Spend", `$${spend.toFixed(2)}`, "iris"], ["Budget", project.defaults?.budget ?? "—", "neutral"]].map(([l, v, t]) => (
                  <div key={l as string} className="rounded-[9px] border border-line-soft bg-well px-2.5 py-2">
                    <div className="font-mono text-[9px] tracking-[.12em] text-ink-4 uppercase">{l as string}</div>
                    <div className={cn("mt-0.5 font-display text-[17px] leading-none font-semibold tracking-[-.02em]", t === "amber" ? "text-amber" : t === "iris" ? "text-iris-soft" : "text-ink")}>{v as string}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-[12px] border border-line-soft bg-raise p-3.5">
                <h3 className="flex items-center gap-1.5 font-display text-[13px] font-semibold text-ink"><IconLayers size={12} className="text-iris-soft" />Inherited defaults</h3>
                <div className="mt-3 space-y-2.5">
                  <div><label className="mb-1 block font-mono text-[9px] tracking-[.12em] text-ink-4 uppercase">agent</label>
                    <Select value={project.defaults?.agentId ?? "claude-code"} onChange={(v) => { const a = agentById(v); onUpdate({ ...project, defaults: { ...project.defaults, agentId: v, modelId: a.models[0].id } }); toast(`${project.name} → ${a.name}`, "mint"); }} options={AGENTS.filter((a) => a.status === "connected").map((a) => ({ value: a.id, label: a.name, hint: a.vendor }))} /></div>
                  <div><label className="mb-1 block font-mono text-[9px] tracking-[.12em] text-ink-4 uppercase">model</label>
                    <Select value={project.defaults?.modelId ?? ""} onChange={(v) => onUpdate({ ...project, defaults: { ...project.defaults, modelId: v } })} options={(agentById(project.defaults?.agentId ?? "claude-code").models).map((m) => ({ value: m.id, label: m.label, hint: m.ctx }))} /></div>
                  <div><label className="mb-1 block font-mono text-[9px] tracking-[.12em] text-ink-4 uppercase">permission mode</label>
                    <Segmented value={(project.defaults?.mode ?? "agent") as ModeId} onChange={(v) => onUpdate({ ...project, defaults: { ...project.defaults, mode: v } })} items={[{ value: "plan", label: "Plan", tone: "iris" }, { value: "agent", label: "Agent", tone: "cyan" }, { value: "readonly", label: "Read", tone: "amber" }, { value: "full", label: "Full", tone: "rose" }]} /></div>
                  <div><label className="mb-1 block font-mono text-[9px] tracking-[.12em] text-ink-4 uppercase">budget</label>
                    <Input value={project.defaults?.budget ?? ""} onChange={(e) => onUpdate({ ...project, defaults: { ...project.defaults, budget: e.target.value } })} placeholder="$25 / session" className="font-mono text-[11.5px]" /></div>
                </div>
              </div>

              <div className="rounded-[12px] border border-line-soft bg-raise p-3.5">
                <div className="flex items-center gap-1.5"><h3 className="font-display text-[13px] font-semibold text-ink"><IconShield size={12} className="text-iris-soft" />Project rules</h3><Button variant="ghost" size="xs" className="ml-auto" icon={IconPlus} onClick={() => onUpdate({ ...project, defaults: { ...project.defaults, rules: [...(project.defaults?.rules ?? []), ""] } })}>Add</Button></div>
                <div className="mt-2.5 space-y-1.5">
                  {(project.defaults?.rules ?? []).length === 0 && <p className="rounded-[8px] border border-dashed border-line px-2.5 py-3 text-center text-[11px] text-ink-4">No rules — threads fall back to your global rules</p>}
                  {(project.defaults?.rules ?? []).map((r, i) => (
                    <div key={i} className="group flex items-center gap-2 rounded-[7px] border border-line-soft bg-well px-2 py-1.5">
                      <span className="font-mono text-[9px] text-ink-4">{String(i + 1).padStart(2, "0")}</span>
                      <input value={r} onChange={(e) => onUpdate({ ...project, defaults: { ...project.defaults, rules: (project.defaults?.rules ?? []).map((x, j) => (j === i ? e.target.value : x)) } })} className="min-w-0 flex-1 bg-transparent text-[11.5px] text-ink-2 focus:outline-none" />
                      <button onClick={() => onUpdate({ ...project, defaults: { ...project.defaults, rules: (project.defaults?.rules ?? []).filter((_, j) => j !== i) } })} className="cursor-pointer text-ink-4 opacity-0 transition-opacity group-hover:opacity-100 hover:text-rose"><IconX size={11} /></button>
                    </div>
                  ))}
                </div>
                <div className="mt-3 border-t border-line-soft pt-2.5">
                  <div className="mb-1.5 font-mono text-[9px] tracking-[.12em] text-ink-4 uppercase">global rules also apply</div>
                  {RULES.slice(0, 3).map((r) => <div key={r.id} className="flex items-start gap-2 py-1 text-[11px] text-ink-3"><i className="mt-[6px] size-[4px] shrink-0 rounded-full bg-line-strong" /><span className="min-w-0 flex-1 truncate">{r.text}</span><span className="font-mono text-[9px] text-ink-4">{r.scope}</span></div>)}
                </div>
              </div>
            </div>

            <div className="overflow-hidden rounded-[12px] border border-line-soft bg-raise">
              <div className="flex items-center justify-between border-b border-line-soft px-3.5 py-2.5"><h3 className="font-display text-[13px] font-semibold text-ink">Threads</h3><span className="font-mono text-[10px] text-ink-4">{threads.length}</span></div>
              {threads.length === 0 && <p className="px-3.5 py-6 text-center text-[11.5px] text-ink-4">No threads yet — start one and it inherits these defaults.</p>}
              {threads.map((s) => {
                const a = agentById(s.agentId);
                const inherits = a.id === project.defaults?.agentId;
                return (
                  <button key={s.id} onClick={() => onOpenThread(s.id)} className="flex w-full cursor-pointer items-center gap-2.5 border-b border-line-soft/60 px-3.5 py-2.5 text-left last:border-0 transition-colors hover:bg-hover/50">
                    <span className={cn("size-[6px] shrink-0 rounded-full", s.status === "running" ? "animate-breathe bg-cyan" : s.status === "waiting" ? "bg-amber" : s.status === "failed" ? "bg-rose" : s.status === "done" ? "bg-mint" : "bg-line-strong")} />
                    <span className="min-w-0 flex-1 truncate text-[12px] text-ink-2">{s.title}</span>
                    {inherits ? <Tip label="Using project defaults"><Badge tone="iris" mono className="text-[9px]">inherited</Badge></Tip> : <Badge tone="neutral" mono className="text-[9px]">override</Badge>}
                    <span className="hidden font-mono text-[10px] text-ink-4 sm:inline">{s.cost}</span>
                    <IconPulse size={11} className="shrink-0 text-ink-4" />
                  </button>
                );
              })}
            </div>

            <div className="rounded-[12px] border border-line-soft bg-raise p-3.5">
              <div className="flex items-center gap-2"><IconZap size={12} className="text-iris-soft" /><h3 className="font-display text-[13px] font-semibold text-ink">Spend by agent</h3><span className="ml-auto font-mono text-[10px] text-ink-4">this project</span></div>
              <div className="mt-2.5 space-y-2">
                {AGENTS.filter((a) => threads.some((t) => t.agentId === a.id)).slice(0, 4).map((a) => {
                  const n = threads.filter((t) => t.agentId === a.id).length;
                  return (
                    <div key={a.id} className="flex items-center gap-2.5">
                      <span className="w-[110px] shrink-0 truncate text-[11.5px] text-ink-2">{a.name}</span>
                      <Bar value={(n / Math.max(threads.length, 1)) * 100} />
                      <span className="w-8 shrink-0 text-right font-mono text-[10px] text-ink-4">{n}</span>
                    </div>
                  );
                })}
                {threads.length === 0 && <p className="text-[11px] text-ink-4">No spend recorded yet.</p>}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
