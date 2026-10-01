"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cn } from "./utils/cn";
import { MODES, SESSIONS, agentById, type ModeId, type Session, type Step } from "./data/catalog";
import { AGENT_SESSIONS, buildAgentResponse } from "./data/extra";
import { DEFAULT_PROJECT_FOR_SESSION, PROJECTS, type Project } from "./data/providers";
import { AppProvider, useApp, useResizable } from "./lib/app";
import { fetchLiveReply, historyFromSteps, sleep } from "./lib/live";

import { AgentMark, IconBolt, IconCheck, IconHelp, IconPanelLeft, IconSpark, IconTerminal, LogoMark } from "./components/Icons";
import { Badge, Button, Kbd, Modal, Ring, Segmented, Toasts } from "./components/ui";
import { NAV_AGENT, NAV_CODE, Sidebar, type ViewId } from "./components/Sidebar";
import { Titlebar } from "./components/Titlebar";
import { Workbench, modeDefs } from "./components/Workbench";

import { CommandPalette } from "./components/CommandPalette";
import { FleetView, ReviewView } from "./components/views";
import { BrowserView, GitView, TasksView } from "./components/views2";
import { ConnectorsView, ProjectsView, SkillsView } from "./components/views3";
import { SettingsView } from "./components/Settings";
import { TerminalPanel } from "./components/Terminal";
import { TodoPopup } from "./components/TodoPopup";
import { ShortcutHelp } from "./components/ShortcutHelp";
import { VoiceAgent } from "./components/VoiceAgent";
import { PRESETS, type LayoutState } from "./components/LayoutControl";
import { ContextPane, type ContextSectionId } from "./components/ContextPane";
import { Sheet } from "./components/Sheet";
import { ErrorBoundary } from "./components/ErrorBoundary";

let uid = 0; const nid = () => `x${++uid}`;
/* run token — bumped on stop/switch so the live streaming loop halts */
let runToken = 0;

function buildCodeResponse(prompt: string, agentId: string, mode: ModeId): Step[] {
  const a = agentById(agentId); const topic = prompt.length > 58 ? prompt.slice(0, 58).trim() + "…" : prompt.trim();
  const plan: Step[] = [
    { id: nid(), type: "thinking", ms: 3100, text: `Touches more than one file and the caller contract isn't obvious. Establish where the behaviour lives, what depends on it, and which invariants break. Mode is ${mode}, so stop at a plan.` },
    { id: nid(), type: "text", text: `Working on **${topic}** with \`${a.name}\`. Here's what I found before proposing changes.` },
    { id: nid(), type: "tool", tool: "grep", title: "Search", target: 'rg -n "export (function|class)" packages --type ts', status: "done", ms: 720, lines: ["packages/bridge/src/server.ts:11  export async function startBridge", "packages/bridge/src/channel.ts:3   export class SessionChannel", "packages/review/src/reviewer.ts:31  export class DiffReviewer"] },
    { id: nid(), type: "tool", tool: "plan", title: "Plan", status: "done", ms: 900, lines: ["1. Throttled flush + acknowledged cursor on SessionChannel", "2. Capability negotiation at attach time", "3. Route editor patches through DiffReviewer", "4. Two-editor integration test · zero lost writes"] },
    { id: nid(), type: "text", text: "Nothing written to disk. Press **⇧Tab** to switch to Agent and execute, or edit the plan in the side panel first." },
  ];
  const agent: Step[] = [
    { id: nid(), type: "thinking", ms: 2200, text: "Smallest safe change first: throttled flush, then cursor. Extend to negotiation once green." },
    { id: nid(), type: "text", text: `On it — taking **${topic}**. Flush + resume cursor first since everything depends on it.` },
    { id: nid(), type: "checkpoint", label: "before channel rewrite", files: 3, at: "now", hash: Math.random().toString(16).slice(2, 9) },
    { id: nid(), type: "tool", tool: "edit", title: "Edit", target: "packages/bridge/src/channel.ts", status: "done", ms: 410, diff: { path: "packages/bridge/src/channel.ts", adds: 9, dels: 1, lang: "ts", lines: [
      { t: "hunk", text: "@@ -2,7 +2,15 @@ export class SessionChannel", o: 2, n: 2 }, { t: "ctx", text: "  head = 0", o: 4, n: 4 }, { t: "del", text: "  push(frame: Frame) { this.buffer.push(frame) }", o: 5 },
      { t: "add", text: "  private acked = 0", n: 5 }, { t: "add", text: "  push(frame: Frame) { this.buffer.push(frame); this.schedule() }", n: 6 }, { t: "add", text: "  private schedule() {", n: 8 }, { t: "add", text: "    if (this.timer) return", n: 9 },
      { t: "add", text: "    this.timer = setTimeout(() => this.flush(), this.opts.flushEveryMs ?? 12)", n: 10 }, { t: "add", text: "  }", n: 11 }, { t: "ctx", text: "}", o: 6, n: 13 } ] } },
    { id: nid(), type: "tool", tool: "bash", title: "Shell", target: "pnpm --filter @aro/bridge test --run", status: "done", ms: 1840, output: ["RUN v3.2.4 packages/bridge", " ✓ session-channel.test.ts (14 tests) 284ms", " ✓ resume-cursor.test.ts (6 tests) 92ms", "Test Files 2 passed · Tests 20 passed"] },
    { id: nid(), type: "text", text: "Flush and resume cursor are in and green. Review the diff in **Changes**, or ask me to continue with capability negotiation." },
  ];
  return mode === "plan" || mode === "readonly" ? plan : agent;
}
const LABELS = { code: ["mapping call sites…", "negotiating editor capabilities…", "writing SessionChannel.flush()…", "running bridge test suite…"], agent: ["browsing sources…", "reading your docs…", "drafting…", "checking calendars…"] };
const SETTINGS_ALIAS: Partial<Record<ViewId, string>> = { agents: "agents", bridges: "editors", design: "design" };
const typing = () => { const el = document.activeElement; return !!el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA"); };
const SHEET_TITLES: Partial<Record<ViewId, [string, string]>> = {
  settings: ["Settings", "preferences"], tasks: ["Tasks", "work graph"], skills: ["Skills", "capability"], connectors: ["Connectors", "integrations"],
  projects: ["Projects", "workspace"], review: ["Changes", "review"], git: ["Git", "source control"], browser: ["Browser", "preview"], runs: ["Fleet", "parallel runs"],
};

function loadSessions(key: string, seeds: Session[]): Session[] {
  const fallback = seeds.map((s) => ({ ...s, projectId: s.projectId ?? DEFAULT_PROJECT_FOR_SESSION[s.id] ?? "platform" }));
  try {
    const saved = JSON.parse(localStorage.getItem(key) || "null") as Partial<Session>[] | null;
    if (!saved?.length) return fallback;
    const seedMap = new Map(fallback.map((s) => [s.id, s]));
    return saved.filter((s): s is Partial<Session> & Pick<Session, "id"> => typeof s.id === "string").map((s) => {
      const seed = seedMap.get(s.id);
      return { ...(seed ?? {} as Session), ...s, steps: seed?.steps ?? [], status: s.status === "running" ? "idle" : s.status ?? seed?.status ?? "idle", projectId: s.projectId ?? seed?.projectId ?? "platform" } as Session;
    });
  } catch { return fallback; }
}

function Shell() {
  const { product, setProduct, toasts, setTodos, toast, termOpen, setTermOpen, termPush, dockOpen, setDockOpen, setSettingsSection, todoOpen, setTodoOpen } = useApp();
  const [view, setViewRaw] = useState<ViewId>("workbench");
  /* Panels are toggles: clicking the open one again returns to the chat. */
  const setView = useCallback((v: ViewId) => {
    const alias = SETTINGS_ALIAS[v];
    if (alias) { setSettingsSection(alias); setViewRaw("settings"); return; }
    setViewRaw((cur) => (cur === v && v !== "workbench" ? "workbench" : v));
  }, [setSettingsSection]);
  const closeSheet = useCallback(() => setViewRaw("workbench"), []);
  const [codeSessions, setCodeSessions] = useState<Session[]>(() => loadSessions("aro.sessions.code", SESSIONS));
  const [agentSessions, setAgentSessions] = useState<Session[]>(() => loadSessions("aro.sessions.agent", AGENT_SESSIONS));
  const [projects, setProjects] = useState<Project[]>(() => {
    try { return JSON.parse(localStorage.getItem("aro.projects") || "null") ?? PROJECTS; }
    catch { return PROJECTS; }
  });
  const sessions = product === "code" ? codeSessions : agentSessions;
  const setSessions = product === "code" ? setCodeSessions : setAgentSessions;
  const [activeIds, setActiveIds] = useState({ code: "s1", agent: "a1" });
  const activeId = activeIds[product];
  const [stepsMap, setStepsMap] = useState<Record<string, Step[]>>(() => Object.fromEntries([...SESSIONS, ...AGENT_SESSIONS].map((s) => [s.id, s.steps])));
  const steps = stepsMap[activeId] ?? [];
  const stepsMapRef = useRef(stepsMap); stepsMapRef.current = stepsMap;
  const [streaming, setStreaming] = useState(true);
  const [label, setLabel] = useState(LABELS.code[0]);
  const [mode, setMode] = useState<ModeId>("agent");
  const [agentId, setAgentId] = useState("aro");
  const [modelId, setModelId] = useState("aro-4-405b");
  const [cmdk, setCmdk] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [layoutOpen, setLayoutOpen] = useState(false);
  const [newOpen, setNewOpen] = useState(false);
  const [newAgent, setNewAgent] = useState("aro");
  const [newMode, setNewMode] = useState<ModeId>("plan");
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [listOpen, setListOpen] = useState(() => localStorage.getItem("aro.sidebar.collapsed") !== "true");
  useEffect(() => localStorage.setItem("aro.sidebar.collapsed", String(!listOpen)), [listOpen]);
  const [statusOpen, setStatusOpen] = useState(true);
  const [paneSection, setPaneSection] = useState<ContextSectionId>("plan");
  /* narrow screens: sidebar becomes a drawer, context pane is tucked away */
  const [isNarrow, setIsNarrow] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 859px)");
    const onChange = () => {
      setIsNarrow(mq.matches);
      if (mq.matches) { setListOpen(false); setInspectorOpen(false); setTermOpen(false); }
    };
    onChange();
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [setTermOpen]);
  const timers = useRef<number[]>([]);
  const list = useResizable(272, 220, 420, "right");
  const pane = useResizable(368, 300, 620, "left");
  const active = useMemo(() => sessions.find((s) => s.id === activeId) ?? sessions[0], [sessions, activeId]);
  const activeIdRef = useRef(activeId); activeIdRef.current = activeId;

  useEffect(() => {
    const snapshot = sessions.map(({ steps: _steps, ...meta }) => meta);
    localStorage.setItem(product === "code" ? "aro.sessions.code" : "aro.sessions.agent", JSON.stringify(snapshot));
  }, [sessions, product]);
  useEffect(() => localStorage.setItem("aro.projects", JSON.stringify(projects)), [projects]);

  const isWb = view === "workbench";
  const panelLabel = "Context";
  const layout: LayoutState = { list: listOpen, terminal: termOpen, panel: inspectorOpen, status: statusOpen };
  const setLayout = (s: LayoutState) => { setListOpen(s.list); setTermOpen(s.terminal); setInspectorOpen(s.panel); setDockOpen(s.panel); setStatusOpen(s.status); };

  /* finish the seeded running step (boot demo) */
  useEffect(() => {
    const t = window.setTimeout(() => {
      setStepsMap((m) => { const p = m.s1; const i = p.findIndex((s) => s.type === "tool" && s.status === "running"); if (i === -1) return m; const c = [...p]; c[i] = { ...(c[i] as Extract<Step, { type: "tool" }>), status: "done", ms: 612 }; return { ...m, s1: [...c, { id: nid(), type: "text", text: "Reconnect test is green — 12 scenarios, zero lost writes.\n\n**Next:** the reviewer still assumes agent-originated patches, so an edit made in Cursor shows up as agent work. That's the last blocker before the editor adapters." } as Step] }; });
      setStreaming(false);
      termPush([{ t: " ✓ reconnect.test.ts (12 tests) 612ms", k: "ok" }, { t: "Test Files 3 passed · Tests 35 passed", k: "ok" }]);
      setTodos((t) => t.map((x) => x.id === "t3" ? { ...x, state: "done" } : x.id === "t4" ? { ...x, state: "doing" } : x));
    }, 3200);
    return () => clearTimeout(t);
  }, [setTodos, termPush]);
  useEffect(() => { if (!streaming) return; const i = window.setInterval(() => setLabel(LABELS[product][Math.floor(Math.random() * 4)]), 2200); return () => clearInterval(i); }, [streaming, product]);
  /* tab title mirrors agent state so a backgrounded tab still tells you when work finishes */
  useEffect(() => { document.title = streaming ? "● Working · Aro" : sessions.some((s) => s.status === "waiting") ? "◐ Needs you · Aro" : "Aro"; }, [streaming, sessions]);

  const clear = () => { timers.current.forEach(clearTimeout); timers.current = []; };
  const run = useCallback((script: Step[], sid: string, done?: () => Promise<void>) => {
    clear(); setStreaming(true);
    const finish = () => { if (done) { void done(); } else setStreaming(false); };
    if (script.length === 0) { window.setTimeout(finish, 320); return; }
    script.forEach((s, i) => timers.current.push(window.setTimeout(() => {
      setStepsMap((m) => ({ ...m, [sid]: [...(m[sid] ?? []), s] }));
      if (s.type === "tool" && (s.tool === "bash" || s.tool === "test")) termPush([{ t: `$ ${s.target}`, k: "cmd" }, ...(s.output ?? []).map((t) => ({ t, k: t.trim().startsWith("✓") ? ("ok" as const) : t.trim().startsWith("✗") ? ("err" as const) : undefined }))]);
      if (s.type === "tool" && s.tool === "edit") { termPush([{ t: `[agent] edited ${s.target}`, k: "dim" }]); setTodos((t) => [{ id: `ag${Date.now()}`, text: `Verify ${s.target?.split("/").pop()} change`, state: "doing", by: "agent" }, ...t]); }
      if (i === script.length - 1) finish();
    }, 320 + i * 1100)));
  }, [setTodos, termPush]);

  const send = (text: string) => {
    const sid = activeIdRef.current;
    setStepsMap((m) => ({ ...m, [sid]: [...(m[sid] ?? []), { id: nid(), type: "user", text, at: "now" }] }));
    setSessions((ss) => ss.map((s) => s.id === sid && s.title === "New session" ? { ...s, title: text.slice(0, 60), status: "running" } : s));
    /* scripted harness steps; the closing text becomes the live-AI fallback */
    const full = product === "code" ? buildCodeResponse(text, agentId, mode) : buildAgentResponse(text);
    let fallback = "Done. Everything checks out — say the word if you want the diff opened for review.";
    const script = [...full];
    const last = script[script.length - 1];
    if (last && last.type === "text") { fallback = last.text; script.pop(); }
    const token = ++runToken;
    run(script, sid, async () => {
      /* live phase: real model reply, streamed word-by-word into the thread */
      const history = historyFromSteps(stepsMapRef.current[sid] ?? []);
      const reply = await fetchLiveReply(history, fallback);
      if (token !== runToken) return;
      const id = nid();
      setStepsMap((m) => ({ ...m, [sid]: [...(m[sid] ?? []), { id, type: "text", text: "", agentId } as Step] }));
      for (const w of reply.split(/(\s+)/)) {
        if (token !== runToken) return;
        await sleep(14 + Math.random() * 22);
        setStepsMap((m) => {
          const arr = [...(m[sid] ?? [])];
          const i = arr.findIndex((s) => s.id === id);
          const cur = i >= 0 ? arr[i] : undefined;
          if (!cur || cur.type !== "text") return m;
          arr[i] = { ...cur, text: cur.text + w };
          return { ...m, [sid]: arr };
        });
      }
      if (token !== runToken) return;
      setSessions((ss) => ss.map((s) => s.id === sid && s.status === "running" ? { ...s, status: "idle" } : s));
      setStreaming(false);
    });
    if (view === "settings") setViewRaw("workbench"); else if (view !== "workbench") setDockOpen(true);
  };
  const stop = () => { clear(); runToken++; setStreaming(false); const sid = activeIdRef.current; setStepsMap((m) => ({ ...m, [sid]: [...(m[sid] ?? []), { id: nid(), type: "notice", tone: "warn", text: "Interrupted — partial edits kept, checkpoint retained." }] })); };
  const select = (id: string) => { clear(); runToken++; setStreaming(false); const s = [...codeSessions, ...agentSessions].find((x) => x.id === id); if (!s) return; setActiveIds((a) => ({ ...a, [product]: id })); setAgentId(s.agentId); setModelId(agentById(s.agentId).models[0].id); };
  const selectAndGo = (id: string) => { select(id); setViewRaw("workbench"); };
  const create = (agent = newAgent, m = newMode, projectId?: string) => {
    const id = `n${Date.now()}`.slice(-7);
    const pid = projectId ?? sessions.find((x) => x.id === activeId)?.projectId ?? "platform";
    /* project defaults are inherited by every new thread */
    const inherited = projects.find((x) => x.id === pid)?.defaults;
    const effAgent = projectId && inherited?.agentId ? inherited.agentId : agent;
    const effMode = (projectId && inherited?.mode ? inherited.mode : m) as ModeId;
    const effModel = projectId && inherited?.modelId ? inherited.modelId : agentById(effAgent).models[0].id;
    const s: Session = { id, title: "New session", agentId: effAgent, branch: "feat/session", status: "idle", updated: "now", group: "Today", tokens: 0, cost: "$0.00", projectId: pid, steps: [] };
    setSessions((p) => [s, ...p]); setStepsMap((mm) => ({ ...mm, [id]: [] })); setActiveIds((a) => ({ ...a, [product]: id }));
    setAgentId(effAgent); setModelId(effModel); setMode(effMode); setNewOpen(false);
    if (projectId && inherited?.agentId) toast(`New thread inherits ${projects.find((x) => x.id === projectId)?.name} defaults`, "iris");
    if (view === "settings") setViewRaw("workbench");
  };
  const updateProject = (next: Project) => setProjects((ps) => ps.map((x) => (x.id === next.id ? next : x)));
  const createProject = (name: string) => {
    const id = `project-${Date.now()}`;
    const palette = ["#8f82ff", "#36c5b6", "#ec9acb", "#eab45a", "#6da8ff"];
    const project: Project = { id, name, color: palette[projects.length % palette.length], kind: "workspace" };
    setProjects((p) => [...p, project]);
    toast(`Project “${name}” created`, "mint");
    return id;
  };
  const archiveSession = (id: string, archived: boolean) => {
    setSessions((ss) => ss.map((s) => s.id === id ? { ...s, archived } : s));
    if (archived && activeId === id) {
      const next = sessions.find((s) => s.id !== id && !s.archived);
      if (next) setActiveIds((a) => ({ ...a, [product]: next.id }));
      else create(agentId, mode);
    }
    toast(archived ? "Session archived. Find it from the archive filter." : "Session restored to active sessions", "mint");
  };
  const deleteSession = (id: string) => {
    setSessions((ss) => ss.filter((s) => s.id !== id));
    setStepsMap((all) => { const next = { ...all }; delete next[id]; return next; });
    if (activeId === id) {
      const next = sessions.find((s) => s.id !== id && !s.archived);
      if (next) setActiveIds((a) => ({ ...a, [product]: next.id }));
      else create(agentId, mode);
    }
    toast("Session deleted", "amber");
  };
  const renameSession = (id: string, title: string) => {
    setSessions((ss) => ss.map((s) => s.id === id ? { ...s, title } : s));
    toast("Session renamed", "mint");
  };
  const moveSession = (id: string, projectId: string) => {
    setSessions((ss) => ss.map((s) => s.id === id ? { ...s, projectId } : s));
    toast(`Moved to ${projects.find((p) => p.id === projectId)?.name ?? "project"}`, "mint");
  };
  const duplicateSession = (id: string) => {
    const source = sessions.find((s) => s.id === id);
    if (!source) return;
    const copyId = `copy-${Date.now()}`;
    const copy: Session = { ...source, id: copyId, title: `${source.title} (copy)`, status: "idle", updated: "now", archived: false, favorite: false, steps: stepsMap[id] ?? source.steps };
    setSessions((ss) => [copy, ...ss]);
    setStepsMap((m) => ({ ...m, [copyId]: copy.steps }));
    setActiveIds((a) => ({ ...a, [product]: copyId }));
    toast("Session duplicated", "mint");
  };
  const togglePin = (id: string) => setSessions((ss) => ss.map((s) => s.id === id ? { ...s, pinned: false, favorite: !(s.favorite ?? s.pinned ?? false) } : s));
  const openNew = (agent?: string, projectId?: string) => { if (agent) { setNewAgent(agent); setNewOpen(true); return; } create(agentId, mode, projectId); setViewRaw("workbench"); };
  const cycleMode = () => { const ms = modeDefs(product); setMode(ms[(ms.findIndex((m) => m.id === mode) + 1) % ms.length].id); };
  const lastEsc = useRef(0);

  /* ---------------- shortcuts (registry: lib/shortcuts.ts) ---------------- */
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const meta = e.metaKey || e.ctrlKey;
      const k = e.key.toLowerCase();
      const views = (product === "code" ? NAV_CODE : NAV_AGENT).map((n) => n.id);

      if (e.key === "Tab" && e.shiftKey) { e.preventDefault(); cycleMode(); toast(`Mode · ${modeDefs(product).find((m) => m.id === mode)!.label} → next`, "iris"); return; }

      if (!meta) {
        if (e.key === "Escape") {
          if (helpOpen) { setHelpOpen(false); return; }
          if (cmdk) { setCmdk(false); return; }
          if (document.querySelector('[aria-modal="true"]')) return; // let the modal handle it
          if (view !== "workbench") {
            if (typing()) { (document.activeElement as HTMLElement).blur(); return; }
            e.preventDefault(); closeSheet(); return;
          }
          if (typing()) return;
          const now = Date.now();
          if (streaming) { e.preventDefault(); stop(); return; }
          if (now - lastEsc.current < 600) { e.preventDefault(); toast("Rewound to checkpoint a91f3c2", "amber"); }
          lastEsc.current = now;
        }
        return;
      }

      const want = (mods: string[]) => mods.every((m) => (m === "⌘" ? e.metaKey || e.ctrlKey : m === "⇧" ? e.shiftKey : m === "⌥" ? e.altKey : true));

      if (k === "k" && want(["⌘"]) && !e.shiftKey) { e.preventDefault(); setCmdk((c) => !c); }
      else if (k === "/" && want(["⌘"])) { e.preventDefault(); setHelpOpen((o) => !o); }
      else if (k === "," && want(["⌘"])) { e.preventDefault(); setViewRaw((c) => (c === "settings" ? "workbench" : "settings")); }
      else if (k === "e" && want(["⌘"])) { e.preventDefault(); setProduct(product === "code" ? "agent" : "code"); }
      else if (k === "n" && want(["⌘"]) && !e.shiftKey) { e.preventDefault(); openNew(); }
      else if (k === "j" && want(["⌘"])) { e.preventDefault(); setTermOpen(!termOpen); }
      else if (k === "l" && want(["⌘"])) { e.preventDefault(); if (isWb) setInspectorOpen((o) => !o); else setDockOpen(!dockOpen); }
      else if (e.key === "\\" && want(["⌘"])) { e.preventDefault(); setInspectorOpen((o) => !o); }
      else if (k === "b" && want(["⌘", "⇧"])) { e.preventDefault(); setListOpen((o) => !o); }
      else if (k === "l" && want(["⌘", "⇧"])) { e.preventDefault(); setLayoutOpen((o) => !o); }
      else if (k === "f" && want(["⌘", "⇧"])) { e.preventDefault(); setLayout(layout.panel || layout.terminal || layout.list ? PRESETS[1].state : PRESETS[0].state); }
      else if (k === "k" && want(["⌘", "⇧"])) { e.preventDefault(); toast("Context compacted · 62% → 31%", "mint"); }
      else if (k === "d" && want(["⌘"])) { e.preventDefault(); toast("Forked into worktree ../cnd-fork", "iris"); }
      else if (e.key === "[" && want(["⌘"])) {
        e.preventDefault();
        const order: ContextSectionId[] = ["plan", "changes", "context", "runs", "git", "browser", "editors", "terminal"];
        const next = order[(order.indexOf(paneSection) + (e.shiftKey ? -1 + order.length : 1)) % order.length];
        setPaneSection(next); setInspectorOpen(true);
      }
      else if (k === "v" && want(["⌘", "⇧"])) { e.preventDefault(); if (view !== "workbench") setViewRaw("workbench"); window.dispatchEvent(new CustomEvent("aro:voice:dictate")); }
      else if (e.key === "Enter" && want(["⌘"])) { e.preventDefault(); toast("Approval granted", "mint"); }
      else if (/^[1-6]$/.test(e.key) && views[+e.key - 1]) { e.preventDefault(); setView(views[+e.key - 1]); }
    };
    window.addEventListener("keydown", h); return () => window.removeEventListener("keydown", h);
  });

  useEffect(() => { const h = () => setHelpOpen(true); window.addEventListener("aro:help", h); return () => window.removeEventListener("aro:help", h); }, []);
  useEffect(() => {
    const h = (event: Event) => {
      const detail = (event as CustomEvent<ContextSectionId>).detail;
      if (detail) setPaneSection(detail);
      setInspectorOpen(true);
    };
    window.addEventListener("aro:pane", h);
    return () => window.removeEventListener("aro:pane", h);
  }, []);
  useEffect(() => { const h = () => setView("settings"); window.addEventListener("aro:settings", h); return () => window.removeEventListener("aro:settings", h); }, [setView]);
  useEffect(() => {
    const terminal = () => setTermOpen(!termOpen);
    const todo = () => setTodoOpen(!todoOpen);
    const providersPage = () => { setSettingsSection("providers"); setViewRaw("settings"); };
    const newSession = () => openNew();
    const switchProduct = (event: Event) => { const next = (event as CustomEvent<"code" | "agent">).detail; if (next === "code" || next === "agent") setProduct(next); };
    window.addEventListener("aro:terminal", terminal);
    window.addEventListener("aro:todo", todo);
    window.addEventListener("aro:providers", providersPage);
    window.addEventListener("aro:new-session", newSession);
    window.addEventListener("aro:product", switchProduct);
    return () => {
      window.removeEventListener("aro:terminal", terminal);
      window.removeEventListener("aro:todo", todo);
      window.removeEventListener("aro:providers", providersPage);
      window.removeEventListener("aro:new-session", newSession);
      window.removeEventListener("aro:product", switchProduct);
    };
  });

  const wbProps = { session: active, sessions, onSelect: select, onNew: () => openNew(), steps, streaming, streamLabel: label, mode, setMode, agentId, setAgentId: (a: string) => { setAgentId(a); setModelId(agentById(a).models[0].id); toast(`Handed off to ${agentById(a).name} — transcript travels with it`, "iris"); }, modelId, setModelId, onSend: send, onStop: stop };
  const latestReply = [...steps].reverse().find((s): s is Extract<Step, { type: "text" }> => s.type === "text")?.text.replace(/[`*#]/g, "") ?? "";

  const modeMeta = modeDefs(product).find((m) => m.id === mode)!;

  return (
    <div className="flex h-full flex-col bg-void text-ink">
      <Titlebar onOpenCmdk={() => setCmdk(true)} onNew={() => openNew()} setView={setView} layout={layout} setLayout={setLayout} panelLabel={panelLabel} layoutOpen={layoutOpen} setLayoutOpen={setLayoutOpen} onHelp={() => setHelpOpen(true)} onToggleSidebar={isNarrow ? () => setListOpen((o) => !o) : undefined} />

      <div className="flex min-h-0 flex-1">
        {isNarrow ? (
          /* sidebar as an overlay drawer */
          listOpen && (
            <div className="fixed inset-0 z-[95] flex">
              <button aria-label="Close sidebar" onClick={() => setListOpen(false)} className="absolute inset-0 animate-fade cursor-default bg-black/50 backdrop-blur-[2px]" />
              <div className="drawer-in relative h-full w-[300px] max-w-[86vw] shadow-e4">
                <Sidebar view={view} setView={(v) => { setView(v); setListOpen(false); }} sessions={sessions} activeId={activeId} onSelect={selectAndGo} onNew={(pid) => { openNew(undefined, pid); setListOpen(false); }} projects={projects} onUpdateProject={updateProject} onArchive={archiveSession} onDelete={deleteSession} onMoveProject={moveSession} onCreateProject={createProject} onDuplicate={duplicateSession} onTogglePin={togglePin} onRename={renameSession} onCollapse={() => setListOpen(false)} />
              </div>
            </div>
          )
        ) : listOpen ? (
          <><div style={{ width: list.size }} className="shrink-0"><Sidebar view={view} setView={setView} sessions={sessions} activeId={activeId} onSelect={selectAndGo} onNew={(pid) => openNew(undefined, pid)} projects={projects} onUpdateProject={updateProject} onArchive={archiveSession} onDelete={deleteSession} onMoveProject={moveSession} onCreateProject={createProject} onDuplicate={duplicateSession} onTogglePin={togglePin} onRename={renameSession} onCollapse={() => setListOpen(false)} /></div><div {...list.handle} /></>
        ) : (
          <button onClick={() => setListOpen(true)} title="Show sidebar · ⇧⌘B" className="flex w-[34px] shrink-0 cursor-pointer flex-col items-center gap-3 border-r border-line-soft bg-sunken pt-3 text-ink-4 transition-colors hover:text-ink">
            <LogoMark size={18} /><IconPanelLeft size={14} />
          </button>
        )}
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="flex min-h-0 flex-1">
            <>
                {/* Chat is always mounted; panels open as sheets over it. */}
                <div className="relative flex min-w-0 flex-1">
                  <Workbench {...wbProps} />
                  {!isWb && (
                    <Sheet title={SHEET_TITLES[view]?.[0] ?? view} eyebrow={SHEET_TITLES[view]?.[1]} onClose={closeSheet} streaming={streaming}>
                     <ErrorBoundary key={view} label={SHEET_TITLES[view]?.[0]}>
                      {view === "settings" && <SettingsView onStart={(a) => openNew(a)} />}
                      {view === "tasks" && <TasksView onOpenSession={selectAndGo} />}
                      {view === "skills" && <SkillsView />}
                      {view === "connectors" && <ConnectorsView />}
                      {view === "projects" && <ProjectsView projects={projects} sessions={sessions} onUpdate={updateProject} onNew={(pid) => openNew(undefined, pid)} onOpenThread={selectAndGo} />}
                      {view === "review" && <ReviewView />}
                      {view === "git" && <GitView onReview={() => setView("review")} />}
                      {view === "browser" && <BrowserView onAskAgent={send} />}
                      {view === "runs" && <FleetView />}
                     </ErrorBoundary>
                    </Sheet>
                  )}
                </div>
                {inspectorOpen && !isNarrow ? (
                  <><div {...pane.handle} /><div style={{ width: pane.size }} className="shrink-0"><ContextPane section={paneSection} setSection={setPaneSection} onNavigate={(v) => setView(v)} onAskAgent={send} onCollapse={() => setInspectorOpen(false)} onOpenTerminal={() => setTermOpen(true)} /></div></>
                ) : !isNarrow ? (
                  <button onClick={() => setInspectorOpen(true)} title="Open context panel · ⌘L" className="group flex w-[30px] shrink-0 cursor-pointer flex-col items-center gap-2 border-l border-line-soft bg-sunken pt-3 text-ink-4 transition-colors hover:text-iris-soft">
                    <IconSpark size={14} /><span className="font-mono text-[9.5px] tracking-[.18em] uppercase [writing-mode:vertical-rl]">context</span>
                    {streaming && <i className="size-[6px] animate-breathe rounded-full bg-cyan" />}
                  </button>
                ) : null}
            </>
          </div>
          {termOpen && <TerminalPanel onAskAgent={send} />}
        </div>
      </div>

      {statusOpen && (
        <footer className="flex h-[26px] shrink-0 items-center gap-3 border-t border-line-soft bg-void px-3 font-mono text-[10px] text-ink-4">
          <button onClick={() => setViewRaw("workbench")} className="flex cursor-pointer items-center gap-1.5 transition-colors hover:text-ink">
            <i className={cn("size-[5px] rounded-full", streaming ? "animate-breathe bg-cyan" : "bg-mint")} />{streaming ? "agent working" : "ready"}
          </button>
          <button onClick={cycleMode} title="Cycle mode · ⇧Tab" className="flex cursor-pointer items-center gap-1 transition-colors hover:text-ink">
            {product === "code" ? "mode" : "autonomy"} <span className={cn("tabular", modeMeta.tone === "rose" ? "text-rose" : modeMeta.tone === "cyan" ? "text-cyan" : modeMeta.tone === "iris" ? "text-iris-soft" : "text-amber")}>{modeMeta.short.toLowerCase()}</span>
          </button>
          {product === "code" && <span>agent <span className="text-ink-2">{agentById(agentId).name.toLowerCase()}</span></span>}
          <span className="hidden items-center gap-1.5 sm:flex">ctx <Ring value={62} size={11} stroke={2} tone="var(--color-ink-3)" /><span className="tabular text-ink-2">62%</span></span>
          <span className="hidden md:inline">{product === "code" ? "feat/ide-bridge · ↑3" : "workspace"}</span>
          <div className="ml-auto flex items-center gap-3">
            <span className="hidden lg:inline">⇧Tab mode</span><span className="hidden lg:inline">⌘J terminal</span><span className="hidden lg:inline">⌘L panel</span>
            <button onClick={() => setHelpOpen(true)} className="flex cursor-pointer items-center gap-1 transition-colors hover:text-ink"><IconHelp size={10} />all shortcuts</button>
          </div>
        </footer>
      )}

      <CommandPalette open={cmdk} onClose={() => setCmdk(false)} setView={setView} onSelectSession={selectAndGo} onNew={openNew} />
      <ShortcutHelp open={helpOpen} onClose={() => setHelpOpen(false)} />
      <TodoPopup />
      <VoiceAgent onSend={send} onNavigate={(target) => { if (target === "workbench") setViewRaw("workbench"); else setView(target); }} reply={latestReply} streaming={streaming} />
      <Toasts items={toasts} />

      <Modal open={newOpen} onClose={() => setNewOpen(false)} title="Start a session" sub="Pick the agent and the starting mode — both change later from the composer." width={660}>
        <div className="space-y-4 p-4">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {["aro", "claude-code", "codex", "cursor", "glm-code", "darwin", "zed", "opencode", "gemini-cli"].map((id) => { const a = agentById(id); const on = newAgent === id; const dis = a.status !== "connected"; return (
              <button key={id} disabled={dis} onClick={() => setNewAgent(id)} className={cn("flex cursor-pointer items-start gap-2 rounded-[10px] border p-2.5 text-left transition-all", on ? "border-iris/50 bg-iris-tint shadow-glow-iris" : "border-line-soft bg-raise hover:border-line-strong", dis && "cursor-not-allowed opacity-40")}>
                <AgentMark glyph={a.glyph} from={a.from} to={a.to} size={22} /><span className="min-w-0 flex-1"><span className="block truncate text-[12.5px] font-medium text-ink">{a.name}</span><span className="block truncate font-mono text-[9.5px] text-ink-4">{dis ? a.status.replace("-", " ") : a.models[0].label}</span></span>{on && <IconCheck size={12} className="text-iris-soft" />}
              </button>); })}
          </div>
          <div>
            <span className="mb-2 block font-mono text-[9.5px] tracking-[.16em] text-ink-4 uppercase">starting mode</span>
            <Segmented value={newMode} onChange={setNewMode} size="md" items={MODES.map((m) => ({ value: m.id, label: m.label, tone: m.tone }))} />
            <p className="mt-2 text-[12px] text-ink-3">{modeDefs(product).find((m) => m.id === newMode)!.desc}</p>
          </div>
          <div className="flex items-center gap-2 border-t border-line-soft pt-3">
            <Badge tone="neutral" mono dot>aro/platform</Badge>
            <span className="flex items-center gap-1.5 font-mono text-[10px] text-ink-4"><Kbd>⌘N</Kbd> skips this dialog next time</span>
            <div className="ml-auto flex gap-2"><Button variant="ghost" onClick={() => setNewOpen(false)}>Cancel</Button><Button variant="primary" icon={IconBolt} onClick={() => create()}>Start</Button></div>
          </div>
        </div>
      </Modal>
      <Boot />
    </div>
  );
}

function Boot() {
  const [gone, setGone] = useState(false);
  useEffect(() => { const t = setTimeout(() => setGone(true), 380); return () => clearTimeout(t); }, []);
  if (gone) return null;
  return (
    <div className="fixed inset-0 z-[200] flex flex-col items-center justify-center gap-3 bg-void">
      <LogoMark size={42} />
      <span className="font-display text-[13px] font-semibold tracking-[.22em] text-ink-3 uppercase">Aro</span>
      <span className="flex items-center gap-1.5 font-mono text-[10px] text-ink-4"><IconTerminal size={10} className="animate-breathe text-iris-soft" />attaching daemon…</span>
    </div>
  );
}

export default function App() { return <AppProvider><ErrorBoundary label="Aro"><Shell /></ErrorBoundary></AppProvider>; }
