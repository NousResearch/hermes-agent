"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "../utils/cn";
import { useApp, useResizableY, type TermLine } from "../lib/app";
import { IconChevronDown, IconPlus, IconSpark, IconTerminal, IconWarning, IconX, IconRefresh } from "./Icons";
import { IconButton, Tip } from "./ui";

type Tab = { id: string; label: string; kind: "agent" | "shell" | "dev" | "problems" };

const DEV_LINES: TermLine[] = [
  { t: "$ pnpm dev", k: "cmd" },
  { t: "  VITE v7.3.2  ready in 412 ms", k: "ok" },
  { t: "  ➜  Local:   http://localhost:5173/", k: "info" },
  { t: "  ➜  Network: use --host to expose", k: "dim" },
  { t: "[hmr] update /packages/review/src/FileList.tsx", k: "dim" },
  { t: "[hmr] update /packages/review/src/DiffView.tsx", k: "dim" },
];
const PROBLEMS = [
  { sev: "err", file: "DiffView.tsx", line: 118, msg: "ResizeObserver loop completed with undelivered notifications" },
  { sev: "warn", file: "FileList.tsx", line: 42, msg: "Each child in a list should have a unique \"key\" prop" },
  { sev: "warn", file: "FileRow.tsx", line: 17, msg: "Dropdown clipped by parent overflow: hidden" },
];

function run(cmd: string): TermLine[] {
  const c = cmd.trim();
  if (!c) return [];
  const map: Record<string, TermLine[]> = {
    help: [{ t: "built-ins: ls · pwd · git status · git log · pnpm test · whoami · agents · clear", k: "dim" }, { t: "prefix with  ?  to ask the agent, e.g.  ? why is reconnect.test flaky", k: "dim" }],
    ls: [{ t: "README.md  docs/  editors/  package.json  packages/  pnpm-lock.yaml  turbo.json", k: "info" }],
    pwd: [{ t: "/Users/dev/code/aro", k: "info" }],
    whoami: [{ t: "dev", k: "info" }],
    "git status": [{ t: "On branch feat/ide-bridge · ahead of origin by 3 commits", k: "info" }, { t: "  modified:   packages/bridge/src/server.ts", k: "warn" }, { t: "  modified:   packages/editor-kit/src/attach.ts", k: "warn" }, { t: "  new file:   packages/bridge/test/reconnect.test.ts", k: "ok" }],
    "git log": [{ t: "a91f3c2 bridge: resumable SessionChannel with 12ms flush  (claude-code)", k: "info" }, { t: "77bd0e4 editor-kit: capability handshake + backoff  (claude-code)", k: "info" }, { t: "1c3a9f8 review: accept editor-originated patches  (dev)", k: "info" }],
    "pnpm test": [{ t: "RUN v3.2.4", k: "dim" }, { t: " ✓ session-channel.test.ts (14)", k: "ok" }, { t: " ✓ attach-editor.test.ts (9)", k: "ok" }, { t: " ✓ reconnect.test.ts (12)", k: "ok" }, { t: "Test Files 3 passed · Tests 35 passed · 1.2s", k: "ok" }],
    agents: [{ t: "claude-code  connected  v2.1.239", k: "ok" }, { t: "codex        connected  v0.94.2", k: "ok" }, { t: "aro          connected  v1.5.0  backend=docker", k: "ok" }, { t: "opencode     degraded", k: "warn" }],
  };
  if (c.startsWith("?")) return [{ t: `→ sent to agent: "${c.slice(1).trim()}"`, k: "info" }];
  return map[c] ?? [{ t: `zsh: command not found: ${c.split(" ")[0]}  (try "help")`, k: "err" }];
}

function Lines({ lines }: { lines: TermLine[] }) {
  return (
    <>
      {lines.map((l, i) => (
        <div key={i} className={cn("whitespace-pre-wrap", l.k === "cmd" && "text-ink", l.k === "ok" && "text-mint", l.k === "err" && "text-rose", l.k === "warn" && "text-amber", l.k === "info" && "text-sky", l.k === "dim" && "text-ink-4", !l.k && "text-ink-3")}>
          {l.k === "cmd" ? <><span className="text-iris-soft">❯ </span>{l.t.replace(/^\$ /, "")}</> : l.t}
        </div>
      ))}
    </>
  );
}

export function TerminalPanel({ onAskAgent }: { onAskAgent: (t: string) => void }) {
  const { termOpen, setTermOpen, termFeed } = useApp();
  const rz = useResizableY(230, 120, 560);
  const [max, setMax] = useState(false);
  const [tabs, setTabs] = useState<Tab[]>([
    { id: "agent", label: "agent", kind: "agent" },
    { id: "zsh-1", label: "zsh", kind: "shell" },
    { id: "dev", label: "pnpm dev", kind: "dev" },
    { id: "problems", label: "Problems", kind: "problems" },
  ]);
  const [active, setActive] = useState("agent");
  const [shells, setShells] = useState<Record<string, TermLine[]>>({ "zsh-1": [{ t: "Last login: today on ttys004 · type help", k: "dim" }] });
  const [input, setInput] = useState("");
  const [hist, setHist] = useState<string[]>([]);
  const [hi, setHi] = useState(-1);
  const body = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const tab = tabs.find((t) => t.id === active)!;

  useEffect(() => { body.current?.scrollTo({ top: 1e6 }); }, [termFeed.length, shells, active, termOpen]);

  if (!termOpen) return null;

  const submit = () => {
    const c = input;
    setHist((h) => [c, ...h]); setHi(-1); setInput("");
    if (c.trim() === "clear") { setShells((s) => ({ ...s, [active]: [] })); return; }
    if (c.trim().startsWith("?")) onAskAgent(c.trim().slice(1).trim());
    setShells((s) => ({ ...s, [active]: [...(s[active] ?? []), { t: c, k: "cmd" }, ...run(c)] }));
  };
  const newShell = () => {
    const n = tabs.filter((t) => t.kind === "shell").length + 1;
    const id = `zsh-${Date.now()}`;
    setTabs((t) => [...t.slice(0, -1), { id, label: `zsh ${n}`, kind: "shell" }, t[t.length - 1]]);
    setShells((s) => ({ ...s, [id]: [{ t: "new shell · ~/code/aro", k: "dim" }] }));
    setActive(id);
  };

  return (
    <>
      {!max && <div {...rz.handle} />}
      <div style={{ height: max ? "70vh" : rz.size }} className="flex shrink-0 flex-col border-t border-line-soft bg-code">
        <div className="flex h-[34px] shrink-0 items-center gap-1 border-b border-line-soft px-2">
          <div className="no-scrollbar flex min-w-0 items-center gap-0.5 overflow-x-auto">
            {tabs.map((t) => (
              <button key={t.id} onClick={() => { setActive(t.id); setTimeout(() => inputRef.current?.focus(), 0); }}
                className={cn("group flex h-[26px] shrink-0 cursor-pointer items-center gap-1.5 rounded-[6px] px-2 font-mono text-[11px] transition-colors", active === t.id ? "bg-raise text-ink shadow-e1" : "text-ink-3 hover:text-ink-2")}>
                {t.kind === "agent" ? <IconSpark size={11} className="text-iris-soft" /> : t.kind === "problems" ? <IconWarning size={11} className="text-amber" /> : <IconTerminal size={11} />}
                {t.label}
                {t.kind === "agent" && <i className="size-[5px] animate-breathe rounded-full bg-cyan" />}
                {t.kind === "problems" && <span className="rounded-full bg-rose-tint px-1 text-[9px] text-rose">{PROBLEMS.length}</span>}
                {t.kind === "shell" && tabs.filter((x) => x.kind === "shell").length > 1 && (
                  <span onClick={(e) => { e.stopPropagation(); setTabs((ts) => ts.filter((x) => x.id !== t.id)); if (active === t.id) setActive("agent"); }} className="opacity-0 group-hover:opacity-100 hover:text-rose"><IconX size={10} /></span>
                )}
              </button>
            ))}
          </div>
          <Tip label="New terminal"><IconButton icon={IconPlus} label="New terminal" size={24} onClick={newShell} /></Tip>
          <span className="ml-auto hidden font-mono text-[10px] text-ink-4 md:inline">{tab.kind === "agent" ? "read-only · agent output" : tab.kind === "shell" ? "~/code/aro · zsh · prefix ? to ask the agent" : tab.kind === "dev" ? "localhost:5173 · hmr" : "from build + browser console"}</span>
          <Tip label="Clear"><IconButton icon={IconRefresh} label="Clear" size={24} onClick={() => setShells((s) => ({ ...s, [active]: [] }))} /></Tip>
          <Tip label={max ? "Restore" : "Maximise"}><IconButton icon={IconChevronDown} label="Maximise" size={24} className={cn(!max && "rotate-180")} onClick={() => setMax((m) => !m)} /></Tip>
          <Tip label="Hide panel · ⌘J"><IconButton icon={IconX} label="Close" size={24} onClick={() => setTermOpen(false)} /></Tip>
        </div>

        <div ref={body} onClick={() => inputRef.current?.focus()} className="scroll-thin min-h-0 flex-1 overflow-y-auto px-3 py-2 font-mono text-[11.5px] leading-[1.7]">
          {tab.kind === "agent" && <Lines lines={termFeed} />}
          {tab.kind === "dev" && <Lines lines={DEV_LINES} />}
          {tab.kind === "problems" && (
            <div className="space-y-0.5">
              {PROBLEMS.map((p) => (
                <div key={p.msg} className="group flex items-center gap-2 rounded-[5px] px-1.5 py-[3px] hover:bg-hover/50">
                  <span className={p.sev === "err" ? "text-rose" : "text-amber"}>{p.sev === "err" ? "✗" : "⚠"}</span>
                  <span className="text-ink-2">{p.msg}</span>
                  <span className="text-ink-4">{p.file}:{p.line}</span>
                  <button onClick={() => onAskAgent(`Fix: ${p.msg} at ${p.file}:${p.line}`)} className="ml-auto cursor-pointer rounded bg-iris-tint px-1.5 text-[10px] text-iris-soft opacity-0 group-hover:opacity-100">fix with agent</button>
                </div>
              ))}
            </div>
          )}
          {tab.kind === "shell" && (
            <>
              <Lines lines={shells[active] ?? []} />
              <div className="flex items-center">
                <span className="text-iris-soft">❯&nbsp;</span>
                <input ref={inputRef} value={input} autoFocus spellCheck={false}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") submit();
                    if (e.key === "ArrowUp") { e.preventDefault(); const n = Math.min(hi + 1, hist.length - 1); setHi(n); setInput(hist[n] ?? ""); }
                    if (e.key === "ArrowDown") { e.preventDefault(); const n = Math.max(hi - 1, -1); setHi(n); setInput(n === -1 ? "" : hist[n]); }
                    if (e.key === "l" && e.ctrlKey) { e.preventDefault(); setShells((s) => ({ ...s, [active]: [] })); }
                  }}
                  className="min-w-0 flex-1 bg-transparent text-ink caret-iris focus:outline-none" />
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
