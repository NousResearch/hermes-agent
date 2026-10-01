"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { cn } from "../utils/cn";
import { AGENTS, BRIDGES, RUNS, SESSIONS, SLASH_COMMANDS, agentById } from "../data/catalog";
import { AgentMark, IconBolt, IconFile, IconGit, IconLayers, IconMonitor, IconPalette, IconPulse, IconSearch, IconSpark, IconTerminal } from "./Icons";
import { Kbd } from "./ui";
import type { ViewId } from "./Sidebar";

type Item = {
  id: string;
  group: "Jump to" | "Sessions" | "Agents" | "Commands" | "Editors";
  label: string;
  hint?: string;
  glyph?: React.ReactNode;
  run: () => void;
};

export function CommandPalette({
  open,
  onClose,
  setView,
  onSelectSession,
  onNew,
}: {
  open: boolean;
  onClose: () => void;
  setView: (v: ViewId) => void;
  onSelectSession: (id: string) => void;
  onNew: (agentId?: string) => void;
}) {
  const [q, setQ] = useState("");
  const [idx, setIdx] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);
  /* reset the selection whenever the query or the palette itself opens */
  const [prevCtx, setPrevCtx] = useState(`${q}|${open}`);
  const ctx = `${q}|${open}`;
  if (prevCtx !== ctx) {
    setPrevCtx(ctx);
    setIdx(0);
  }

  const items = useMemo<Item[]>(() => {
    const nav: Item[] = [
      { id: "v1", group: "Jump to", label: "Workbench — live sessions", glyph: <IconPulse size={13} />, run: () => setView("workbench") },
      { id: "v2", group: "Jump to", label: "Fleet — parallel runs", glyph: <IconLayers size={13} />, run: () => setView("runs") },
      { id: "v3", group: "Jump to", label: "Agents — connected CLIs", glyph: <IconBolt size={13} />, run: () => setView("agents") },
      { id: "v4", group: "Jump to", label: "Editors — external IDE bridge", glyph: <IconMonitor size={13} />, run: () => setView("bridges") },
      { id: "v5", group: "Jump to", label: "Review — diffs & checkpoints", glyph: <IconGit size={13} />, run: () => setView("review") },
      { id: "v6", group: "Jump to", label: "Design system reference", glyph: <IconPalette size={13} />, run: () => setView("design") },
      { id: "v7", group: "Jump to", label: "Tasks — work graph", glyph: <IconFile size={13} />, run: () => setView("tasks") },
      { id: "v12", group: "Jump to", label: "Skills — runnable procedures", glyph: <IconSpark size={13} />, run: () => setView("skills") },
      { id: "v13", group: "Jump to", label: "Connectors — MCP servers & tools", glyph: <IconSpark size={13} />, run: () => setView("connectors") },
      { id: "v14", group: "Jump to", label: "Projects — inherited defaults", glyph: <IconFile size={13} />, run: () => setView("projects") },
      { id: "v8", group: "Jump to", label: "Git — repos, commits, PRs", glyph: <IconGit size={13} />, run: () => setView("git") },
      { id: "v9", group: "Jump to", label: "Browser — preview & pick elements", glyph: <IconMonitor size={13} />, run: () => setView("browser") },
      { id: "v10", group: "Jump to", label: "Settings", hint: "⌘,", glyph: <IconSpark size={13} />, run: () => setView("settings") },
      { id: "v11", group: "Jump to", label: "Settings → Permissions & default mode", glyph: <IconSpark size={13} />, run: () => setView("settings") },
    ];
    const sess: Item[] = SESSIONS.map((s) => {
      const a = agentById(s.agentId);
      return {
        id: s.id,
        group: "Sessions",
        label: s.title,
        hint: `${a.name} · ${s.cost}`,
        glyph: <AgentMark glyph={a.glyph} from={a.from} to={a.to} size={15} />,
        run: () => onSelectSession(s.id),
      };
    });
    const ag: Item[] = AGENTS.map((a) => ({
      id: a.id,
      group: "Agents",
      label: `New session with ${a.name}`,
      hint: a.vendor,
      glyph: <AgentMark glyph={a.glyph} from={a.from} to={a.to} size={15} />,
      run: () => onNew(a.id),
    }));
    const cmds: Item[] = SLASH_COMMANDS.map((c) => ({
      id: c.cmd,
      group: "Commands",
      label: c.cmd,
      hint: c.desc,
      glyph: <IconTerminal size={13} />,
      run: () => setView("workbench"),
    }));
    const eds: Item[] = BRIDGES.map((b) => ({
      id: b.id,
      group: "Editors",
      label: `Attach ${b.ide} to this session`,
      hint: b.latency,
      glyph: <IconMonitor size={13} />,
      run: () => setView("bridges"),
    }));
    const runs: Item[] = RUNS.map((r) => ({
      id: r.id,
      group: "Sessions",
      label: `Run · ${r.title}`,
      hint: r.worktree,
      glyph: <IconFile size={13} />,
      run: () => setView("runs"),
    }));
    return [...nav, ...sess, ...runs, ...ag, ...cmds, ...eds];
  }, [setView, onSelectSession, onNew]);

  const filtered = useMemo(() => {
    if (!q.trim()) return items.slice(0, 22);
    const t = q.toLowerCase();
    return items
      .filter((i) => (i.label + " " + (i.hint ?? "")).toLowerCase().includes(t))
      .slice(0, 22);
  }, [q, items]);

  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setIdx((i) => Math.min(i + 1, filtered.length - 1));
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setIdx((i) => Math.max(i - 1, 0));
      }
      if (e.key === "Enter") {
        e.preventDefault();
        filtered[idx]?.run();
        onClose();
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, filtered, idx, onClose]);

  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-idx="${idx}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [idx]);

  if (!open) return null;

  let lastGroup = "";
  return (
    <div className="fixed inset-0 z-[120] flex items-start justify-center pt-[10vh]">
      <div className="absolute inset-0 animate-fade bg-black/70 backdrop-blur-[4px]" onClick={onClose} />
      <div className="relative w-[min(660px,92vw)] animate-pop overflow-hidden rounded-[14px] border border-line-strong bg-overlay shadow-e4">
        <div className="flex items-center gap-2.5 border-b border-line-soft px-3.5 py-3">
          <IconSearch size={15} className="shrink-0 text-ink-4" />
          <input
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search sessions, agents, editors, commands…"
            className="min-w-0 flex-1 bg-transparent text-[13.5px] text-ink placeholder:text-ink-4 focus:outline-none"
          />
          <Kbd>esc</Kbd>
        </div>

        <div ref={listRef} className="scroll-thin max-h-[52vh] overflow-y-auto p-1.5">
          {filtered.length === 0 && (
            <div className="px-3 py-10 text-center text-[12px] text-ink-4">No matches</div>
          )}
          {filtered.map((it, i) => {
            const showGroup = it.group !== lastGroup;
            lastGroup = it.group;
            return (
              <div key={it.id + i}>
                {showGroup && (
                  <div className="px-2 pt-2.5 pb-1 font-mono text-[9px] tracking-[.16em] text-ink-4 uppercase">
                    {it.group}
                  </div>
                )}
                <button
                  data-idx={i}
                  onMouseEnter={() => setIdx(i)}
                  onClick={() => {
                    it.run();
                    onClose();
                  }}
                  className={cn(
                    "flex w-full cursor-pointer items-center gap-2.5 rounded-[8px] px-2 py-[7px] text-left transition-colors",
                    i === idx ? "bg-iris-tint" : "hover:bg-hover",
                  )}
                >
                  <span className={cn("shrink-0", i === idx ? "text-iris-soft" : "text-ink-4")}>{it.glyph}</span>
                  <span
                    className={cn(
                      "min-w-0 flex-1 truncate text-[12.5px]",
                      i === idx ? "text-ink" : "text-ink-2",
                    )}
                  >
                    {it.label}
                  </span>
                  {it.hint && <span className="shrink-0 truncate font-mono text-[10px] text-ink-4">{it.hint}</span>}
                  {i === idx && <Kbd>⏎</Kbd>}
                </button>
              </div>
            );
          })}
        </div>

        <div className="flex items-center gap-3 border-t border-line-soft bg-raise/60 px-3 py-2">
          <span className="flex items-center gap-1.5 font-mono text-[9.5px] text-ink-4">
            <Kbd>↑</Kbd>
            <Kbd>↓</Kbd> navigate
          </span>
          <span className="flex items-center gap-1.5 font-mono text-[9.5px] text-ink-4">
            <Kbd>⏎</Kbd> run
          </span>
          <button onClick={() => { onClose(); window.dispatchEvent(new CustomEvent("aro:help")); }} className="ml-auto flex cursor-pointer items-center gap-1.5 font-mono text-[9.5px] text-ink-4 transition-colors hover:text-iris-soft">
            <Kbd>⌘</Kbd><Kbd>/</Kbd> all shortcuts
          </button>
        </div>
      </div>
    </div>
  );
}
