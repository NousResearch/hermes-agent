"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { cn } from "../utils/cn";
import { AGENTS, agentById, type ModeId, type Session } from "../data/catalog";
import { ASSISTANT } from "../data/extra";
import { MODES } from "../data/catalog";
import type { Project } from "../data/providers";
import { useApp } from "../lib/app";
import {
  AgentMark, IconArchive, IconBrain, IconCheck, IconChevronDown, IconClock, IconCopy, IconFolder, IconGrip, IconLayers, IconList, IconMore, IconPanelLeft,
  IconPencil, IconPin, IconPlug, IconPlus, IconPulse, IconSearch, IconSettings, IconShield, IconSpark, IconTrash, IconX, IconZap, LogoMark,
} from "./Icons";
import { Badge, Button, IconButton, Input, Kbd, Menu, MenuItem, MenuLabel, MenuSep, Modal, Segmented, Select, Tip, Toggle } from "./ui";

export type ViewId =
  | "workbench" | "tasks" | "skills" | "connectors" | "projects"
  | "automations" | "memory" | "artifacts"
  | "runs" | "agents" | "bridges" | "git" | "browser" | "review" | "design" | "settings";

/* Sidebar navigation — Codex style: a short list, no icon rail.
   Fleet / Changes / Git / Browser live in the right context pane as icons. */
export const NAV_CODE: { id: ViewId; label: string; icon: React.ComponentType<{ size?: number; className?: string }>; key: string }[] = [
  { id: "workbench", label: "Threads", icon: IconPulse, key: "1" },
  { id: "tasks", label: "Tasks", icon: IconList, key: "2" },
  { id: "automations", label: "Automations", icon: IconClock, key: "3" },
  { id: "skills", label: "Skills", icon: IconSpark, key: "4" },
  { id: "connectors", label: "Connectors", icon: IconPlug, key: "5" },
  { id: "projects", label: "Projects", icon: IconFolder, key: "6" },
  { id: "memory", label: "Memory", icon: IconBrain, key: "7" },
  { id: "artifacts", label: "Artifacts", icon: IconPin, key: "8" },
];
export const NAV_AGENT: typeof NAV_CODE = [
  { id: "workbench", label: "Chats", icon: IconPulse, key: "1" },
  { id: "tasks", label: "Tasks", icon: IconList, key: "2" },
  { id: "automations", label: "Automations", icon: IconClock, key: "3" },
  { id: "skills", label: "Skills", icon: IconSpark, key: "4" },
  { id: "connectors", label: "Connectors", icon: IconPlug, key: "5" },
  { id: "projects", label: "Projects", icon: IconFolder, key: "6" },
  { id: "memory", label: "Memory", icon: IconBrain, key: "7" },
  { id: "artifacts", label: "Artifacts", icon: IconPin, key: "8" },
];

const statusDot: Record<Session["status"], string> = { running: "bg-cyan animate-breathe", idle: "bg-line-strong", waiting: "bg-amber", done: "bg-mint", failed: "bg-rose" };
type Grouping = "project" | "date" | "agent";

const LS = {
  folds: "aro.sidebar.folds",
  collapsed: "aro.sidebar.collapsed",
  grouping: "aro.sidebar.grouping",
};
const readJSON = <T,>(key: string, fallback: T): T => {
  try { const v = JSON.parse(localStorage.getItem(key) || "null"); return v ?? fallback; } catch { return fallback; }
};

export type SidebarProps = {
  view: ViewId; setView: (v: ViewId) => void;
  sessions: Session[]; activeId: string; onSelect: (id: string) => void; onNew: (projectId?: string) => void;
  projects: Project[]; onUpdateProject: (p: Project) => void;
  onArchive: (id: string, archived: boolean) => void; onDelete: (id: string) => void; onMoveProject: (id: string, projectId: string) => void;
  onCreateProject: (name: string) => string; onDuplicate: (id: string) => void; onTogglePin: (id: string) => void; onRename: (id: string, title: string) => void;
  onCollapse: () => void;
};

/* ============================================================
   CODEX-STYLE SIDEBAR
   New thread · search · Tasks / Skills / Connectors / Projects
   · threads nested under projects · todo + settings + account
   ============================================================ */
export function Sidebar(p: SidebarProps) {
  const { product, todos, setTodoOpen, toast } = useApp();
  const nav = product === "code" ? NAV_CODE : NAV_AGENT;
  const [q, setQ] = useState("");
  const [grouping, setGrouping] = useState<Grouping>(() => readJSON<Grouping>(LS.grouping, "project"));
  const [folds, setFolds] = useState<Record<string, boolean>>(() => readJSON(LS.folds, {}));
  const [showArchived, setShowArchived] = useState(false);
  const [dragId, setDragId] = useState<string | null>(null);
  const [dropKey, setDropKey] = useState<string | null>(null);
  const [cursor, setCursor] = useState(-1);
  const [deleteTarget, setDeleteTarget] = useState<Session | null>(null);
  const [renameTarget, setRenameTarget] = useState<Session | null>(null);
  const [renameTitle, setRenameTitle] = useState("");
  const [projectTarget, setProjectTarget] = useState<Project | null>(null);
  const [newProject, setNewProject] = useState("");
  const [creatingProject, setCreatingProject] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const openTodos = todos.filter((t) => t.state !== "done").length;

  useEffect(() => localStorage.setItem(LS.folds, JSON.stringify(folds)), [folds]);
  useEffect(() => localStorage.setItem(LS.grouping, JSON.stringify(grouping)), [grouping]);
  useEffect(() => {
    const h = () => setShowArchived((v) => !v);
    window.addEventListener("aro:archive", h);
    return () => window.removeEventListener("aro:archive", h);
  }, []);

  const visible = useMemo(
    () => p.sessions.filter((s) => Boolean(s.archived) === showArchived && s.title.toLowerCase().includes(q.toLowerCase())),
    [p.sessions, showArchived, q],
  );
  const pinned = visible.filter((s) => s.favorite || s.pinned);
  const attentionTotal = visible.filter((s) => s.status === "waiting" || s.status === "failed").length;

  const groupKey = (s: Session) => grouping === "project" ? (s.projectId ?? "platform") : grouping === "date" ? s.group : product === "agent" ? "Aro" : agentById(s.agentId).name;
  const groupLabel = (k: string) => grouping === "project" ? (p.projects.find((x) => x.id === k)?.name ?? "Unsorted") : k;
  const groupColor = (k: string) => grouping === "project" ? p.projects.find((x) => x.id === k)?.color : undefined;
  const groupKeys = grouping === "project"
    ? [...p.projects.map((x) => x.id), ...[...new Set(visible.map(groupKey))].filter((k) => !p.projects.some((x) => x.id === k))]
    : [...new Set(visible.map(groupKey))];

  /* flat order for keyboard navigation */
  const flat = useMemo(() => {
    const out: Session[] = [];
    if (pinned.length && !showArchived) out.push(...pinned);
    groupKeys.forEach((k) => { if (!folds[k]) out.push(...visible.filter((s) => groupKey(s) === k && !(pinned.includes(s) && !showArchived))); });
    return out;
  }, [visible, groupKeys, folds, pinned, showArchived, grouping, product]);

  useEffect(() => { if (cursor >= flat.length) setCursor(-1); }, [flat.length, cursor]);
  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-row="${cursor}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [cursor]);

  const onListKeyDown = (e: React.KeyboardEvent) => {
    if (!flat.length) return;
    const start = cursor < 0 ? Math.max(0, flat.findIndex((s) => s.id === p.activeId)) : cursor;
    if (e.key === "ArrowDown") { e.preventDefault(); setCursor(Math.min(start + 1, flat.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setCursor(Math.max(start - 1, 0)); }
    else if (e.key === "Enter" && cursor >= 0) { e.preventDefault(); p.onSelect(flat[cursor].id); }
    else if ((e.key === "Backspace" || e.key === "Delete") && cursor >= 0) { e.preventDefault(); const s = flat[cursor]; p.onArchive(s.id, !s.archived); toast(s.archived ? `Restored “${s.title.slice(0, 28)}”` : `Archived “${s.title.slice(0, 28)}”`, "mint"); }
    else if (e.key === "Home") { e.preventDefault(); setCursor(0); }
    else if (e.key === "End") { e.preventDefault(); setCursor(flat.length - 1); }
  };

  const submitProject = () => { const name = newProject.trim(); if (!name) return; p.onCreateProject(name); setNewProject(""); setCreatingProject(false); };

  /* ---------------- thread row ---------------- */
  const Row = ({ s, index }: { s: Session; index: number }) => {
    const a = agentById(s.agentId);
    const on = s.id === p.activeId && p.view === "workbench";
    const focused = index === cursor;
    return (
      <div
        data-row={index}
        draggable
        onDragStart={(e) => { setDragId(s.id); e.dataTransfer.effectAllowed = "move"; }}
        onDragEnd={() => { setDragId(null); setDropKey(null); }}
        onClick={() => { p.onSelect(s.id); setCursor(index); }}
        tabIndex={0}
        role="option"
        aria-selected={on}
        className={cn(
          "group relative flex w-full cursor-pointer items-center gap-1.5 rounded-[7px] py-[5px] pr-1 pl-1.5 text-left outline-none transition-all duration-150",
          on ? "bg-raise text-ink shadow-e1 hairline" : "text-ink-2 hover:bg-raise/60 hover:text-ink",
          focused && "ring-1 ring-iris/60",
          dragId === s.id && "opacity-40",
        )}
      >
        {on && <span className="absolute top-1.5 bottom-1.5 -left-[5px] w-[2px] rounded-full bg-iris" />}
        <IconGrip size={10} className="shrink-0 text-ink-4 opacity-0 transition-opacity group-hover:opacity-70" />
        <span className={cn("size-[6px] shrink-0 rounded-full", statusDot[s.status])} title={s.status} />
        <span className="min-w-0 flex-1 truncate text-[12px] leading-[1.35]">{s.title}</span>
        {s.status === "waiting" && <span className="shrink-0 rounded-[3px] bg-amber-tint px-1 font-mono text-[8px] font-bold tracking-wide text-amber uppercase">you</span>}
        {(s.favorite || s.pinned) && <IconPin size={9} className="shrink-0 text-amber opacity-80" />}
        {product === "code" && <AgentMark glyph={a.glyph} from={a.from} to={a.to} size={13} className="shrink-0 opacity-55 transition-opacity group-hover:opacity-100" />}
        <div onClick={(e) => e.stopPropagation()} className="shrink-0 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
          <Menu align="right" width={240} trigger={() => <IconButton icon={IconMore} label="Thread actions" size={22} />}>
            {(close) => <>
              <MenuLabel>{s.title.slice(0, 32)}{s.title.length > 32 ? "…" : ""}</MenuLabel>
              <MenuItem icon={<IconPencil size={12} />} label="Rename…" onClick={() => { setRenameTarget(s); setRenameTitle(s.title); close(); }} />
              <MenuItem icon={<IconPin size={12} />} label={s.favorite || s.pinned ? "Unpin" : "Pin to top"} onClick={() => { p.onTogglePin(s.id); close(); }} />
              <MenuItem icon={<IconCopy size={12} />} label="Duplicate" onClick={() => { p.onDuplicate(s.id); close(); }} />
              <MenuSep /><MenuLabel>move to project</MenuLabel>
              {p.projects.map((pr) => <MenuItem key={pr.id} icon={<i className="size-[8px] rounded-[2px]" style={{ background: pr.color }} />} label={pr.name} active={pr.id === s.projectId} onClick={() => { p.onMoveProject(s.id, pr.id); close(); }} />)}
              <MenuSep />
              <MenuItem icon={<IconArchive size={12} />} label={s.archived ? "Restore" : "Archive"} hint="⌫" onClick={() => { p.onArchive(s.id, !s.archived); close(); }} />
              <MenuItem danger icon={<IconTrash size={12} />} label="Delete…" onClick={() => { setDeleteTarget(s); close(); }} />
            </>}
          </Menu>
        </div>
      </div>
    );
  };

  /* ---------------- project header (drop target) ---------------- */
  const ProjectHeader = ({ k, items }: { k: string; items: Session[] }) => {
    const isCollapsed = folds[k];
    const color = groupColor(k);
    const attention = items.filter((s) => s.status === "waiting" || s.status === "failed").length;
    const running = items.filter((s) => s.status === "running").length;
    const project = grouping === "project" ? p.projects.find((x) => x.id === k) : undefined;
    const isDrop = dropKey === k && dragId;
    return (
      <div
        onDragOver={(e) => { if (grouping !== "project" || !dragId) return; e.preventDefault(); setDropKey(k); }}
        onDragLeave={() => setDropKey((d) => (d === k ? null : d))}
        onDrop={(e) => {
          e.preventDefault();
          if (grouping !== "project" || !dragId) return;
          p.onMoveProject(dragId, k);
          setDragId(null); setDropKey(null);
        }}
        className={cn("group -mx-1 rounded-[7px] px-1 transition-colors", isDrop && "bg-iris-tint ring-1 ring-iris/50")}
      >
        <div className="flex items-center gap-1 py-[5px]">
          <button onClick={() => setFolds((f) => ({ ...f, [k]: !f[k] }))} className="flex min-w-0 flex-1 cursor-pointer items-center gap-1.5 text-left" title={isCollapsed ? "Expand" : "Collapse"}>
            <IconChevronDown size={11} className={cn("shrink-0 text-ink-4 transition-transform duration-200", isCollapsed && "-rotate-90")} />
            {color && <i className="size-[7px] shrink-0 rounded-[2px] transition-transform group-hover:scale-125" style={{ background: color }} />}
            <span className="truncate text-[12px] font-medium text-ink-2 group-hover:text-ink">{groupLabel(k)}</span>
            <span className="font-mono text-[9.5px] text-ink-4">{items.length}</span>
            {attention > 0 && (
              <Tip label={`${attention} thread${attention > 1 ? "s" : ""} need attention`}>
                <span className="flex h-[14px] min-w-[14px] shrink-0 items-center justify-center rounded-full bg-amber px-1 font-mono text-[8.5px] font-bold text-on-iris shadow-[0_0_0_3px_color-mix(in_srgb,var(--color-amber)_18%,transparent)]">{attention}</span>
              </Tip>
            )}
            {running > 0 && <i className="size-[5px] shrink-0 animate-breathe rounded-full bg-cyan" title={`${running} running`} />}
          </button>
          {grouping === "project" && !showArchived && (
            <div onClick={(e) => e.stopPropagation()} className="flex shrink-0 items-center opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
              <Tip label="New thread here"><button onClick={() => p.onNew(k)} className="flex size-[20px] cursor-pointer items-center justify-center rounded-[5px] text-ink-4 hover:bg-hover hover:text-ink"><IconPlus size={11} /></button></Tip>
              {project && <Tip label="Project settings"><button onClick={() => setProjectTarget(project)} className="flex size-[20px] cursor-pointer items-center justify-center rounded-[5px] text-ink-4 hover:bg-hover hover:text-ink"><IconSettings size={11} /></button></Tip>}
            </div>
          )}
        </div>
      </div>
    );
  };

  let rowIndex = -1;
  const nextIndex = () => ++rowIndex;

  return (
    <aside className="flex h-full min-w-0 flex-col border-r border-line-soft bg-sunken">
      {/* brand */}
      <div className="flex h-[44px] shrink-0 items-center gap-2 px-3">
        <LogoMark size={20} />
        <span className="font-display text-[13.5px] font-semibold tracking-[-.02em] text-ink">Aro</span>
        <span className="rounded-[4px] border border-line bg-raise px-1 py-[1px] font-mono text-[8.5px] text-ink-4">{product === "code" ? "CODE" : "AGENT"}</span>
        <Tip label="Hide sidebar · ⇧⌘B"><IconButton icon={IconPanelLeft} label="Hide sidebar" size={26} className="ml-auto" onClick={p.onCollapse} /></Tip>
      </div>

      {/* new thread + search */}
      <div className="space-y-1.5 px-2.5 pb-2">
        <button onClick={() => p.onNew()} className="group relative flex h-[34px] w-full cursor-pointer items-center gap-2 overflow-hidden rounded-[9px] border border-line bg-raise px-2.5 text-[12.5px] font-medium text-ink shadow-e1 transition-all hover:border-iris/50 active:scale-[.99]">
          <span className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-iris/12 to-transparent transition-transform duration-700 group-hover:translate-x-full" />
          <span className="relative flex size-[18px] items-center justify-center rounded-[5px] bg-iris text-on-iris"><IconPlus size={11} /></span>
          <span className="relative">{product === "code" ? "New thread" : "New chat"}</span>
          <Kbd className="relative ml-auto">⌘N</Kbd>
        </button>
        <div className="relative">
          <IconSearch size={12} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-ink-4" />
          <Input placeholder="Search threads…" value={q} onChange={(e) => setQ(e.target.value)} className="h-[30px] bg-base pl-7" />
        </div>
      </div>

      {/* nav */}
      <nav className="space-y-px px-2.5 pb-2">
        {nav.filter((n) => n.id !== "workbench").map((n) => {
          const on = p.view === n.id;
          return (
            <button key={n.id} onClick={() => p.setView(n.id)} title={`${n.label} · ⌘${n.key}`}
              className={cn("group flex h-[29px] w-full cursor-pointer items-center gap-2.5 rounded-[7px] px-2 text-left text-[12.5px] transition-all duration-150", on ? "bg-raise font-medium text-ink shadow-e1 hairline" : "text-ink-2 hover:bg-raise/60 hover:text-ink")}>
              <n.icon size={14} className={cn("transition-colors", on ? "text-iris-soft" : "text-ink-4 group-hover:text-ink-3")} />
              {n.label}
              {n.id === "tasks" && openTodos > 0 && <span className="ml-auto rounded-full bg-hover px-1.5 font-mono text-[9px] text-ink-3">{openTodos}</span>}
              {n.id !== "tasks" && <span className="ml-auto font-mono text-[9px] text-ink-4 opacity-0 transition-opacity group-hover:opacity-100">⌘{n.key}</span>}
            </button>
          );
        })}
      </nav>

      {/* threads */}
      <div ref={listRef} role="listbox" aria-label="Threads" onKeyDown={onListKeyDown} tabIndex={-1}
        className="scroll-thin min-h-0 flex-1 overflow-y-auto px-2.5 pb-2 focus:outline-none">
        <div className="sticky top-0 z-10 -mx-2.5 flex items-center gap-1 border-b border-line-soft/70 bg-sunken/95 px-2.5 pt-1.5 pb-1 backdrop-blur">
          <span className="font-mono text-[9.5px] font-semibold tracking-[.14em] text-ink-4 uppercase">{showArchived ? "Archive" : grouping === "project" ? "Projects" : grouping === "date" ? "Recent" : "By agent"}</span>
          <span className="font-mono text-[9.5px] text-ink-4">{visible.length}</span>
          {attentionTotal > 0 && !showArchived && (
            <Tip label={`${attentionTotal} thread${attentionTotal > 1 ? "s" : ""} waiting on you or failed`}>
              <span className="flex items-center gap-1 rounded-full bg-amber-tint px-1.5 py-[1px] font-mono text-[9px] font-bold text-amber">{attentionTotal} need you</span>
            </Tip>
          )}
          <Menu align="right" width={216} className="ml-auto" trigger={() => <IconButton icon={IconMore} label="List options" size={22} />}>
            {(close) => <>
              <MenuLabel>group by</MenuLabel>
              {([["project", "Project"], ["date", "Recent"], ["agent", "Agent"]] as const).map(([v, l]) => <MenuItem key={v} label={l} active={grouping === v} onClick={() => { setGrouping(v); close(); }} />)}
              <MenuSep />
              <MenuItem icon={<IconFolder size={12} />} label="New project…" onClick={() => { setCreatingProject(true); close(); }} />
              <MenuItem icon={<IconArchive size={12} />} label={showArchived ? "Show active threads" : "Show archive"} onClick={() => { setShowArchived((v) => !v); close(); }} />
              <MenuSep />
              <div className="px-2 pb-1 font-mono text-[9px] leading-[1.5] text-ink-4">↑↓ move · ⏎ open · ⌫ archive<br />drag a thread onto a project</div>
            </>}
          </Menu>
        </div>

        {creatingProject && (
          <div className="mb-1.5 mt-1 flex gap-1">
            <Input autoFocus value={newProject} onChange={(e) => setNewProject(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") submitProject(); if (e.key === "Escape") setCreatingProject(false); }} placeholder="Project name…" className="h-[28px]" />
            <Button size="xs" variant="primary" onClick={submitProject}>Add</Button>
            <IconButton icon={IconX} label="Cancel" size={26} onClick={() => setCreatingProject(false)} />
          </div>
        )}

        {pinned.length > 0 && !showArchived && (
          <div className="mb-1">
            <div className="flex items-center gap-1.5 px-1 py-1 font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase"><IconPin size={9} />Pinned<span className="h-px flex-1 bg-line-soft" /></div>
            {pinned.map((s) => <Row key={`pin-${s.id}`} s={s} index={nextIndex()} />)}
          </div>
        )}

        {groupKeys.map((k) => {
          const items = visible.filter((s) => groupKey(s) === k && !(pinned.includes(s) && !showArchived));
          if (!items.length && grouping !== "project") return null;
          return (
            <div key={k} className="mb-0.5">
              <ProjectHeader k={k} items={items} />
              {!folds[k] && (
                <div className="ml-[7px] space-y-px border-l border-line-soft pl-1.5">
                  {items.map((s) => <Row key={s.id} s={s} index={nextIndex()} />)}
                  {items.length === 0 && <p className="px-2 py-1 text-[10.5px] text-ink-4">{grouping === "project" ? "No threads — drop one here" : "Empty"}</p>}
                </div>
              )}
            </div>
          );
        })}
        {visible.length === 0 && groupKeys.length === 0 && <p className="px-2 py-6 text-center text-[11px] text-ink-4">{showArchived ? "Archive is empty" : "No matching threads"}</p>}
      </div>

      {dragId && grouping === "project" && (
        <div className="animate-slide-up border-t border-iris/25 bg-iris-tint px-3 py-1.5 text-center font-mono text-[9.5px] text-iris-soft">
          drop on a project to move this thread
        </div>
      )}

      {/* bottom */}
      <div className="shrink-0 space-y-px border-t border-line-soft p-2">
        <button onClick={() => setTodoOpen(true)} className="flex h-[28px] w-full cursor-pointer items-center gap-2.5 rounded-[7px] px-2 text-left text-[12.5px] text-ink-2 transition-colors hover:bg-raise/60 hover:text-ink">
          <IconCheck size={14} className="text-ink-4" />Todo
          {openTodos > 0 && <span className="ml-auto rounded-full bg-iris px-1.5 font-mono text-[9px] font-bold text-on-iris">{openTodos}</span>}
        </button>
        <button onClick={() => p.setView("settings")} className={cn("flex h-[28px] w-full cursor-pointer items-center gap-2.5 rounded-[7px] px-2 text-left text-[12.5px] transition-colors", p.view === "settings" ? "bg-raise text-ink shadow-e1" : "text-ink-2 hover:bg-raise/60 hover:text-ink")}>
          <IconSettings size={14} className={p.view === "settings" ? "text-iris-soft" : "text-ink-4"} />Settings<Kbd className="ml-auto">⌘,</Kbd>
        </button>
        <div className="mt-1 flex items-center gap-2 rounded-[8px] border border-line-soft bg-raise px-2 py-1.5">
          <span className="flex size-[22px] items-center justify-center rounded-[6px] bg-gradient-to-br from-iris/30 to-cyan/20 font-mono text-[9.5px] font-bold text-ink">DV</span>
          <div className="min-w-0 flex-1 leading-tight"><div className="truncate text-[11.5px] font-medium text-ink">Dev Vale</div><div className="font-mono text-[9.5px] text-ink-4">Max · $184 / $400</div></div>
          <div className="h-[4px] w-9 overflow-hidden rounded-full bg-track"><div className="h-full w-[46%] bg-gradient-to-r from-iris to-cyan" /></div>
        </div>
      </div>

      {/* delete */}
      <Modal open={Boolean(deleteTarget)} onClose={() => setDeleteTarget(null)} title="Delete this thread?" sub="Removes the conversation from this device. This can't be undone." width={420}>
        <div className="p-4"><div className="rounded-[9px] border border-rose/20 bg-rose-tint/50 p-3 text-[12px] text-ink-2">{deleteTarget?.title}</div><div className="mt-4 flex justify-end gap-2"><Button variant="ghost" onClick={() => setDeleteTarget(null)}>Cancel</Button><Button variant="danger" icon={IconTrash} onClick={() => { if (deleteTarget) p.onDelete(deleteTarget.id); setDeleteTarget(null); }}>Delete</Button></div></div>
      </Modal>

      {/* rename */}
      <Modal open={Boolean(renameTarget)} onClose={() => setRenameTarget(null)} title="Rename thread" width={420}>
        <div className="space-y-3 p-4"><Input autoFocus value={renameTitle} onChange={(e) => setRenameTitle(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && renameTarget && renameTitle.trim()) { p.onRename(renameTarget.id, renameTitle.trim()); setRenameTarget(null); } }} /><div className="flex justify-end gap-2"><Button variant="ghost" onClick={() => setRenameTarget(null)}>Cancel</Button><Button variant="primary" onClick={() => { if (renameTarget && renameTitle.trim()) p.onRename(renameTarget.id, renameTitle.trim()); setRenameTarget(null); }}>Save</Button></div></div>
      </Modal>

      {/* project settings */}
      <ProjectSettings project={projectTarget} projects={p.projects} sessions={p.sessions} onClose={() => setProjectTarget(null)} onSave={(pr) => { p.onUpdateProject(pr); setProjectTarget(null); toast(`${pr.name} defaults saved — new threads inherit them`, "mint"); }} />
    </aside>
  );
}

/* ============================================================
   PROJECT SETTINGS — defaults inherited by threads
   ============================================================ */
function ProjectSettings({ project, projects, sessions, onClose, onSave }: { project: Project | null; projects: Project[]; sessions: Session[]; onClose: () => void; onSave: (p: Project) => void }) {
  const [draft, setDraft] = useState<Project | null>(project);
  /* resync the editable draft whenever a different project opens */
  const [prevProject, setPrevProject] = useState(project);
  if (prevProject !== project) {
    setPrevProject(project);
    setDraft(project);
  }
  if (!project || !draft) return null;
  const d = draft.defaults ?? {};
  const set = (patch: Partial<NonNullable<Project["defaults"]>>) => setDraft({ ...draft, defaults: { ...d, ...patch } });
  const threadCount = sessions.filter((s) => s.projectId === project.id).length;
  const agent = d.agentId ? agentById(d.agentId) : null;

  return (
    <Modal open onClose={onClose} title={`${project.name} · project settings`} sub={`Defaults inherited by ${threadCount} thread${threadCount === 1 ? "" : "s"}. A thread can still override any of them.`} width={620}>
      <div className="space-y-4 p-4">
        <div className="flex items-center gap-3 rounded-[11px] border border-line-soft bg-raise p-3">
          <span className="flex size-[34px] items-center justify-center rounded-[9px] font-mono text-[11px] font-bold" style={{ color: project.color, background: `color-mix(in srgb, ${project.color} 15%, transparent)`, boxShadow: `inset 0 0 0 1px ${project.color}44` }}>{project.name.slice(0, 2).toUpperCase()}</span>
          <div className="min-w-0 flex-1">
            <input value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} className="w-full bg-transparent font-display text-[14px] font-semibold text-ink focus:outline-none" />
            <div className="mt-0.5 font-mono text-[10px] text-ink-4">{draft.path ?? "workspace"} · {draft.kind}</div>
          </div>
          <div className="flex gap-1">{projects.length > 1 && ["#8f82ff", "#36c5b6", "#ec9acb", "#eab45a", "#6da8ff", "#ff8a4c"].map((c) => <button key={c} onClick={() => setDraft({ ...draft, color: c })} className={cn("size-[16px] cursor-pointer rounded-[4px] transition-transform hover:scale-110", draft.color === c && "ring-2 ring-ink/60")} style={{ background: c }} />)}</div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <div className="rounded-[11px] border border-line-soft bg-raise p-3">
            <div className="mb-2 flex items-center gap-1.5 font-mono text-[9.5px] tracking-[.14em] text-ink-4 uppercase"><IconLayers size={10} />default agent</div>
            <Select value={d.agentId ?? "claude-code"} onChange={(v) => { const a = agentById(v); set({ agentId: v, modelId: a.models[0].id }); }} options={AGENTS.filter((a) => a.status === "connected").map((a) => ({ value: a.id, label: a.name, hint: a.vendor }))} />
            {agent && <p className="mt-2 text-[11px] leading-[1.5] text-ink-3">New threads here start on <span className="text-ink-2">{agent.name}</span> · {agent.models.find((m) => m.id === d.modelId)?.label ?? agent.models[0].label}.</p>}
          </div>
          <div className="rounded-[11px] border border-line-soft bg-raise p-3">
            <div className="mb-2 flex items-center gap-1.5 font-mono text-[9.5px] tracking-[.14em] text-ink-4 uppercase"><IconZap size={10} />default model</div>
            <Select value={d.modelId ?? ""} onChange={(v) => set({ modelId: v })} options={(agent?.models ?? []).map((m) => ({ value: m.id, label: m.label, hint: m.ctx }))} />
            <div className="mt-2.5 mb-1.5 font-mono text-[9.5px] tracking-[.14em] text-ink-4 uppercase">permission mode</div>
            <Segmented value={(d.mode ?? "agent") as ModeId} onChange={(v) => set({ mode: v })} items={MODES.map((m) => ({ value: m.id, label: m.label, tone: m.tone }))} />
          </div>
        </div>

        <div className="rounded-[11px] border border-line-soft bg-raise p-3">
          <div className="mb-2 flex items-center gap-1.5 font-mono text-[9.5px] tracking-[.14em] text-ink-4 uppercase"><IconShield size={10} />project rules</div>
          <div className="space-y-1.5">
            {(d.rules ?? []).map((r, i) => (
              <div key={i} className="group flex items-center gap-2 rounded-[7px] border border-line-soft bg-well px-2 py-1.5">
                <span className="font-mono text-[9px] text-ink-4">{String(i + 1).padStart(2, "0")}</span>
                <input value={r} onChange={(e) => set({ rules: (d.rules ?? []).map((x, j) => (j === i ? e.target.value : x)) })} className="min-w-0 flex-1 bg-transparent text-[11.5px] text-ink-2 focus:outline-none" />
                <button onClick={() => set({ rules: (d.rules ?? []).filter((_, j) => j !== i) })} className="cursor-pointer text-ink-4 opacity-0 transition-opacity group-hover:opacity-100 hover:text-rose"><IconX size={11} /></button>
              </div>
            ))}
            <button onClick={() => set({ rules: [...(d.rules ?? []), ""] })} className="flex cursor-pointer items-center gap-1.5 rounded-[7px] border border-dashed border-line px-2 py-1.5 text-[11.5px] text-ink-4 transition-colors hover:border-iris/40 hover:text-iris-soft"><IconPlus size={11} />Add rule</button>
          </div>
        </div>

        <div className="flex items-center gap-2 rounded-[11px] border border-line-soft bg-raise p-3">
          <span className="font-mono text-[9.5px] tracking-[.14em] text-ink-4 uppercase">budget</span>
          <Input value={d.budget ?? ""} onChange={(e) => set({ budget: e.target.value })} placeholder="$25 / session" className="h-[28px] w-[150px] font-mono text-[11.5px]" />
          <span className="text-[11px] text-ink-3">Stops runs that exceed it.</span>
          <div className="ml-auto flex items-center gap-2"><span className="text-[11.5px] text-ink-3">Inherit to existing</span><Toggle checked={false} onChange={() => { /* opt-in bulk apply */ }} size="sm" /></div>
        </div>

        <div className="flex items-center gap-2 border-t border-line-soft pt-3">
          <Badge tone="neutral" mono>{threadCount} threads</Badge>
          <Badge tone="iris" mono>inherits: agent · model · mode · rules · budget</Badge>
          <div className="ml-auto flex gap-2"><Button variant="ghost" onClick={onClose}>Cancel</Button><Button variant="primary" icon={IconCheck} onClick={() => onSave(draft)}>Save defaults</Button></div>
        </div>
      </div>
    </Modal>
  );
}

export { ASSISTANT };
