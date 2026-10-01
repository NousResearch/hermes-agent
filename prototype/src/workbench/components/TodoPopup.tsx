"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "../utils/cn";
import { useApp, type Todo } from "../lib/app";
import { IconCheck, IconPlus, IconSpark, IconX, IconChevronDown } from "./Icons";
import { Bar, IconButton } from "./ui";

export function TodoPopup() {
  const { todos, setTodos, todoOpen, setTodoOpen } = useApp();
  const [text, setText] = useState("");
  const [min, setMin] = useState(false);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const done = todos.filter((t) => t.state === "done").length;

  useEffect(() => {
    const h = (e: KeyboardEvent) => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "t") { e.preventDefault(); setTodoOpen(!todoOpen); } };
    window.addEventListener("keydown", h); return () => window.removeEventListener("keydown", h);
  }, [todoOpen, setTodoOpen]);

  const onDrag = (e: React.MouseEvent) => {
    drag.current = { x: e.clientX, y: e.clientY, px: pos.x, py: pos.y };
    const move = (ev: MouseEvent) => drag.current && setPos({ x: drag.current.px + ev.clientX - drag.current.x, y: drag.current.py + ev.clientY - drag.current.y });
    const up = () => { drag.current = null; window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
    window.addEventListener("mousemove", move); window.addEventListener("mouseup", up);
  };
  const cycle = (t: Todo) => setTodos((ts) => ts.map((x) => x.id === t.id ? { ...x, state: x.state === "todo" ? "doing" : x.state === "doing" ? "done" : "todo" } : x));
  const add = () => { if (!text.trim()) return; setTodos((t) => [{ id: `u${Date.now()}`, text: text.trim(), state: "todo", by: "you" }, ...t]); setText(""); };

  if (!todoOpen) return null;
  return (
    <div style={{ transform: `translate(${pos.x}px, ${pos.y}px)` }} className="fixed right-[92px] bottom-10 z-[90] w-[320px] animate-slide-up overflow-hidden rounded-[14px] border border-line-strong bg-overlay shadow-e4">
      <div onMouseDown={onDrag} className="flex cursor-grab items-center gap-2 border-b border-line-soft bg-raise/70 px-3 py-2 select-none active:cursor-grabbing">
        <IconCheck size={13} className="text-iris-soft" /><span className="font-display text-[13px] font-semibold tracking-[-.01em] text-ink">Todo</span>
        <span className="font-mono text-[10px] text-ink-4">{done}/{todos.length}</span>
        <span className="ml-1 flex items-center gap-1 rounded-full bg-cyan-tint px-1.5 py-[1px] font-mono text-[9px] text-cyan"><i className="size-[4px] animate-breathe rounded-full bg-cyan" />agent-synced</span>
        <div className="ml-auto flex items-center"><IconButton icon={IconChevronDown} label={min ? "Expand" : "Minimise"} size={24} className={cn(min && "rotate-180")} onClick={() => setMin((m) => !m)} /><IconButton icon={IconX} label="Close" size={24} onClick={() => setTodoOpen(false)} /></div>
      </div>
      {!min && (<>
        <div className="px-3 pt-2.5"><Bar value={(done / Math.max(todos.length, 1)) * 100} tone="mint" height={3} /></div>
        <div className="scroll-thin max-h-[300px] space-y-[2px] overflow-y-auto p-2">
          {todos.map((t) => (
            <div key={t.id} className="group flex items-start gap-2 rounded-[7px] px-1.5 py-[6px] transition-colors hover:bg-hover/60">
              <button onClick={() => cycle(t)} className={cn("mt-[2px] flex size-[15px] shrink-0 cursor-pointer items-center justify-center rounded-[4px] border transition-all", t.state === "done" ? "border-mint bg-mint text-on-iris" : t.state === "doing" ? "border-cyan bg-cyan-tint" : "border-line-strong hover:border-ink-4")}>
                {t.state === "done" && <IconCheck size={9} />}{t.state === "doing" && <i className="size-[6px] animate-breathe rounded-[2px] bg-cyan" />}
              </button>
              <span className={cn("min-w-0 flex-1 text-[12px] leading-[1.45]", t.state === "done" ? "text-ink-4 line-through" : "text-ink-2")}>{t.text}</span>
              <span className={cn("shrink-0 rounded-[3px] px-1 font-mono text-[8.5px]", t.by === "agent" ? "bg-iris-tint text-iris-soft" : "bg-hover text-ink-4")}>{t.by === "agent" ? <IconSpark size={8} /> : "you"}</span>
              <button onClick={() => setTodos((ts) => ts.filter((x) => x.id !== t.id))} className="cursor-pointer text-ink-4 opacity-0 transition-opacity group-hover:opacity-100 hover:text-rose"><IconX size={10} /></button>
            </div>
          ))}
        </div>
        <div className="flex items-center gap-2 border-t border-line-soft px-2.5 py-2">
          <input value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && add()} placeholder="Add a todo… (⏎)" className="h-[28px] min-w-0 flex-1 rounded-[6px] bg-sunken px-2 text-[12px] text-ink placeholder:text-ink-4 focus:outline-none" />
          <button onClick={add} className="flex size-[28px] cursor-pointer items-center justify-center rounded-[6px] bg-iris text-on-iris"><IconPlus size={13} /></button>
        </div>
      </>)}
    </div>
  );
}
