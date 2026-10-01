"use client";

import { useEffect, useMemo, useState } from "react";
import { cn } from "../utils/cn";
import { SHORTCUTS, SHORTCUT_GROUPS } from "../lib/shortcuts";
import { IconCommand, IconSearch, IconX } from "./Icons";
import { IconButton, Kbd } from "./ui";

export function ShortcutHelp({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [q, setQ] = useState("");
  /* clear the filter each time the overlay opens */
  const [prevOpen, setPrevOpen] = useState(open);
  if (prevOpen !== open) {
    setPrevOpen(open);
    if (open) setQ("");
  }
  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, onClose]);

  const groups = useMemo(
    () => SHORTCUT_GROUPS.map((g) => ({ g, items: SHORTCUTS.filter((s) => s.group === g && (s.label + s.keys.join("") + (s.note ?? "")).toLowerCase().includes(q.toLowerCase())) })).filter((x) => x.items.length),
    [q],
  );

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-[130] flex items-start justify-center pt-[8vh]">
      <div className="absolute inset-0 animate-fade bg-black/60 backdrop-blur-[3px]" onClick={onClose} />
      <div className="relative w-[min(760px,94vw)] animate-pop overflow-hidden rounded-[16px] border border-line-strong bg-base shadow-e4">
        <div className="flex items-center gap-3 border-b border-line-soft bg-raise/50 px-4 py-3">
          <span className="flex size-[28px] items-center justify-center rounded-[8px] border border-line bg-well text-iris-soft"><IconCommand size={14} /></span>
          <div className="min-w-0 flex-1">
            <h2 className="font-display text-[15px] font-semibold tracking-[-.01em] text-ink">Keyboard shortcuts</h2>
            <p className="text-[11.5px] text-ink-3">Everything is reachable without the mouse. Defaults follow VS Code and Cursor.</p>
          </div>
          <div className="relative hidden w-[190px] sm:block">
            <IconSearch size={12} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-ink-4" />
            <input autoFocus value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter…" className="h-[30px] w-full rounded-[7px] border border-line bg-sunken pr-2 pl-7 text-[12px] text-ink placeholder:text-ink-4 focus:border-iris/50 focus:ring-focus focus:outline-none" />
          </div>
          <IconButton icon={IconX} label="Close" onClick={onClose} size={28} />
        </div>

        <div className="scroll-thin grid max-h-[64vh] grid-cols-1 gap-x-8 gap-y-1 overflow-y-auto px-5 py-4 md:grid-cols-2">
          {groups.map(({ g, items }) => (
            <div key={g} className={cn("mb-3", g === "Global" && "md:col-span-2")}>
              <div className="mb-1.5 flex items-center gap-2">
                <span className="font-mono text-[9px] font-semibold tracking-[.18em] text-ink-4 uppercase">{g}</span>
                <span className="h-px flex-1 bg-line-soft" />
              </div>
              <div className={cn("space-y-px", g === "Global" && "grid grid-cols-1 gap-x-8 md:grid-cols-2")}>
                {items.map((s) => (
                  <div key={s.id} className="group flex items-center gap-3 rounded-[7px] px-2 py-[7px] transition-colors hover:bg-hover/70">
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[12.5px] text-ink-2 group-hover:text-ink">{s.label}</div>
                      {s.note && <div className="truncate text-[10.5px] text-ink-4">{s.note}</div>}
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                      {s.keys.map((k, i) => <Kbd key={i} className="h-[20px] min-w-[20px] px-1.5 text-[10px]">{k}</Kbd>)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
          {groups.length === 0 && <div className="col-span-2 py-12 text-center text-[12.5px] text-ink-4">No shortcut matches “{q}”</div>}
        </div>

        <div className="flex items-center gap-3 border-t border-line-soft bg-raise/40 px-4 py-2.5 font-mono text-[10px] text-ink-4">
          <span className="flex items-center gap-1.5"><Kbd>⌘</Kbd><Kbd>/</Kbd> toggle</span>
          <span className="flex items-center gap-1.5"><Kbd>esc</Kbd> close</span>
          <span className="ml-auto">Rebind in Settings → Keyboard</span>
        </div>
      </div>
    </div>
  );
}
