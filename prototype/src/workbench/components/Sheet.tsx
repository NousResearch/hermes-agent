"use client";

import { useEffect, useRef } from "react";
import { IconChevronDown, IconX } from "./Icons";
import { Kbd } from "./ui";

/**
 * A panel that slides over the chat instead of replacing it.
 * Chat stays mounted underneath — drafts, scroll and streaming are preserved.
 * Close with the back button, the ✕, Esc, or by clicking the same nav item again.
 */
export function Sheet({ title, eyebrow, onClose, children, streaming }: { title: string; eyebrow?: string; onClose: () => void; children: React.ReactNode; streaming?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    ref.current?.focus({ preventScroll: true });
    return () => { prev?.focus?.({ preventScroll: true }); };
  }, []);

  return (
    <div className="absolute inset-0 z-30 flex">
      {/* scrim — clicking the sliver of chat closes the sheet */}
      <button aria-label="Back to chat" onClick={onClose} className="absolute inset-0 animate-fade cursor-default bg-void/40 backdrop-blur-[2px]" />
      <div ref={ref} tabIndex={-1} role="dialog" aria-label={title}
        className="relative ml-auto flex h-full w-full animate-sheet flex-col overflow-hidden border-l border-line-soft bg-base shadow-e4 outline-none lg:w-[calc(100%-24px)]">
        <header className="flex h-[44px] shrink-0 items-center gap-2 border-b border-line-soft bg-base/90 px-2.5 backdrop-blur-xl">
          <button onClick={onClose} className="group flex h-[28px] cursor-pointer items-center gap-1 rounded-[7px] pr-2.5 pl-1.5 text-[12px] text-ink-3 transition-colors hover:bg-hover hover:text-ink">
            <IconChevronDown size={13} className="rotate-90 transition-transform group-hover:-translate-x-0.5" />
            Chat
            {streaming && <i className="ml-0.5 size-[5px] animate-breathe rounded-full bg-cyan" title="Agent is still working" />}
          </button>
          <span className="h-4 w-px bg-line" />
          {eyebrow && <span className="font-mono text-[9.5px] tracking-[.16em] text-ink-4 uppercase">{eyebrow}</span>}
          <h2 className="font-display text-[13.5px] font-semibold tracking-[-.01em] text-ink">{title}</h2>
          <div className="ml-auto flex items-center gap-2">
            <span className="hidden items-center gap-1 font-mono text-[10px] text-ink-4 sm:flex"><Kbd>esc</Kbd> to close</span>
            <button onClick={onClose} className="flex size-[28px] cursor-pointer items-center justify-center rounded-[7px] text-ink-3 transition-colors hover:bg-hover hover:text-ink" aria-label="Close panel"><IconX size={14} /></button>
          </div>
        </header>
        <div className="flex min-h-0 flex-1 flex-col">{children}</div>
      </div>
    </div>
  );
}
