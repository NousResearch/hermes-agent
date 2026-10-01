"use client";

import React, { useEffect, useRef, useState } from "react";
import { cn } from "../utils/cn";
import { IconChevronDown, IconX } from "./Icons";

/* ================================ BUTTON ================================ */
type Variant = "primary" | "secondary" | "ghost" | "outline" | "danger" | "success";
type Size = "xs" | "sm" | "md" | "lg";
const variantMap: Record<Variant, string> = {
  primary: "bg-iris text-on-iris shadow-e1 hover:brightness-110 active:brightness-95",
  secondary: "bg-raise text-ink border border-line hover:bg-hover hover:border-line-strong",
  ghost: "text-ink-2 hover:bg-hover hover:text-ink",
  outline: "border border-line text-ink-2 bg-transparent hover:border-line-strong hover:text-ink hover:bg-hover/60",
  danger: "bg-rose-tint text-rose border border-rose/25 hover:brightness-110",
  success: "bg-mint-tint text-mint border border-mint/25 hover:brightness-110",
};
const sizeMap: Record<Size, string> = {
  xs: "h-[24px] px-2 text-[11.5px] gap-1 rounded-[6px]",
  sm: "h-[30px] px-3 text-[12.5px] gap-1.5 rounded-[7px]",
  md: "h-[34px] px-3.5 text-[13px] gap-2 rounded-[8px]",
  lg: "h-[40px] px-5 text-[14px] gap-2 rounded-[10px]",
};
export function Button({ variant = "secondary", size = "sm", icon: Icon, iconRight: IconRight, className, children, ...rest }:
  React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: Size; icon?: React.ComponentType<{ size?: number; className?: string }>; iconRight?: React.ComponentType<{ size?: number; className?: string }> }) {
  return (
    <button {...rest} className={cn("inline-flex shrink-0 cursor-pointer items-center justify-center font-medium whitespace-nowrap transition-all duration-150 select-none focus-visible:ring-focus disabled:pointer-events-none disabled:opacity-40", variantMap[variant], sizeMap[size], className)}>
      {Icon && <Icon size={size === "xs" ? 12 : size === "lg" ? 16 : 13} />}
      {children}
      {IconRight && <IconRight size={size === "xs" ? 11 : 13} />}
    </button>
  );
}

export function IconButton({ icon: Icon, label, active, size = 30, className, ...rest }:
  React.ButtonHTMLAttributes<HTMLButtonElement> & { icon: React.ComponentType<{ size?: number }>; label?: string; active?: boolean; size?: number }) {
  return (
    <button {...rest} title={label} aria-label={label} style={{ width: size, height: size }}
      className={cn("inline-flex shrink-0 cursor-pointer items-center justify-center rounded-[7px] text-ink-3 transition-all duration-150 hover:bg-hover hover:text-ink focus-visible:ring-focus active:scale-[.94]", active && "bg-iris-tint text-iris-soft", className)}>
      <Icon size={Math.round(size * 0.5)} />
    </button>
  );
}

/* ================================ BADGE ================================ */
export type Tone = "neutral" | "iris" | "cyan" | "mint" | "amber" | "rose" | "sky" | "plum";
export const toneText: Record<Tone, string> = { neutral: "text-ink-3", iris: "text-iris-soft", cyan: "text-cyan", mint: "text-mint", amber: "text-amber", rose: "text-rose", sky: "text-sky", plum: "text-plum" };
export const toneBg: Record<Tone, string> = { neutral: "bg-hover text-ink-2", iris: "bg-iris-tint text-iris-soft", cyan: "bg-cyan-tint text-cyan", mint: "bg-mint-tint text-mint", amber: "bg-amber-tint text-amber", rose: "bg-rose-tint text-rose", sky: "bg-sky-tint text-sky", plum: "bg-plum-tint text-plum" };
export const toneDot: Record<Tone, string> = { neutral: "bg-line-strong", iris: "bg-iris", cyan: "bg-cyan", mint: "bg-mint", amber: "bg-amber", rose: "bg-rose", sky: "bg-sky", plum: "bg-plum" };
export const toneHex = (t: Tone) => `var(--color-${t === "neutral" ? "line-strong" : t})`;

export function Badge({ tone = "neutral", children, className, dot, mono }: { tone?: Tone; children: React.ReactNode; className?: string; dot?: boolean; mono?: boolean }) {
  return (
    <span className={cn("inline-flex shrink-0 items-center gap-1.5 rounded-full px-2 py-[2px] text-[10.5px] font-medium", mono && "font-mono text-[10px]", toneBg[tone], className)}>
      {dot && <i className="size-[5px] rounded-full bg-current" />}
      {children}
    </span>
  );
}
export function Kbd({ children, className }: { children: React.ReactNode; className?: string }) {
  return <kbd className={cn("inline-flex h-[18px] min-w-[18px] items-center justify-center rounded-[4px] border border-line-strong/80 bg-well px-1 font-mono text-[9.5px] leading-none font-medium text-ink-3", className)}>{children}</kbd>;
}

/* ================================ SURFACES ================================ */
export function Panel({ className, children, ...rest }: React.HTMLAttributes<HTMLDivElement>) {
  return <div {...rest} className={cn("rounded-[10px] border border-line-soft bg-raise", className)}>{children}</div>;
}
export function SectionLabel({ children, className, right }: { children: React.ReactNode; className?: string; right?: React.ReactNode }) {
  return (
    <div className={cn("flex items-center justify-between gap-2 px-2.5 pt-2.5 pb-1.5", className)}>
      <span className="font-mono text-[9.5px] font-semibold tracking-[.14em] text-ink-4 uppercase">{children}</span>
      {right}
    </div>
  );
}
export function Divider({ className }: { className?: string }) { return <div className={cn("h-px w-full bg-line-soft", className)} />; }

/* ================================ FORMS ================================ */
export function Input({ className, ...rest }: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...rest} className={cn("h-[32px] w-full rounded-[7px] border border-line bg-sunken px-2.5 text-[12.5px] text-ink placeholder:text-ink-4 transition-shadow focus:border-iris/50 focus:ring-focus focus:outline-none", className)} />;
}

/* Generic popover menu */
export function Menu({ trigger, children, align = "left", width = 220, className, open: controlled, onOpenChange }: { trigger: (open: boolean) => React.ReactNode; children: (close: () => void) => React.ReactNode; align?: "left" | "right"; width?: number; className?: string; open?: boolean; onOpenChange?: (v: boolean) => void }) {
  const [inner, setInner] = useState(false);
  const open = controlled ?? inner;
  const setOpen = (v: boolean) => { setInner(v); onOpenChange?.(v); };
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    const k = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("mousedown", h); window.addEventListener("keydown", k);
    return () => { window.removeEventListener("mousedown", h); window.removeEventListener("keydown", k); };
  }, [open]);
  return (
    <div ref={ref} className={cn("relative", className)}>
      <div onClick={() => setOpen(!open)}>{trigger(open)}</div>
      {open && (
        <div style={{ width }} className={cn("absolute z-[60] mt-1.5 origin-top animate-slide-down overflow-hidden rounded-[10px] border border-line-strong bg-overlay p-1 shadow-e4", align === "right" ? "right-0" : "left-0")}>
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  );
}
export function MenuItem({ icon, label, hint, active, onClick, danger }: { icon?: React.ReactNode; label: React.ReactNode; hint?: React.ReactNode; active?: boolean; onClick?: () => void; danger?: boolean }) {
  return (
    <button onClick={onClick} className={cn("flex w-full cursor-pointer items-center gap-2.5 rounded-[7px] px-2 py-[7px] text-left text-[12.5px] transition-colors", active ? "bg-iris-tint text-iris-soft" : danger ? "text-rose hover:bg-rose-tint" : "text-ink-2 hover:bg-hover hover:text-ink")}>
      {icon && <span className="flex w-4 shrink-0 items-center justify-center">{icon}</span>}
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {hint && <span className="shrink-0 font-mono text-[10px] text-ink-4">{hint}</span>}
    </button>
  );
}
export function MenuLabel({ children }: { children: React.ReactNode }) {
  return <div className="px-2 pt-2 pb-1 font-mono text-[9px] tracking-[.14em] text-ink-4 uppercase">{children}</div>;
}
export function MenuSep() { return <div className="my-1 h-px bg-line-soft" />; }

export function Select({ value, onChange, options, className, size = "sm", align = "left", bare }: { value: string; onChange: (v: string) => void; options: { value: string; label: string; hint?: string }[]; className?: string; size?: "xs" | "sm"; align?: "left" | "right"; bare?: boolean }) {
  const current = options.find((o) => o.value === value);
  return (
    <Menu align={align} className={className} width={200} trigger={(open) => (
      <button className={cn("inline-flex w-full cursor-pointer items-center justify-between gap-2 rounded-[7px] text-ink-2 transition-colors hover:text-ink", !bare && "border border-line bg-raise hover:border-line-strong", size === "xs" ? "h-[24px] px-1.5 text-[11.5px]" : "h-[30px] px-2 text-[12.5px]")}>
        <span className="truncate font-medium">{current?.label ?? value}</span>
        <IconChevronDown size={12} className={cn("shrink-0 opacity-60 transition-transform", open && "rotate-180")} />
      </button>
    )}>
      {(close) => options.map((o) => <MenuItem key={o.value} label={o.label} hint={o.hint} active={o.value === value} onClick={() => { onChange(o.value); close(); }} />)}
    </Menu>
  );
}

export function Toggle({ checked, onChange, size = "md" }: { checked: boolean; onChange: (v: boolean) => void; size?: "sm" | "md" }) {
  const w = size === "sm" ? 30 : 36, h = size === "sm" ? 17 : 20, k = h - 4;
  return (
    <button role="switch" aria-checked={checked} onClick={() => onChange(!checked)} style={{ width: w, height: h }} className={cn("relative shrink-0 cursor-pointer rounded-full transition-colors duration-200", checked ? "bg-iris" : "bg-line-strong")}>
      <span style={{ width: k, height: k, transform: `translateX(${checked ? w - k - 2 : 2}px)` }} className="absolute top-[2px] left-0 rounded-full bg-white shadow-e1 transition-transform duration-200" />
    </button>
  );
}

export function Segmented<T extends string>({ value, onChange, items, className, size = "sm" }: { value: T; onChange: (v: T) => void; items: { value: T; label: string; icon?: React.ComponentType<{ size?: number }>; tone?: Tone }[]; className?: string; size?: "sm" | "md" }) {
  return (
    <div className={cn("inline-flex items-center gap-0.5 rounded-[8px] border border-line bg-sunken p-0.5", className)}>
      {items.map((it) => {
        const on = it.value === value;
        return (
          <button key={it.value} onClick={() => onChange(it.value)} className={cn("inline-flex cursor-pointer items-center gap-1.5 rounded-[6px] font-medium transition-all duration-150", size === "md" ? "px-3 py-[6px] text-[12.5px]" : "px-2.5 py-[4px] text-[12px]", on ? cn("bg-raise text-ink shadow-e1", it.tone && toneText[it.tone]) : "text-ink-3 hover:text-ink-2")}>
            {it.icon && <it.icon size={12} />}
            {it.label}
          </button>
        );
      })}
    </div>
  );
}

export function Tabs<T extends string>({ value, onChange, items, className }: { value: T; onChange: (v: T) => void; items: { value: T; label: string; count?: number }[]; className?: string }) {
  return (
    <div className={cn("flex items-center gap-0.5", className)}>
      {items.map((it) => {
        const on = it.value === value;
        return (
          <button key={it.value} onClick={() => onChange(it.value)} className={cn("relative cursor-pointer rounded-[6px] px-2.5 py-[6px] text-[12.5px] font-medium transition-colors", on ? "text-ink" : "text-ink-3 hover:text-ink-2")}>
            <span className="flex items-center gap-1.5">
              {it.label}
              {it.count !== undefined && <span className={cn("rounded-full px-1.5 font-mono text-[9.5px]", on ? "bg-iris-tint text-iris-soft" : "bg-hover text-ink-4")}>{it.count}</span>}
            </span>
            {on && <span className="absolute inset-x-1.5 -bottom-[5px] h-[2px] rounded-full bg-iris" />}
          </button>
        );
      })}
    </div>
  );
}

/* ================================ PROGRESS ================================ */
export function Bar({ value, tone = "iris", className, indeterminate, height = 4 }: { value?: number; tone?: Tone; className?: string; indeterminate?: boolean; height?: number }) {
  return (
    <div style={{ height }} className={cn("w-full overflow-hidden rounded-full bg-track", className)}>
      {indeterminate ? <div className="relative h-full w-1/3"><div className="h-full w-full animate-sweep rounded-full" style={{ background: toneHex(tone) }} /></div>
        : <div className="h-full rounded-full transition-[width] duration-500" style={{ width: `${value ?? 0}%`, background: toneHex(tone) }} />}
    </div>
  );
}
export function Ring({ value, size = 56, stroke = 5, tone = "var(--color-iris)", children }: { value: number; size?: number; stroke?: number; tone?: string; children?: React.ReactNode }) {
  const r = (size - stroke) / 2, c = 2 * Math.PI * r;
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--color-track)" strokeWidth={stroke} />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={tone} strokeWidth={stroke} strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c - (c * Math.min(value, 100)) / 100} style={{ transition: "stroke-dashoffset .7s var(--ease-out-quint)" }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">{children}</div>
    </div>
  );
}
export function Sparkline({ data, color = "var(--color-iris)", w = 64, h = 20 }: { data: number[]; color?: string; w?: number; h?: number }) {
  const max = Math.max(...data, 1), min = Math.min(...data, 0);
  const pts = data.map((d, i) => `${((i / (data.length - 1)) * w).toFixed(1)},${(h - ((d - min) / (max - min || 1)) * (h - 2) - 1).toFixed(1)}`);
  return (
    <svg width={w} height={h} className="overflow-visible">
      <polyline points={`0,${h} ${pts.join(" ")} ${w},${h}`} fill={color} opacity="0.12" stroke="none" />
      <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth="1.4" strokeLinejoin="round" />
    </svg>
  );
}
export function Stat({ label, value, sub, tone = "neutral", chart }: { label: string; value: React.ReactNode; sub?: React.ReactNode; tone?: Tone; chart?: number[] }) {
  return (
    <div className="flex flex-col gap-1 rounded-[10px] border border-line-soft bg-raise p-3">
      <span className="font-mono text-[9.5px] font-semibold tracking-[.12em] text-ink-4 uppercase">{label}</span>
      <div className="flex items-end justify-between gap-2">
        <span className={cn("text-[20px] leading-none font-semibold tracking-tight", tone === "neutral" ? "text-ink" : toneText[tone])}>{value}</span>
        {chart && <Sparkline data={chart} w={54} h={18} />}
      </div>
      {sub && <span className="text-[11px] text-ink-3">{sub}</span>}
    </div>
  );
}

/* ================================ MISC ================================ */
export function Tip({ label, children, side = "top" }: { label: string; children: React.ReactNode; side?: "top" | "bottom" }) {
  return (
    <span className="group/tip relative inline-flex">
      {children}
      <span className={cn("pointer-events-none absolute left-1/2 z-[70] -translate-x-1/2 scale-95 rounded-[6px] border border-line-strong bg-overlay px-2 py-1 text-[11px] whitespace-nowrap text-ink-2 opacity-0 shadow-e3 transition-all duration-150 group-hover/tip:scale-100 group-hover/tip:opacity-100", side === "top" ? "bottom-[calc(100%+7px)]" : "top-[calc(100%+7px)]")}>{label}</span>
    </span>
  );
}
export function EmptyState({ icon: Icon, title, body, action }: { icon: React.ComponentType<{ size?: number; className?: string }>; title: string; body: string; action?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-8 py-14 text-center">
      <div className="flex size-11 items-center justify-center rounded-[12px] border border-line bg-raise text-ink-3"><Icon size={19} /></div>
      <div className="space-y-1"><p className="text-[13.5px] font-semibold text-ink">{title}</p><p className="max-w-[280px] text-[12px] leading-relaxed text-ink-3">{body}</p></div>
      {action}
    </div>
  );
}
export function Modal({ open, onClose, title, sub, width = 620, children }: { open: boolean; onClose: () => void; title: string; sub?: string; width?: number; children: React.ReactNode }) {
  useEffect(() => { if (!open) return; const h = (e: KeyboardEvent) => e.key === "Escape" && onClose(); window.addEventListener("keydown", h); return () => window.removeEventListener("keydown", h); }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[10vh]">
      <div className="absolute inset-0 animate-fade bg-black/55 backdrop-blur-[3px]" onClick={onClose} />
      <div role="dialog" aria-modal="true" aria-label={title} style={{ width, maxWidth: "94vw" }} className="relative max-h-[78vh] animate-pop overflow-hidden rounded-[14px] border border-line-strong bg-base shadow-e4">
        <div className="flex items-start justify-between gap-4 border-b border-line-soft bg-raise/60 px-4 py-3">
          <div><h3 className="font-display text-[15px] font-semibold tracking-[-.015em] text-ink">{title}</h3>{sub && <p className="mt-0.5 text-[12px] leading-[1.5] text-ink-3">{sub}</p>}</div>
          <IconButton icon={IconX} label="Close" onClick={onClose} size={28} />
        </div>
        <div className="scroll-thin max-h-[66vh] overflow-y-auto">{children}</div>
      </div>
    </div>
  );
}
export function Toasts({ items }: { items: { id: number; msg: string; tone: string }[] }) {
  return (
    <div role="status" aria-live="polite" className="pointer-events-none fixed bottom-10 left-1/2 z-[150] flex -translate-x-1/2 flex-col items-center gap-2">
      {items.map((t) => (
        <div key={t.id} className={cn("animate-slide-up rounded-[9px] border bg-overlay px-3 py-2 text-[12px] text-ink shadow-e3", t.tone === "mint" ? "border-mint/30" : t.tone === "rose" ? "border-rose/30" : t.tone === "amber" ? "border-amber/30" : "border-iris/30")}>
          <span className={cn("mr-2 inline-block size-[6px] rounded-full", toneDot[(t.tone as Tone) || "iris"])} />{t.msg}
        </div>
      ))}
    </div>
  );
}
