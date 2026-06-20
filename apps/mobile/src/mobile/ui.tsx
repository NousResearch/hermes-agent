/**
 * ui.tsx — minimal shared primitives for the Phase 0 connect/login screens,
 * styled with the vendored design-system tokens (bg-card, text-foreground,
 * border-border, bg-primary…) so they already match the desktop look.
 */

import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode } from 'react'

/** Full-screen, safe-area-aware, vertically-centered container. */
export function Screen({ children }: { children: ReactNode }) {
  return (
    <div
      className="flex min-h-full flex-col items-center justify-center px-6"
      style={{
        paddingTop: 'calc(env(safe-area-inset-top) + 1.5rem)',
        paddingBottom: 'calc(env(safe-area-inset-bottom) + 1.5rem)',
      }}
    >
      <div className="flex w-full max-w-sm flex-col gap-5">{children}</div>
    </div>
  )
}

export function Brand({ subtitle }: { subtitle: string }) {
  return (
    <div className="flex flex-col items-center gap-1 text-center">
      <div
        className="text-3xl font-bold tracking-tight text-primary"
        style={{ fontFamily: "'Collapse', var(--dt-font-sans)" }}
      >
        Hermes
      </div>
      <div className="text-sm text-muted-foreground">{subtitle}</div>
    </div>
  )
}

export function Card({ children }: { children: ReactNode }) {
  return (
    <div className="flex flex-col gap-4 rounded-2xl border border-border bg-card/60 p-5 backdrop-blur">
      {children}
    </div>
  )
}

export function Field({
  label,
  ...props
}: { label: string } & InputHTMLAttributes<HTMLInputElement>) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
      <input
        className="w-full rounded-xl border border-input bg-background/40 px-3.5 py-3 text-base text-foreground outline-none transition focus:border-ring focus:ring-2 focus:ring-ring/30 placeholder:text-muted-foreground/60"
        {...props}
      />
    </label>
  )
}

export function Button({
  children,
  variant = 'primary',
  busy,
  ...props
}: {
  variant?: 'primary' | 'ghost'
  busy?: boolean
} & ButtonHTMLAttributes<HTMLButtonElement>) {
  const base =
    'w-full rounded-xl px-4 py-3 text-base font-semibold transition active:scale-[0.99] disabled:opacity-50 disabled:active:scale-100'
  const styles =
    variant === 'primary'
      ? 'bg-primary text-primary-foreground hover:opacity-90'
      : 'border border-border bg-transparent text-foreground hover:bg-accent/40'
  return (
    <button className={`${base} ${styles}`} disabled={busy || props.disabled} {...props}>
      {busy ? 'Working…' : children}
    </button>
  )
}

export function ErrorNote({ children }: { children: ReactNode }) {
  if (!children) return null
  return (
    <div className="rounded-xl border border-destructive/40 bg-destructive/10 px-3.5 py-2.5 text-sm text-destructive-foreground">
      {children}
    </div>
  )
}
