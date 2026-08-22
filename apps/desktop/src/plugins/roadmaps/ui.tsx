/**
 * Roadmaps plugin — shared local UI pieces.
 *
 * Small presentational components reused across views. Machine-state labels
 * and tones come from config (English on purpose: the UI never rewrites the
 * state machine, only labels it).
 */

import { StatusDot } from '@hermes/plugin-sdk'
import type { ReactNode } from 'react'

import { NODE_STATE_LABEL, NODE_TONE } from './config'

export function ProgressBar({ value }: { value: number | null | undefined }) {
  const pct = Math.max(0, Math.min(100, Number.isFinite(value) ? (value ?? 0) : 0))

  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="h-1 w-14 overflow-hidden rounded-full bg-(--ui-stroke-secondary)">
        <span className="h-full rounded-full bg-primary transition-all" style={{ width: `${pct}%` }} />
      </span>
      <span className="text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">{pct}%</span>
    </span>
  )
}

export function SectionTitle({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2 px-0.5 text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
      <span className="truncate">{children}</span>
      {right ?? null}
    </div>
  )
}

/** State tag: StatusDot + label, no box — tone carries the semantics. */
export function NodeStateTag({ state }: { state: string }) {
  return (
    <span className="inline-flex items-center gap-1 text-[0.625rem] text-(--ui-text-tertiary)">
      <StatusDot tone={NODE_TONE[state] ?? 'muted'} />
      {NODE_STATE_LABEL[state] ?? state}
    </span>
  )
}
