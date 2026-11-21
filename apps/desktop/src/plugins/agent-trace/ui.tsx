/**
 * Shared AgentTrace UI atoms — formatters, status tones, gauge and code
 * blocks. Everything paints with theme variables / semantic tokens; no
 * hardcoded colors (the plugin contract).
 */

import { Badge, cn, Codicon, compactNumber, CopyButton, fmtDayTime, Tip } from '@hermes/plugin-sdk'
import type { ReactNode } from 'react'

import type { SubagentNodeStatus, ToolCallStatus } from './types'

/** Unix seconds → compact local day+clock; em dash when unknown. */
export function fmtTs(ts?: number | null): string {
  return ts != null && Number.isFinite(ts) ? fmtDayTime.format(ts * 1000) : '—'
}

/** Seconds → human latency ("420 ms", "1.4 s", "2 m 05 s"). */
export function fmtDurationS(s?: number | null): string {
  if (s == null || !Number.isFinite(s)) {
    return '—'
  }

  if (s < 1) {
    return `${Math.round(s * 1000)} ms`
  }

  if (s < 60) {
    return `${s.toFixed(1)} s`
  }

  const m = Math.floor(s / 60)

  return `${m} m ${String(Math.round(s % 60)).padStart(2, '0')} s`
}

export function shortId(id: string): string {
  return id.length > 10 ? `…${id.slice(-8)}` : id
}

export function toolTone(status: ToolCallStatus): 'bad' | 'good' | 'muted' | 'warn' {
  switch (status) {
    case 'running':
      return 'good'

    case 'generating':
      return 'warn'

    case 'error':
      return 'bad'

    default:
      return 'muted'
  }
}

export function subagentBadgeVariant(
  status: SubagentNodeStatus
): 'default' | 'destructive' | 'muted' | 'success' | 'warn' {
  switch (status) {
    case 'running':
      return 'default'

    case 'completed':
      return 'success'

    case 'failed':

    case 'error':
      return 'destructive'

    case 'timeout':
      return 'warn'

    default:
      return 'muted'
  }
}

/** Thin context-window bar — reads the streamed usage figures. */
export function ContextGauge({
  used,
  max,
  percent,
  className
}: {
  used?: number | null
  max?: number | null
  percent?: number | null
  className?: string
}) {
  const pct = percent ?? (used != null && max ? (used / max) * 100 : null)

  if (pct == null || !Number.isFinite(pct)) {
    return <span className={cn('text-(--ui-text-quaternary)', className)}>—</span>
  }

  const clamped = Math.max(0, Math.min(100, pct))

  return (
    <Tip label={`${compactNumber(used)} / ${compactNumber(max)} tokens (${clamped.toFixed(0)}%)`}>
      <span className={cn('inline-flex items-center gap-1.5', className)}>
        <span className="h-1 w-12 overflow-hidden rounded-full bg-(--ui-bg-quinary)">
          <span
            className={cn(
              'block h-full rounded-full',
              clamped > 90 ? 'bg-destructive' : clamped > 70 ? 'bg-amber-500' : 'bg-primary'
            )}
            style={{ width: `${clamped}%` }}
          />
        </span>
        <span className="tabular-nums text-(--ui-text-tertiary)">{clamped.toFixed(0)}%</span>
      </span>
    </Tip>
  )
}

/** One label/value cell in the drawer's meta grid. */
export function MetaItem({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  return (
    <div className={cn('min-w-0', className)}>
      <div className="text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-quaternary)">{label}</div>
      <div className="mt-0.5 truncate text-xs text-foreground">{children}</div>
    </div>
  )
}

/** Monospace block with a copy affordance — args, output, raw frames. */
export function CodeBlock({ text, tone, className }: { text: string; tone?: 'error'; className?: string }) {
  return (
    <div
      className={cn(
        'group/code relative min-w-0 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary)',
        className
      )}
    >
      <div className="absolute right-1.5 top-1.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover/code:opacity-100">
        <CopyButton appearance="icon" buttonSize="icon-xs" buttonVariant="ghost" text={text} />
      </div>
      <pre
        className={cn(
          'max-h-80 overflow-auto whitespace-pre-wrap break-words p-2.5 pr-8 font-mono text-[0.7rem] leading-relaxed',
          tone === 'error' ? 'text-destructive' : 'text-(--ui-text-secondary)'
        )}
      >
        {text}
      </pre>
    </div>
  )
}

/** Pretty-printed JSON for the Arguments tab (falls back to raw text). */
export function argsToText(args?: Record<string, unknown> | null, argsText?: string | null): string | null {
  if (argsText) {
    return argsText
  }

  if (!args || Object.keys(args).length === 0) {
    return null
  }

  try {
    return JSON.stringify(args, null, 2)
  } catch {
    return String(args)
  }
}

/** Status badge for a child node. */
export function SubagentStatusBadge({ status, label }: { status: SubagentNodeStatus; label: string }) {
  return (
    <Badge size="xs" variant={subagentBadgeVariant(status)}>
      {status === 'running' && <Codicon name="loading" size="0.6rem" spinning />}
      {label}
    </Badge>
  )
}
