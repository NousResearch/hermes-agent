import { compactNumber } from '@hermes/shared'
import { useState } from 'react'

import { StableText } from '@/components/chat/stable-text'
import { useViewedInterval } from '@/hooks/use-viewed-interval'
import type { UsageStats } from '@/types/hermes'

export function formatDuration(elapsedMs: number): string {
  const totalSeconds = Math.max(0, Math.floor(elapsedMs / 1000))
  const seconds = totalSeconds % 60
  const minutes = Math.floor(totalSeconds / 60) % 60
  const hours = Math.floor(totalSeconds / 3600)
  const ss = String(seconds).padStart(2, '0')
  const mm = String(minutes).padStart(2, '0')

  return hours > 0 ? `${hours}:${mm}:${ss}` : `${minutes}:${ss}`
}

export function compactPath(path: string, max = 44): string {
  const trimmed = path.trim()

  if (trimmed.length <= max) {
    return trimmed
  }

  const segments = trimmed.split('/').filter(Boolean)

  if (segments.length < 2) {
    return `…${trimmed.slice(-(max - 1))}`
  }

  const tail = segments.slice(-2).join('/')

  return tail.length + 2 >= max ? `…${tail.slice(-(max - 1))}` : `…/${tail}`
}

export function contextBar(percent: number | undefined, width = 10): string {
  const bounded = Math.max(0, Math.min(100, percent ?? 0))
  const filled = Math.round((bounded / 100) * width)

  return `${'█'.repeat(filled)}${'░'.repeat(width - filled)}`
}

export function usageContextLabel(usage: UsageStats): string {
  if (usage.context_max) {
    return `${usage.context_estimated ? '~' : ''}${compactNumber(usage.context_used ?? 0)}/${compactNumber(usage.context_max)}`
  }

  return usage.total > 0 ? `${compactNumber(usage.total)} tok` : ''
}

export function contextBarLabel(usage: UsageStats): string {
  if (!usage.context_max) {
    return ''
  }

  const pct = Math.max(0, Math.min(100, Math.round(usage.context_percent ?? 0)))

  return `[${contextBar(usage.context_percent)}] ${usage.context_estimated ? '~' : ''}${pct}%`
}

/** `87%` for a reported hit rate; '' when the backend omitted it (no cache
 *  reads yet, or a provider that doesn't report them). The backend already
 *  clamps and rounds, so this only guards a malformed/absent field. */
export function cacheHitLabel(usage: UsageStats): string {
  const pct = usage.cache_hit_pct

  return typeof pct === 'number' && Number.isFinite(pct) ? `${Math.round(pct)}%` : ''
}

/** `42 t/s` for the rolling throughput; '' before the first completed call. */
export function tokensPerSecondLabel(usage: UsageStats): string {
  const tps = usage.avg_tps

  return typeof tps === 'number' && Number.isFinite(tps) && tps > 0 ? `${Math.round(tps)} t/s` : ''
}

/** `11.9s` for the rolling API latency; '' before the first completed call.
 *  Backend (`_get_usage`) omits the field rather than sending 0 when it has no
 *  data — a provider with no reported timings, or a session before its call. */
export function latencyLabel(usage: UsageStats): string {
  const latency = usage.avg_latency_s

  return typeof latency === 'number' && Number.isFinite(latency) && latency > 0 ? `${latency.toFixed(1)}s` : ''
}

/** CLI parity (`_status_bar_context_style`): ≥95% destructive, >80% orange,
 *  ≥50% caution, else unstyled. Returns a class NAME, never a colour —
 *  theming rides the CSS custom properties the classes resolve to. */
export function contextUsageClass(usage: UsageStats): string {
  const pct = usage.context_percent ?? 0

  if (pct >= 95) {return 'text-destructive hover:text-destructive'}

  if (pct > 80) {return 'text-(--ui-orange) hover:text-(--ui-orange)'}

  if (pct >= 50) {return 'text-(--ui-yellow) hover:text-(--ui-yellow)'}

  return ''
}

/** CLI cache bar is INVERTED: a low hit rate is the bad state (you are paying
 *  full price for repeated prefixes). ≥70% good, ≥40% caution, else orange. */
export function cacheHitClass(usage: UsageStats): string {
  const pct = usage.cache_hit_pct

  if (typeof pct !== 'number' || !Number.isFinite(pct)) {return ''}

  if (pct >= 70) {return 'text-(--ui-green) hover:text-(--ui-green)'}

  if (pct >= 40) {return 'text-(--ui-yellow) hover:text-(--ui-yellow)'}

  return 'text-(--ui-orange) hover:text-(--ui-orange)'
}

/** Compression count ladder (CLI `_compression_count_style`): ≥10 destructive,
 *  ≥5 caution, else unstyled — the count itself is normal, repeated
 *  compaction of a shrinking window is what needs eyes. */
export function compressionCountClass(count: number): string {
  if (count >= 10) {return 'text-destructive hover:text-destructive'}

  if (count >= 5) {return 'text-(--ui-yellow) hover:text-(--ui-yellow)'}

  return ''
}

export function LiveDuration({ since }: { since: number | null | undefined }) {
  const [now, setNow] = useState(() => Date.now())

  useViewedInterval(() => setNow(Date.now()), 1000, Boolean(since))

  if (!since) {
    return null
  }

  return <StableText>{formatDuration(now - since)}</StableText>
}
