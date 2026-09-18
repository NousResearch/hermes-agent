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

export function LiveDuration({ since }: { since: number | null | undefined }) {
  const [now, setNow] = useState(() => Date.now())

  useViewedInterval(() => setNow(Date.now()), 1000, Boolean(since))

  if (!since) {
    return null
  }

  return <StableText>{formatDuration(now - since)}</StableText>
}

/** Seconds left on the prompt cache, or null when the route reports no window at all.
 *  Zero is a real answer (the window lapsed) and is falsy, so callers can gate on it. */
export function cacheTtlRemainingSeconds(
  ttlS: number | undefined,
  refreshedAt: number | undefined,
  nowMs: number = Date.now()
): number | null {
  if (!ttlS || !refreshedAt) {
    return null
  }

  return Math.max(0, Math.floor(refreshedAt + ttlS - nowMs / 1000))
}

/** `m:ss` left on the prompt cache; empty when the route reports no window. */
export function cacheTtlRemainingLabel(
  ttlS: number | undefined,
  refreshedAt: number | undefined,
  nowMs: number = Date.now()
): string {
  const remaining = cacheTtlRemainingSeconds(ttlS, refreshedAt, nowMs)

  if (remaining === null) {
    return ''
  }

  return `${Math.floor(remaining / 60)}:${String(remaining % 60).padStart(2, '0')}`
}

// Renders `0:00` rather than nothing once the window lapses: the statusbar slot keeps its
// chrome either way, so an empty label leaves a hoverable gap next to a stale tooltip.
export function CacheTtlCountdown({ ttlS, refreshedAt }: { ttlS: number; refreshedAt: number }) {
  const [now, setNow] = useState(() => Date.now())

  useViewedInterval(() => setNow(Date.now()), 1000, true)

  return <StableText>{`cache ${cacheTtlRemainingLabel(ttlS, refreshedAt, now)}`}</StableText>
}
