import { compactNumber } from '@hermes/shared'
import { useState } from 'react'

import { StableText } from '@/components/chat/stable-text'
import { useViewedInterval } from '@/hooks/use-viewed-interval'
import { formatDisplayCost } from '@/lib/currency'
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

/** `↓12.3k ↑4.2k` — session-cumulative prompt in / completion out; '' before the first call.
 *  The status bar's token readout, paired with the context meter (occupancy) and the cache hit
 *  rate (how much of the prompt was served from cache). */
export function sessionTokensLabel(usage: UsageStats): string {
  const input = usage.input ?? 0
  const output = usage.output ?? 0

  if (!input && !output) {
    return ''
  }

  return `↓${compactNumber(input)} ↑${compactNumber(output)}`
}

/** Money formatting sized to what these sessions actually cost: sub-cent turns keep three or four
 *  decimals instead of collapsing to a useless `¥0.00`. The backend bills in USD; the conversion
 *  and the symbol live in `lib/currency.ts` so the stored figure never changes meaning. */
export function formatCostUsd(cost: number): string {
  return formatDisplayCost(cost)
}

/** `~$0.0312` for estimated spend, the provider's own figure when it reported one, and ''
 *  when the route has no pricing entry (the backend omits `cost_usd` rather than sending $0).
 *  `included` rides through as its own label: a subscription session has no per-turn price. */
export function sessionCostLabel(usage: UsageStats, includedLabel = 'included'): string {
  if (usage.cost_status === 'included') {
    return includedLabel
  }

  const cost = usage.cost_usd

  if (typeof cost !== 'number' || !Number.isFinite(cost)) {
    return ''
  }

  return `${usage.cost_status === 'estimated' ? '~' : ''}${formatCostUsd(cost)}`
}

export function LiveDuration({ since }: { since: number | null | undefined }) {
  const [now, setNow] = useState(() => Date.now())

  useViewedInterval(() => setNow(Date.now()), 1000, Boolean(since))

  if (!since) {
    return null
  }

  return <StableText>{formatDuration(now - since)}</StableText>
}
