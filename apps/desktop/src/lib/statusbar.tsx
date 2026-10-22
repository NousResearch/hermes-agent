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

/**
 * Session spend: `$4.27` when the provider reported it, `~$4.27` when it is our
 * estimate, `$0.00` for a subscription session (its calls cost nothing extra),
 * '' while the backend has no figure — a missing value means "not measured",
 * and painting `$0.00` there is the exact dishonesty that withdrew cost
 * readouts from every surface in `fd2a35b169`.
 *
 * Mirrors `agent/usage_pricing.py::format_cost_label` so the desktop chip and
 * the CLI/insights cost rows read the same at sub-cent amounts: 4dp below a
 * cent, and `<$0.0001` rather than a zero-looking `$0.0000`.
 */
export function sessionCostLabel(usage: UsageStats): string {
  const cost = usage.cost_usd

  if (typeof cost !== 'number' || !Number.isFinite(cost) || cost < 0) {
    return ''
  }

  // `actual` is provider-reported and `included` is a definitive $0, so
  // neither is an approximation and neither takes the `~`.
  const exact = usage.cost_status === 'actual' || usage.cost_status === 'included'

  if (cost === 0) {
    return exact ? '$0.00' : ''
  }

  if (cost < 0.01) {
    const sub = cost.toFixed(4)

    return `${exact ? '' : '~'}$${sub === '0.0000' ? '<0.0001' : sub}`
  }

  return `${exact ? '' : '~'}$${cost.toFixed(2)}`
}

/**
 * Fold an incoming tick's spend into what is already shown, never downwards.
 *
 * A session's spend only grows, so a smaller incoming figure is a MEASUREMENT
 * boundary, not a correction: the gateway rebuilds every counter at 0 on resume,
 * so the first turn after one reports only its own post-resume cost. Letting
 * that through would drop a $5.00 session to $0.02. Returns the value to keep —
 * the current one when the tick does not beat it — because the caller spreads
 * the result AFTER the raw tick, so an empty object would leave the smaller
 * figure in place.
 */
export function costMerge(
  current: UsageStats,
  incoming: Pick<UsageStats, 'cost_status' | 'cost_usd'>
): Pick<UsageStats, 'cost_status' | 'cost_usd'> {
  const next = incoming.cost_usd

  if (typeof next !== 'number' || !Number.isFinite(next)) {
    return {}
  }

  const shown = current.cost_usd

  if (typeof shown === 'number' && Number.isFinite(shown) && shown >= next) {
    return { cost_status: current.cost_status, cost_usd: shown }
  }

  return { cost_status: incoming.cost_status ?? 'unknown', cost_usd: next }
}

export function LiveDuration({ since }: { since: number | null | undefined }) {
  const [now, setNow] = useState(() => Date.now())

  useViewedInterval(() => setNow(Date.now()), 1000, Boolean(since))

  if (!since) {
    return null
  }

  return <StableText>{formatDuration(now - since)}</StableText>
}
