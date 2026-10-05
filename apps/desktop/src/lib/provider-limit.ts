import type { ModelOptionProvider, ProviderLimit } from '@hermes/shared'

import { DAY, fmtClock, fmtDayTime, startOfLocalDay } from '@/lib/time'

function parseMs(value: null | string | undefined): null | number {
  const ms = value ? Date.parse(value) : NaN

  return Number.isFinite(ms) ? ms : null
}

/** The provider's rate limit (`account`: the whole login is out, another model
 *  won't help; `models`: only those models are), or null when it has none or
 *  every reset has already passed, so a stale catalog clears itself. */
export function providerLimit(provider: ModelOptionProvider, nowMs = Date.now()): null | ProviderLimit {
  const limit = provider.limit

  if (!limit) {
    return null
  }

  if (limit.scope === 'account') {
    const resetMs = parseMs(limit.resets_at)

    return resetMs === null || resetMs > nowMs ? limit : null
  }

  const live = Object.entries(limit.models ?? {}).filter(([, at]) => (parseMs(at) ?? 0) > nowMs)

  return live.length > 0 ? { ...limit, models: Object.fromEntries(live) } : null
}

/** Account-wide reset time (ms) while the whole provider is limited. */
export function accountResetMs(provider: ModelOptionProvider, nowMs = Date.now()): null | number {
  const limit = providerLimit(provider, nowMs)

  return limit?.scope === 'account' ? (parseMs(limit.resets_at) ?? Number.POSITIVE_INFINITY) : null
}

/** This model's own reset time (ms) when only it is cooling down. */
export function modelResetMs(provider: ModelOptionProvider, model: string, nowMs = Date.now()): null | number {
  const limit = providerLimit(provider, nowMs)

  return limit?.scope === 'models' ? parseMs(limit.models?.[model]) : null
}

/** Remaining share at or under which a picker shows the usage chip, and turns it amber. */
export const USAGE_NOTICE_PERCENT = 20
export const USAGE_WARN_PERCENT = 10

export interface UsageWindowView {
  label: string
  remaining: number
  resetMs: null | number
}

/** The provider's live usage windows (rolled-over ones dropped), tightest first, or null when it
 *  reports none. The first is the one you'll hit first, so it's the one the chip shows. */
export function usageWindows(provider: ModelOptionProvider, nowMs = Date.now()): null | UsageWindowView[] {
  const windows = (provider.usage?.windows ?? [])
    .map(w => ({
      label: w.label,
      remaining: Math.max(0, Math.min(100, Math.round(100 - w.used_percent))),
      resetMs: parseMs(w.resets_at)
    }))
    .filter(w => w.resetMs === null || w.resetMs > nowMs)
    .sort((a, b) => a.remaining - b.remaining)

  return windows.length > 0 ? windows : null
}

/** `4:30 PM` today, `Oct 6, 9:00 AM` on another day. Infinity = unknown. */
export function formatReset(ms: number, nowMs = Date.now()): null | string {
  if (!Number.isFinite(ms)) {
    return null
  }

  return startOfLocalDay(ms) === startOfLocalDay(nowMs) || ms - nowMs < DAY / 4
    ? fmtClock.format(ms)
    : fmtDayTime.format(ms)
}
