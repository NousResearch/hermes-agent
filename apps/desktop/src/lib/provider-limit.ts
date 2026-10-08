import type { ModelOptionProvider, ProviderLimit, ProviderUsageAccount, ProviderUsageWindow } from '@hermes/shared'

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
  // A provider-scoped observation describes one credential, never its siblings.
  if (provider.usage?.accounts?.length) {
    return null
  }

  const windows = liveUsageWindows(provider.usage?.windows ?? [], nowMs)

  return windows.length > 0 ? windows : null
}

function liveUsageWindows(windows: ProviderUsageWindow[], nowMs: number): UsageWindowView[] {
  return windows
    .filter(w => Number.isFinite(w.used_percent))
    .map(w => ({
      label: w.label,
      remaining: Math.max(0, Math.min(100, Math.round(100 - w.used_percent))),
      resetMs: parseMs(w.resets_at)
    }))
    .filter(w => w.resetMs === null || w.resetMs > nowMs)
    .sort((a, b) => a.remaining - b.remaining)
}

export interface PoolAccountView {
  id: string
  label: string
  state: ProviderUsageAccount['state']
  resetMs: null | number
  windows: UsageWindowView[]
}

/** Keep unknown accounts unknown; a passed reset is not evidence of fresh quota. */
export function poolUsage(provider: ModelOptionProvider, nowMs = Date.now()) {
  const entries = provider.usage?.accounts

  if (!entries?.length) {
    return null
  }

  const accounts: PoolAccountView[] = entries.map(entry => {
    const resetMs = parseMs(entry.resets_at)
    const windows = liveUsageWindows(entry.windows ?? [], nowMs)
    const expired = resetMs !== null && resetMs <= nowMs
    const stale = entry.state === 'ready' && windows.length === 0

    return {
      id: entry.id,
      label: entry.label ?? '',
      windows,
      resetMs,
      state: expired || stale ? 'unknown' : entry.state
    }
  })

  const limited = accounts.filter(entry => entry.state === 'limited')

  return {
    accounts,
    limited: limited.length,
    // Each account's reset already includes ALL its exhausted windows.
    resetMs:
      limited.length === accounts.length
        ? Math.min(...limited.map(entry => entry.resetMs ?? Number.POSITIVE_INFINITY))
        : null
  }
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
