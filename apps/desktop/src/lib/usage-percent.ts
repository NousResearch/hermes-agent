/**
 * Whole-percent used for a usage-window allowance. Same rule as the portal
 * (`formatUsedPercentLabel`) and the CLI (`format_percent_used`): floor, any
 * sliver reads `<1%` (never `0%`), and `100%` only once it is actually exhausted.
 */
export function formatPercentUsed(used: number): string {
  // Round to a micro-basis-point first: `1 - left` can land a hair under an exact bp.
  const bp = Math.floor(Math.round(Math.min(1, Math.max(0, used)) * 10_000 * 1e6) / 1e6)

  if (bp <= 0) {
    return '0%'
  }

  if (bp < 100) {
    return '<1%'
  }

  return `${Math.floor(bp / 100)}%`
}

/** `5h` / `2d 3h` / `12m` until a window allowance resets; null when unknown or under a minute away
 *  (the caller words it). */
export function formatResetsIn(resetsAtMs?: null | number, nowMs = Date.now()): null | string {
  if (typeof resetsAtMs !== 'number' || !Number.isFinite(resetsAtMs)) {
    return null
  }

  const seconds = Math.max(0, Math.floor((resetsAtMs - nowMs) / 1000))

  if (seconds < 60) {
    return null
  }

  const days = Math.floor(seconds / 86_400)
  const hours = Math.floor((seconds % 86_400) / 3_600)

  if (days > 0) {
    return hours > 0 ? `${days}d ${hours}h` : `${days}d`
  }

  const minutes = Math.floor((seconds % 3_600) / 60)

  return hours > 0 ? `${hours}h` : `${minutes}m`
}
