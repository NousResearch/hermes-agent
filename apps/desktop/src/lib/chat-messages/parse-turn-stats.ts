import type { TurnStats } from '@/types/hermes'

/** Wire key → `TurnStats` key. The gateway also sends `model` and `provider`; nothing
 *  renders them, so they are not carried into the app's shape. */
const NUMERIC_FIELDS: ReadonlyArray<readonly [string, keyof TurnStats]> = [
  ['duration_s', 'durationS'],
  ['input', 'input'],
  ['output', 'output'],
  ['reasoning', 'reasoning'],
  ['cache_read', 'cacheRead'],
  ['cache_write', 'cacheWrite'],
  ['calls', 'calls'],
  ['cost_usd', 'costUsd']
]

/** Lenient snake_case dict → TurnStats. Negative and non-numeric values are dropped, so a
 *  provider that reports a field oddly loses that segment rather than the whole strip. */
export function parseTurnStats(raw: unknown): TurnStats | undefined {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return undefined
  }

  const src = raw as Record<string, unknown>
  const out: TurnStats = {}

  for (const [wireKey, key] of NUMERIC_FIELDS) {
    const value = src[wireKey]

    if (typeof value === 'number' && Number.isFinite(value) && value >= 0) {
      out[key] = value
    }
  }

  return Object.keys(out).length > 0 ? out : undefined
}
