/** Hermes' reasoning levels, in ascending order — mirrors the backend's
 *  VALID_REASONING_EFFORTS (hermes_constants.py). `none` is not a level: it's
 *  thinking disabled. */
export const REASONING_EFFORTS = ['minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'] as const

export type ReasoningEffort = (typeof REASONING_EFFORTS)[number]

/** The scale plus the off state — the full set a config value may hold. */
export const REASONING_EFFORT_VALUES = ['none', ...REASONING_EFFORTS] as const

export type ReasoningEffortValue = (typeof REASONING_EFFORT_VALUES)[number]

/** Hermes' built-in level when neither the surface nor the profile config
 *  specifies one (mirrors the backend's own fallback). */
export const DEFAULT_REASONING_EFFORT: ReasoningEffort = 'medium'

/** True for a real level (case-insensitive, trimmed); `none` is not a level. */
export const isReasoningEffort = (value: string): value is ReasoningEffort =>
  REASONING_EFFORTS.includes(value.trim().toLowerCase() as ReasoningEffort)

/**
 * The levels a picker should offer for a route whose clamp vocabulary is
 * `supported` (#114029): the internal ladder intersected with it, in ladder
 * order. Unknown (`null`/empty) or a set with no ladder members keeps the
 * full ladder — a bespoke non-ladder vocabulary is never a narrowing signal.
 * `keep` (the surface's current value) is appended when the filter would drop
 * it, so a saved pick never leaves a Select blank or a radio deselected.
 */
export function filterReasoningEfforts(
  supported: readonly string[] | null | undefined,
  keep?: string
): readonly ReasoningEffort[] {
  if (!supported || supported.length === 0) {
    return REASONING_EFFORTS
  }

  const supportedSet = new Set(supported.map(v => v.trim().toLowerCase()))
  const filtered = REASONING_EFFORTS.filter(level => supportedSet.has(level))

  if (filtered.length === 0) {
    return REASONING_EFFORTS
  }

  const kept = (keep ?? '').trim().toLowerCase() as ReasoningEffort

  return kept && REASONING_EFFORTS.includes(kept) && !filtered.includes(kept)
    ? [...filtered, kept]
    : filtered
}
