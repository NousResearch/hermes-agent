import { normalize } from '@/lib/text'

/** Hermes' reasoning levels, in ascending order — mirrors the backend's
 *  VALID_REASONING_EFFORTS (hermes_constants.py). `none` is not a level: it's
 *  thinking disabled, owned by the Thinking toggle rather than the scale. */
export const REASONING_EFFORTS = ['minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'] as const

export type ReasoningEffort = (typeof REASONING_EFFORTS)[number]

/** The scale plus the off state — the full set a config value may hold. */
export const REASONING_EFFORT_VALUES = ['none', ...REASONING_EFFORTS] as const

/** Hermes' built-in level when neither the surface nor the profile config
 *  specifies one (mirrors the backend's own fallback). */
export const DEFAULT_REASONING_EFFORT: ReasoningEffort = 'medium'

/** Return exact model-supported levels in Hermes' canonical order.
 * Missing, empty, or malformed metadata deliberately falls back to the full
 * ladder so an older backend or an unknown catalog shape never blanks the UI. */
export function reasoningEffortsForModel(supportedEfforts?: readonly string[]): ReasoningEffort[] {
  if (!supportedEfforts?.length) {
    return [...REASONING_EFFORTS]
  }

  const normalized = supportedEfforts.map(value => normalize(value))

  if (normalized.some(value => !isReasoningEffort(value))) {
    return [...REASONING_EFFORTS]
  }

  const supported = new Set(normalized)
  const filtered = REASONING_EFFORTS.filter(value => supported.has(value))

  return filtered.length > 0 ? filtered : [...REASONING_EFFORTS]
}

/** Compact labels for chrome where space is tight (pill, picker rows). Menus
 *  and settings use the translated `shell.modelOptions` strings instead. */
const SHORT_LABELS: Record<string, string> = {
  none: 'Off',
  minimal: 'Min',
  low: 'Low',
  medium: 'Med',
  high: 'High',
  xhigh: 'XHigh',
  max: 'Max',
  ultra: 'Ultra'
}

export function reasoningEffortLabel(effort: string): string {
  const key = normalize(effort)

  return key ? (SHORT_LABELS[key] ?? effort) : ''
}

export const isReasoningEffort = (value: string): value is ReasoningEffort =>
  REASONING_EFFORTS.includes(normalize(value) as ReasoningEffort)

/** Thinking is on unless a level explicitly says otherwise; an empty value
 *  means "inherit", so it resolves through `fallback` first. */
export const isThinkingEnabled = (effort: string, fallback: string = DEFAULT_REASONING_EFFORT): boolean =>
  normalize(effort || fallback) !== 'none'

/** The level a scale control should show. Empty inherits `fallback`; `none`
 *  (thinking off) selects nothing; anything unrecognized clamps to the default. */
export function resolveSupportedReasoningEffort(
  effort: string,
  fallback: string = DEFAULT_REASONING_EFFORT,
  supportedEfforts?: readonly string[],
  canDisableReasoning = true
): string {
  const levels = reasoningEffortsForModel(supportedEfforts)
  const value = normalize(effort || fallback)

  if (value === 'none' && canDisableReasoning) {
    return 'none'
  }

  if (isReasoningEffort(value) && levels.includes(value)) {
    return value
  }

  const fallbackValue = normalize(fallback)

  if (fallbackValue === 'none' && canDisableReasoning) {
    return 'none'
  }

  if (isReasoningEffort(fallbackValue) && levels.includes(fallbackValue)) {
    return fallbackValue
  }

  return levels[0] ?? DEFAULT_REASONING_EFFORT
}

export function resolveReasoningEffort(
  effort: string,
  fallback: string = DEFAULT_REASONING_EFFORT,
  supportedEfforts?: readonly string[],
  canDisableReasoning = true
): string {
  const resolved = resolveSupportedReasoningEffort(effort, fallback, supportedEfforts, canDisableReasoning)

  return resolved === 'none' ? '' : resolved
}
