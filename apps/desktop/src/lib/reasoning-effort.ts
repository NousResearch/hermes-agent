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
export function resolveReasoningEffort(effort: string, fallback: string = DEFAULT_REASONING_EFFORT): string {
  const value = normalize(effort || fallback)

  if (value === 'none') {
    return ''
  }

  return isReasoningEffort(value) ? value : DEFAULT_REASONING_EFFORT
}

/** True when the composer's effort selection just mirrors the profile default
 *  (or is empty) — i.e. the user never picked a distinct level for this chat.
 *  useHermesConfig seeds the composer atom from `agent.reasoning_effort`, so
 *  the seeded value must ride `session.create` as inherited state, not as an
 *  explicit per-session override: a recorded override marks a deliberate user
 *  pick and disables adaptive reasoning escalation on the backend. An empty
 *  profile default means the backend's own fallback (medium). */
export function isInheritedEffortSelection(effort: string, profileDefault: string): boolean {
  const selected = normalize(effort)

  if (!selected) {
    return true
  }

  return selected === (normalize(profileDefault) || DEFAULT_REASONING_EFFORT)
}
