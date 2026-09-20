import {
  DEFAULT_REASONING_EFFORT,
  isReasoningEffort,
  type ModelCapabilities,
  type ReasoningEffort
} from '@hermes/shared'

import { normalize } from '@/lib/text'

/** Compact labels for chrome where space is tight (pill, picker rows). Menus
 *  and settings use the translated `shell.modelOptions` strings instead. */
const SHORT_LABELS: Record<string, string> = {
  'budget:-1': 'Dynamic',
  auto: '', // No explicit level; do not invent an effort label.
  none: 'Off',
  minimal: 'Min',
  low: 'Low',
  medium: 'Med',
  high: 'High',
  xhigh: 'XHigh',
  max: 'Max',
  ultra: 'Ultra'
}

/**
 * A pick the route does not send verbatim: `ultra` is a Hermes-internal step
 * that every route clamps to its strongest level (`max` on OpenAI-compatible wires), and the
 * CLI's `/reasoning` says so ("ultra (sends max on this route)"). The wire
 * level comes from the gateway's `session.info.reasoning_effort_wire`; nothing
 * is inferred client-side, so an unknown ('' — not yet stamped, or an
 * optimistic pick) or verbatim wire reads as "no clamp".
 */
export function reasoningEffortClamp(
  effort: string,
  wire: string | undefined
): { effort: ReasoningEffort; wire: ReasoningEffort } | null {
  const picked = normalize(effort)
  const sent = normalize(wire ?? '')

  if (!sent || sent === picked || !isReasoningEffort(picked) || !isReasoningEffort(sent)) {
    return null
  }

  return { effort: picked, wire: sent }
}

/** Compact label; a clamped pick shows both ends ("Ultra→Max") so the pill
 *  never presents a Hermes step as a wire level the route does not have. */
export function reasoningEffortLabel(effort: string, wire?: string): string {
  const key = normalize(effort)
  const clamp = reasoningEffortClamp(effort, wire)

  if (clamp) {
    return `${SHORT_LABELS[clamp.effort]}→${SHORT_LABELS[clamp.wire]}`
  }

  return /^budget:\d+$/.test(key)
    ? `${Number(key.slice(7)).toLocaleString()} tok`
    : key
      ? (SHORT_LABELS[key] ?? effort)
      : ''
}

/** Resolve a row without inventing a level outside its provider contract.
 * Empty legacy values inherit; an explicit descriptor resolves empty to
 * `auto`, which leaves the level to the provider. */
export function resolveModelReasoningEffort(
  effort: string,
  inherited: string = '',
  capabilities?: Partial<
    Pick<ModelCapabilities, 'reasoning' | 'reasoning_control' | 'reasoning_efforts' | 'reasoning_budget'>
  >
): string {
  if (capabilities?.reasoning === false || capabilities?.reasoning_control === 'unsupported') {
    return ''
  }

  const value = normalize(effort || inherited)
  const allowed = capabilities?.reasoning_efforts

  if (value.startsWith('budget:')) {
    const budget = capabilities?.reasoning_budget

    if (value === 'budget:-1' && budget?.dynamic) {
      return value
    }

    const tokens = Number(value.slice(7))

    return budget &&
      /^budget:\d+$/.test(value) &&
      Number.isSafeInteger(tokens) &&
      tokens >= budget.min &&
      tokens <= budget.max
      ? value
      : ''
  }

  if (value === 'auto') {
    return value
  }

  if (allowed == null) {
    return value === 'none' ? value : resolveReasoningEffort(value)
  }

  if (!value) {
    return 'auto'
  }

  return allowed.includes(value) ? value : ''
}

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
