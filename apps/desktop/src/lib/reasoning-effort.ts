import { DEFAULT_REASONING_EFFORT, isReasoningEffort } from '@hermes/shared'
import type { ModelCapabilities } from '@hermes/shared'

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

export function reasoningEffortLabel(effort: string): string {
  const key = normalize(effort)

  return /^budget:\d+$/.test(key)
    ? `${Number(key.slice(7)).toLocaleString()} tok`
    : key
      ? (SHORT_LABELS[key] ?? effort)
      : ''
}

/** Unknown saved values stay unselected instead of acquiring a false label.
 *  Empty inherits; auto explicitly leaves the level to the provider. */
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
    if (value === 'budget:-1' && budget?.dynamic) return value
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
