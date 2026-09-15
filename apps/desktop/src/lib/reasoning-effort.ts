import { DEFAULT_REASONING_EFFORT, isReasoningEffort } from '@hermes/shared'

import { normalize } from '@/lib/text'

/** Default compact labels for non-rendering callers. UI callers pass their
 *  reactive translated modelOptions labels, including the common off label. */
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

export function reasoningEffortLabel(effort: string, labels: Readonly<Record<string, string>> = SHORT_LABELS): string {
  const key = normalize(effort)

  return key ? (Object.hasOwn(SHORT_LABELS, key) && Object.hasOwn(labels, key) ? labels[key] : effort) : ''
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
