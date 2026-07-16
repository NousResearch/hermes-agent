import { normalize } from '@/lib/text'

export const ENABLED_REASONING_EFFORTS = ['minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'] as const

export const REASONING_EFFORTS = ['none', ...ENABLED_REASONING_EFFORTS] as const

export const REASONING_DISPLAY_VALUES = ['show', 'hide', 'full', 'clamp'] as const

export const REASONING_COMMAND_HELP = [...REASONING_EFFORTS, ...REASONING_DISPLAY_VALUES].join('|')

export type EnabledReasoningEffort = (typeof ENABLED_REASONING_EFFORTS)[number]
export type ReasoningEffort = (typeof REASONING_EFFORTS)[number]

export function isEnabledReasoningEffort(value: unknown): value is EnabledReasoningEffort {
  return typeof value === 'string' && (ENABLED_REASONING_EFFORTS as readonly string[]).includes(value)
}

export function isReasoningEffort(value: unknown): value is ReasoningEffort {
  return typeof value === 'string' && (REASONING_EFFORTS as readonly string[]).includes(value)
}

export function normalizeEnabledReasoningEffort(
  value: unknown,
  fallback: EnabledReasoningEffort = 'medium'
): EnabledReasoningEffort {
  const normalized = normalize(String(value ?? ''))

  return isEnabledReasoningEffort(normalized) ? normalized : fallback
}
