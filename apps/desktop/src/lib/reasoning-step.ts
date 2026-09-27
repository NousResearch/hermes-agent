import { DEFAULT_REASONING_EFFORT, isReasoningEffort } from '@hermes/shared'

import { normalize } from '@/lib/text'

/** The level ladder a shortcut steps through: thinking off, then the normal
 *  levels in ascending order. `max`/`ultra` are deliberately absent — they are
 *  expensive tiers kept behind an explicit menu pick, so stepping must not
 *  cross into them (#71627). */
export const REASONING_STEP_LEVELS = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh'] as const

export type ReasoningStepLevel = (typeof REASONING_STEP_LEVELS)[number]

export type ReasoningStepDirection = -1 | 1

/** Resolve what a step starts from: `none` (thinking off), a real level, or
 *  the fallback for an unset/unknown value. */
const DEFAULT_STEP_LEVEL = DEFAULT_REASONING_EFFORT as ReasoningStepLevel

export function normalizeReasoningStepLevel(effort: string, fallback: string = DEFAULT_REASONING_EFFORT): ReasoningStepLevel {
  const value = normalize(effort || fallback)

  if (value === 'none') {
    return 'none'
  }

  return isReasoningEffort(value) && REASONING_STEP_LEVELS.includes(value as ReasoningStepLevel)
    ? (value as ReasoningStepLevel)
    : DEFAULT_STEP_LEVEL
}

/** One notch up/down the ladder, clamped at both ends — never wraps, never
 *  reaches past `xhigh` into `max`/`ultra`. The expensive tiers sit ABOVE the
 *  ladder: stepping down rejoins it at `xhigh`; stepping up has nowhere to go,
 *  so the current value is returned unchanged (callers treat that as a no-op). */
export function stepReasoningEffort(
  effort: string,
  direction: ReasoningStepDirection,
  fallback: string = DEFAULT_REASONING_EFFORT
): ReasoningStepLevel | 'max' | 'ultra' {
  const value = normalize(effort || fallback)

  if (value === 'max' || value === 'ultra') {
    return direction === -1 ? 'xhigh' : value
  }

  const current = normalizeReasoningStepLevel(value)
  const index = REASONING_STEP_LEVELS.indexOf(current)
  const next = Math.min(Math.max(index + direction, 0), REASONING_STEP_LEVELS.length - 1)

  return REASONING_STEP_LEVELS[next]
}

type RequestGateway = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

/** Push a stepped level onto the session (session-scoped `config.set`, same
 *  RPC the model menu uses). Returns the gateway-reported value so the caller
 *  can settle its optimistic write on the truth, not the guess. */
export async function writeSessionReasoningEffort(
  request: RequestGateway,
  sessionId: string,
  effort: ReasoningStepLevel | 'max' | 'ultra'
): Promise<ReasoningStepLevel> {
  const result = await request<{ value?: unknown }>('config.set', { key: 'reasoning', session_id: sessionId, value: effort })

  return normalizeReasoningStepLevel(typeof result?.value === 'string' ? result.value : effort)
}
