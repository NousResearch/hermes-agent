import type { GatewayRequester } from './yolo-session'

export const REASONING_EFFORT_OPTIONS = ['minimal', 'low', 'medium', 'high', 'xhigh'] as const
export const REASONING_STEP_LEVELS = ['none', ...REASONING_EFFORT_OPTIONS] as const

export type ReasoningEffort = (typeof REASONING_STEP_LEVELS)[number]
export type ReasoningStepDirection = -1 | 1

/** Empty means Hermes' default (medium); unknown stale values normalize there too. */
export function normalizeReasoningEffort(effort: string): ReasoningEffort {
  const value = effort.trim().toLowerCase()

  if (REASONING_STEP_LEVELS.some(level => level === value)) {
    return value as ReasoningEffort
  }

  return 'medium'
}

/** Move one notch through off → minimal → low → medium → high → xhigh. */
export function stepReasoningEffort(effort: string, direction: ReasoningStepDirection): ReasoningEffort {
  const current = normalizeReasoningEffort(effort)
  const index = REASONING_STEP_LEVELS.indexOf(current)
  const nextIndex = Math.min(Math.max(index + direction, 0), REASONING_STEP_LEVELS.length - 1)

  return REASONING_STEP_LEVELS[nextIndex]
}

export async function writeSessionReasoningEffort(
  requestGateway: GatewayRequester,
  sessionId: string,
  effort: ReasoningEffort
): Promise<ReasoningEffort> {
  const result = await requestGateway<{ value?: string }>('config.set', {
    key: 'reasoning',
    session_id: sessionId,
    value: effort
  })

  return normalizeReasoningEffort(result?.value ?? effort)
}
