import { describe, expect, it } from 'vitest'

import {
  ENABLED_REASONING_EFFORTS,
  isReasoningEffort,
  normalizeEnabledReasoningEffort,
  REASONING_COMMAND_HELP,
  REASONING_DISPLAY_VALUES,
  REASONING_EFFORTS
} from './reasoning-effort'

describe('desktop reasoning effort contract', () => {
  it('keeps the enabled and off-inclusive Hermes effort ladders ordered', () => {
    expect(ENABLED_REASONING_EFFORTS).toEqual(['minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'])
    expect(REASONING_EFFORTS).toEqual(['none', ...ENABLED_REASONING_EFFORTS])
  })

  it('keeps display commands separate from effort values', () => {
    expect(REASONING_DISPLAY_VALUES).toEqual(['show', 'hide', 'full', 'clamp'])
    expect(REASONING_COMMAND_HELP).toBe([...REASONING_EFFORTS, ...REASONING_DISPLAY_VALUES].join('|'))
    expect(REASONING_DISPLAY_VALUES.every(value => !isReasoningEffort(value))).toBe(true)
  })

  it('recognizes exact effort values and normalizes user-facing enabled values', () => {
    expect(REASONING_EFFORTS.every(isReasoningEffort)).toBe(true)
    expect(isReasoningEffort(' ULTRA ')).toBe(false)
    expect(normalizeEnabledReasoningEffort(' ULTRA ')).toBe('ultra')
    expect(normalizeEnabledReasoningEffort('unknown')).toBe('medium')
  })
})
