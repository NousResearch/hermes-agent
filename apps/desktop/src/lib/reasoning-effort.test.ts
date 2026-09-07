import { describe, expect, it } from 'vitest'

import {
  DEFAULT_REASONING_EFFORT,
  isReasoningEffort,
  isThinkingEnabled,
  REASONING_EFFORT_VALUES,
  REASONING_EFFORTS,
  reasoningEffortLabel,
  resolveReasoningEffort
} from './reasoning-effort'

describe('reasoning-effort', () => {
  it('keeps the scale ascending and `none` off it', () => {
    expect(REASONING_EFFORTS).not.toContain('none')
    expect(REASONING_EFFORT_VALUES[0]).toBe('none')
    expect(REASONING_EFFORT_VALUES).toHaveLength(REASONING_EFFORTS.length + 1)
  })

  it('labels every level it claims to support', () => {
    for (const effort of REASONING_EFFORT_VALUES) {
      expect(reasoningEffortLabel(effort)).not.toBe('')
    }

    expect(reasoningEffortLabel('')).toBe('')
    // Unknown values pass through rather than silently reading as a real level.
    expect(reasoningEffortLabel('bogus')).toBe('bogus')
  })

  it('recognizes only real scale levels', () => {
    expect(isReasoningEffort(DEFAULT_REASONING_EFFORT)).toBe(true)
    expect(isReasoningEffort('HIGH')).toBe(true)
    expect(isReasoningEffort('none')).toBe(false)
    expect(isReasoningEffort('bogus')).toBe(false)
  })

  it('treats empty as inherit and only `none` as off', () => {
    expect(isThinkingEnabled('none')).toBe(false)
    expect(isThinkingEnabled('high')).toBe(true)
    // Empty inherits the fallback, so an off fallback reads as off.
    expect(isThinkingEnabled('', 'none')).toBe(false)
    expect(isThinkingEnabled('', 'high')).toBe(true)
  })

  it('resolves a scale value: inherit, off, or clamp', () => {
    expect(resolveReasoningEffort('high')).toBe('high')
    // Empty inherits the profile default rather than snapping to medium.
    expect(resolveReasoningEffort('', 'ultra')).toBe('ultra')
    // Off selects nothing on the scale.
    expect(resolveReasoningEffort('none')).toBe('')
    expect(resolveReasoningEffort('bogus')).toBe(DEFAULT_REASONING_EFFORT)
  })
})
