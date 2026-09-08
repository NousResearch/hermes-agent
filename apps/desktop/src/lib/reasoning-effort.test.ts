import { describe, expect, it } from 'vitest'

import {
  DEFAULT_REASONING_EFFORT,
  isInheritedEffortSelection,
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

  it('treats a composer selection mirroring the profile default as inherited', () => {
    // The defect scenario: useHermesConfig seeds the composer with the
    // profile default — session.create must not record it as an explicit
    // override (which disables adaptive reasoning escalation).
    expect(isInheritedEffortSelection('medium', 'medium')).toBe(true)
    expect(isInheritedEffortSelection(' High ', 'high')).toBe(true)
    expect(isInheritedEffortSelection('none', 'none')).toBe(true)
    expect(isInheritedEffortSelection('', 'high')).toBe(true)
  })

  it('treats a distinct composer selection as an explicit override', () => {
    expect(isInheritedEffortSelection('high', 'medium')).toBe(false)
    expect(isInheritedEffortSelection('none', 'medium')).toBe(false)
    expect(isInheritedEffortSelection('high', '')).toBe(false)
  })

  it('reads an empty profile default as the backend fallback (medium)', () => {
    expect(isInheritedEffortSelection(DEFAULT_REASONING_EFFORT, '')).toBe(true)
  })
})
