import { DEFAULT_REASONING_EFFORT, REASONING_EFFORT_VALUES } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import {
  isInheritedEffortSelection,
  isThinkingEnabled,
  reasoningEffortClamp,
  reasoningEffortLabel,
  resolveReasoningEffort
} from './reasoning-effort'

describe('reasoning-effort', () => {
  it('labels every level it claims to support', () => {
    for (const effort of REASONING_EFFORT_VALUES) {
      expect(reasoningEffortLabel(effort)).not.toBe('')
    }

    expect(reasoningEffortLabel('')).toBe('')
    // Unknown values pass through rather than silently reading as a real level.
    expect(reasoningEffortLabel('bogus')).toBe('bogus')
  })

  it('labels a route clamp from the gateway wire level only, never by inference', () => {
    expect(reasoningEffortLabel('ultra', 'max')).toBe('Ultra→Max')
    expect(reasoningEffortClamp('ultra', 'max')).toEqual({ effort: 'ultra', wire: 'max' })
    // Unknown ('' — not stamped yet / optimistic pick) or verbatim: plain label, no claim.
    expect(reasoningEffortLabel('ultra', '')).toBe('Ultra')
    expect(reasoningEffortLabel('ultra')).toBe('Ultra')
    expect(reasoningEffortLabel('high', 'high')).toBe('High')
    expect(reasoningEffortClamp('high', 'high')).toBeNull()
    expect(reasoningEffortClamp('none', '')).toBeNull()
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
