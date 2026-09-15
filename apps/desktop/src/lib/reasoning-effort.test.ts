import { DEFAULT_REASONING_EFFORT, REASONING_EFFORT_VALUES } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import { isThinkingEnabled, reasoningEffortLabel, resolveReasoningEffort } from './reasoning-effort'

describe('reasoning-effort', () => {
  it('labels every level it claims to support', () => {
    for (const effort of REASONING_EFFORT_VALUES) {
      expect(reasoningEffortLabel(effort)).not.toBe('')
    }

    expect(reasoningEffortLabel('')).toBe('')
    // Unknown values pass through rather than silently reading as a real level.
    expect(reasoningEffortLabel('bogus')).toBe('bogus')
    expect(reasoningEffortLabel('constructor')).toBe('constructor')
    const labels = { medium: '중간', none: '꺼짐', fast: '빠름' }
    expect(reasoningEffortLabel('medium', labels)).toBe(labels.medium)
    expect(reasoningEffortLabel('none', labels)).toBe(labels.none)
    expect(reasoningEffortLabel('constructor', labels)).toBe('constructor')
    expect(reasoningEffortLabel('fast', labels)).toBe('fast')
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
