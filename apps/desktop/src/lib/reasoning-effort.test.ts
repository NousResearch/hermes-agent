import { DEFAULT_REASONING_EFFORT, isReasoningEffort, REASONING_EFFORT_VALUES } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import {
  isThinkingEnabled,
  reasoningEffortLabel,
  resolveModelReasoningEffort,
  resolveReasoningEffort
} from './reasoning-effort'

describe('reasoning-effort', () => {
  it('preserves valid saved budgets and refuses to transfer them to effort-only models', () => {
    const caps = { reasoning: true, reasoning_efforts: [], reasoning_budget: { min: 128, max: 32768 } }
    expect(resolveModelReasoningEffort('budget:4096', '', caps)).toBe('budget:4096')
    expect(resolveModelReasoningEffort('', 'budget:4096', caps)).toBe('budget:4096')
    expect(resolveModelReasoningEffort('budget:127', '', caps)).toBe('')
    expect(resolveModelReasoningEffort('budget:4096', '', { reasoning_efforts: ['high'] })).toBe('')
    expect(reasoningEffortLabel('budget:4096')).toContain('tok')
  })
  it('labels every level it claims to support', () => {
    for (const effort of REASONING_EFFORT_VALUES) {
      expect(reasoningEffortLabel(effort)).not.toBe('')
    }

    expect(reasoningEffortLabel('')).toBe('')
    // Unknown values pass through rather than silently reading as a real level.
    expect(reasoningEffortLabel('bogus')).toBe('bogus')
  })

  it('keeps the unset state out of visible effort labels', () => {
    expect(reasoningEffortLabel('auto')).toBe('')
    expect(REASONING_EFFORT_VALUES).not.toContain('auto')
    expect(isReasoningEffort('auto')).toBe(false)
    expect(resolveModelReasoningEffort('auto', 'high', { reasoning: true, reasoning_efforts: [] })).toBe('auto')
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

it('keeps inheritance, provider default and unsupported settings distinct', () => {
  const caps = { reasoning_efforts: ['low', 'high'] }
  expect(resolveModelReasoningEffort('', 'high', caps)).toBe('high')
  expect(resolveModelReasoningEffort('auto', 'high', caps)).toBe('auto')
  expect(resolveModelReasoningEffort('medium', 'high', caps)).toBe('')
  expect(resolveModelReasoningEffort('auto', 'none', caps)).toBe('auto')
  expect(resolveModelReasoningEffort('', 'high')).toBe(resolveReasoningEffort('', 'high'))
})
