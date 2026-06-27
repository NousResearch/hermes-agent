import { describe, expect, it } from 'vitest'

import { resolveModelReasoningPolicy } from './model-picker-policy'

describe('model picker reasoning policy', () => {
  it('keeps the legacy medium adjustable fallback for catalog models', () => {
    expect(
      resolveModelReasoningPolicy({ fast: false, reasoning: true }, { currentEffort: '', isCurrent: false })
    ).toEqual({ configurable: true, defaultEffort: 'medium', effort: 'medium', label: '' })
  })

  it('uses a configured default instead of inventing medium', () => {
    const policy = resolveModelReasoningPolicy(
      { fast: false, reasoning: true, reasoning_configurable: true, reasoning_default: 'high' },
      { currentEffort: '', isCurrent: false }
    )

    expect(policy.effort).toBe('high')
    expect(policy.configurable).toBe(true)
  })

  it('represents an externally fixed policy without an editable control', () => {
    expect(
      resolveModelReasoningPolicy(
        {
          fast: false,
          reasoning: true,
          reasoning_configurable: false,
          reasoning_default: 'high',
          reasoning_label: 'High reasoning'
        },
        { currentEffort: 'medium', isCurrent: true }
      )
    ).toEqual({ configurable: false, defaultEffort: 'high', effort: 'medium', label: 'High reasoning' })
  })
})
