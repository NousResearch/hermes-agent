import { describe, expect, it } from 'vitest'

import {
  ENABLED_REASONING_EFFORTS,
  isEnabledReasoningEffort,
  isReasoningEffort,
  normalizeEnabledReasoningEffort,
  REASONING_EFFORTS
} from './reasoning-effort'

describe('reasoning effort contract', () => {
  it('exposes the enabled reasoning efforts in Hermes order', () => {
    expect(ENABLED_REASONING_EFFORTS).toEqual(['minimal', 'low', 'medium', 'high', 'xhigh', 'max'])
  })

  it('recognizes none plus every enabled reasoning effort', () => {
    expect(REASONING_EFFORTS).toEqual(['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'])
    expect(isReasoningEffort('none')).toBe(true)
    expect(isReasoningEffort('max')).toBe(true)
    expect(isReasoningEffort(' MAX ')).toBe(false)
    expect(isReasoningEffort('show')).toBe(false)
  })

  it('distinguishes enabled efforts without aliasing xhigh and max', () => {
    expect(isEnabledReasoningEffort('xhigh')).toBe(true)
    expect(isEnabledReasoningEffort('max')).toBe(true)
    expect(isEnabledReasoningEffort('none')).toBe(false)
    expect(isEnabledReasoningEffort(' XHIGH ')).toBe(false)
    expect(normalizeEnabledReasoningEffort(' XHIGH ')).toBe('xhigh')
    expect(normalizeEnabledReasoningEffort(' max ')).toBe('max')
  })

  it('falls back to medium for empty or unknown values', () => {
    expect(normalizeEnabledReasoningEffort('')).toBe('medium')
    expect(normalizeEnabledReasoningEffort(undefined)).toBe('medium')
    expect(normalizeEnabledReasoningEffort('show')).toBe('medium')
  })
})
