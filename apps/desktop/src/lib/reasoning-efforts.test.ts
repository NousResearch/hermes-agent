import { describe, expect, it } from 'vitest'

import {
  LEGACY_REASONING_DEFAULT_OPTIONS,
  normalizeReasoningEffort,
  normalizeReasoningEffortForRequest,
  reasoningCanDisable,
  reasoningEffortOptions
} from './reasoning-efforts'

describe('reasoning effort capabilities', () => {
  it('uses provider/model-specific efforts exactly when supplied', () => {
    const options = reasoningEffortOptions(['low', 'medium', 'high', 'xhigh'], LEGACY_REASONING_DEFAULT_OPTIONS)

    expect(options).toEqual(['low', 'medium', 'high', 'xhigh'])
    expect(reasoningCanDisable(options)).toBe(false)
    expect(normalizeReasoningEffort('minimal', options)).toBe('medium')
    expect(normalizeReasoningEffort('none', options)).toBe('')
    expect(normalizeReasoningEffort('extra high', options)).toBe('xhigh')
  })

  it('falls back to the legacy desktop default surface only when capabilities are absent', () => {
    expect(reasoningEffortOptions(undefined, LEGACY_REASONING_DEFAULT_OPTIONS)).toEqual([
      'none',
      'minimal',
      'low',
      'medium',
      'high',
      'xhigh'
    ])
  })

  it('normalizes stale presets before applying them to a live session', () => {
    const gpt55 = ['low', 'medium', 'high', 'xhigh']

    expect(normalizeReasoningEffortForRequest('minimal', gpt55)).toBe('medium')
    expect(normalizeReasoningEffortForRequest('max', gpt55)).toBe('xhigh')
    expect(normalizeReasoningEffortForRequest('none', gpt55)).toBe('medium')
  })
})
