import { describe, expect, it } from 'vitest'

import { consumeContinuationPrompt, rememberContinuationPrompt } from './continuation-recovery'

describe('continuation recovery prompt bridge', () => {
  it('returns the last prompt once for a session', () => {
    rememberContinuationPrompt('rt-1', 'first prompt')
    rememberContinuationPrompt('rt-1', 'latest prompt')

    expect(consumeContinuationPrompt('rt-1')).toBe('latest prompt')
    expect(consumeContinuationPrompt('rt-1')).toBeNull()
  })
})
