import { beforeEach, describe, expect, it, vi } from 'vitest'

import { consumeContinuationPrompt, rememberContinuationPrompt } from './continuation-recovery'

beforeEach(() => {
  window.localStorage.clear()
})

describe('continuation recovery prompt bridge', () => {
  it('persists the prompt so renderer reloads do not lose recovery', async () => {
    rememberContinuationPrompt('rt-reload', 'recover after reload')

    vi.resetModules()
    const reloaded = await import('./continuation-recovery')

    expect(reloaded.consumeContinuationPrompt('rt-reload')).toBe('recover after reload')
    expect(reloaded.consumeContinuationPrompt('rt-reload')).toBeNull()
  })

  it('returns the last prompt once for a session', () => {
    rememberContinuationPrompt('rt-1', 'first prompt')
    rememberContinuationPrompt('rt-1', 'latest prompt')

    expect(consumeContinuationPrompt('rt-1')).toBe('latest prompt')
    expect(consumeContinuationPrompt('rt-1')).toBeNull()
  })
})
