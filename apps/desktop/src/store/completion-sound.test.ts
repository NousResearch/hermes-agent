import { describe, expect, it, vi } from 'vitest'

import { DEFAULT_COMPLETION_SOUND_VOLUME, resolveCompletionSoundVolume } from './completion-sound'

describe('resolveCompletionSoundVolume', () => {
  it('returns the default when the raw value is below the minimum', () => {
    expect(resolveCompletionSoundVolume(-1)).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
    expect(resolveCompletionSoundVolume(-0.01)).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
  })

  it('returns the default when the raw value is above the maximum', () => {
    expect(resolveCompletionSoundVolume(3.01)).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
    expect(resolveCompletionSoundVolume(100)).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
  })

  it('returns the default for non-finite values', () => {
    expect(resolveCompletionSoundVolume(NaN)).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
    expect(resolveCompletionSoundVolume(Infinity)).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
    expect(resolveCompletionSoundVolume(-Infinity)).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
  })

  it('passes through values within the valid range', () => {
    expect(resolveCompletionSoundVolume(0)).toBe(0)
    expect(resolveCompletionSoundVolume(0.48)).toBe(0.48)
    expect(resolveCompletionSoundVolume(1)).toBe(1)
    expect(resolveCompletionSoundVolume(2.5)).toBe(2.5)
    expect(resolveCompletionSoundVolume(3)).toBe(3)
  })

  it('falls back to the default when nothing is persisted', async () => {
    window.localStorage.clear()
    vi.resetModules()

    const { $completionSoundVolume } = await import('./completion-sound')

    expect($completionSoundVolume.get()).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
  })
})
