import { describe, expect, it } from 'vitest'

import {
  COMPLETION_SOUND_VOLUME_MAX,
  COMPLETION_SOUND_VOLUME_MIN,
  DEFAULT_COMPLETION_SOUND_VOLUME,
  resolveCompletionSoundVolume
} from './completion-sound'

describe('resolveCompletionSoundVolume', () => {
  it('clamps to the slider range and rounds to a whole step', () => {
    expect(resolveCompletionSoundVolume(-5)).toBe(COMPLETION_SOUND_VOLUME_MIN)
    expect(resolveCompletionSoundVolume(250)).toBe(COMPLETION_SOUND_VOLUME_MAX)
    expect(resolveCompletionSoundVolume(42.6)).toBe(43)
  })

  it('falls back to the default for values that are not numbers', () => {
    expect(resolveCompletionSoundVolume(Number.NaN)).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
    expect(resolveCompletionSoundVolume(Number.POSITIVE_INFINITY)).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
  })
})
