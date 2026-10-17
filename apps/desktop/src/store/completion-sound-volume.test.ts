// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $completionSoundVolume,
  DEFAULT_COMPLETION_SOUND_VOLUME,
  MAX_COMPLETION_SOUND_VOLUME,
  MIN_COMPLETION_SOUND_VOLUME,
  resolveCompletionSoundVolume,
  setCompletionSoundVolume
} from './completion-sound-volume'

const STORAGE_KEY = 'hermes.desktop.completionSoundVolume'

describe('completion sound volume store', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $completionSoundVolume.set(DEFAULT_COMPLETION_SOUND_VOLUME)
  })

  afterEach(() => {
    window.localStorage.clear()
  })

  it('clamps to the supported range', () => {
    expect(resolveCompletionSoundVolume(5)).toBe(MAX_COMPLETION_SOUND_VOLUME)
    expect(resolveCompletionSoundVolume(-1)).toBe(MIN_COMPLETION_SOUND_VOLUME)
    expect(resolveCompletionSoundVolume(1.5)).toBe(1.5)
  })

  it('falls back to the default for unusable input', () => {
    expect(resolveCompletionSoundVolume(Number.NaN)).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
    expect(resolveCompletionSoundVolume(Number.POSITIVE_INFINITY)).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
  })

  it('persists the chosen volume', () => {
    setCompletionSoundVolume(1.5)
    expect($completionSoundVolume.get()).toBe(1.5)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('1.5')
  })

  it('clamps on write', () => {
    setCompletionSoundVolume(9)
    expect($completionSoundVolume.get()).toBe(MAX_COMPLETION_SOUND_VOLUME)
  })

  it('starts at the shipped default', () => {
    expect(DEFAULT_COMPLETION_SOUND_VOLUME).toBe(1)
    expect(MIN_COMPLETION_SOUND_VOLUME).toBe(0)
    expect($completionSoundVolume.get()).toBe(DEFAULT_COMPLETION_SOUND_VOLUME)
  })
})
