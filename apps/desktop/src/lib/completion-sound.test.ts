import { beforeEach, describe, expect, it, vi } from 'vitest'

describe('completion sound playback', () => {
  beforeEach(() => {
    vi.resetModules()
    window.localStorage.clear()
  })

  it('keeps turn-end cues silent even when a chime preset is stored', async () => {
    window.localStorage.setItem('hermes.desktop.completionSoundLegacyDefaultMigrated', 'true')
    window.localStorage.setItem('hermes.desktop.completionSoundVariantId', '1')

    const AudioContextMock = vi.fn(() => {
      throw new Error('turn-end audio should stay disabled')
    })

    Object.defineProperty(window, 'AudioContext', {
      configurable: true,
      value: AudioContextMock
    })

    const { playCompletionSound } = await import('./completion-sound')

    playCompletionSound()

    expect(AudioContextMock).not.toHaveBeenCalled()
  })
})
