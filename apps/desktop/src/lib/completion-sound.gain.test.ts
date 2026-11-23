// Proves the one line the volume setting actually controls: the gain the
// completion cue is scheduled with. Every other completion-sound suite mocks
// getAudioContext to null, so without this file the gain could be reverted to a
// constant — or pinned to 0 — and the whole suite would stay green.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  DEFAULT_COMPLETION_SOUND_VOLUME,
  setCompletionSoundVariantId,
  setCompletionSoundVolume
} from '@/store/completion-sound'

const audio = vi.hoisted(() => ({
  ac: null as null | Record<string, unknown>,
  gains: [] as { gain: { setValueAtTime: ReturnType<typeof vi.fn> }; connect: ReturnType<typeof vi.fn> }[]
}))

vi.mock('@/lib/audio-context', () => ({ getAudioContext: () => audio.ac }))

const param = () => ({ setValueAtTime: vi.fn(), exponentialRampToValueAtTime: vi.fn() })

// Only the nodes playVariant actually touches: master/dry/wet gains, the
// low-pass, one oscillator per voice, and the reverb buffer + convolver.
// sampleRate stays small so the noise-filled impulse buffer costs nothing.
function fakeContext() {
  return {
    currentTime: 0,
    sampleRate: 400,
    createBuffer: (channels: number, length: number) => ({
      getChannelData: () => new Float32Array(length),
      numberOfChannels: channels
    }),
    createBiquadFilter: () => ({ type: 'lowpass', frequency: param(), Q: param(), connect: vi.fn() }),
    createConvolver: () => ({ buffer: null, connect: vi.fn() }),
    createGain: () => {
      const node = { gain: param(), connect: vi.fn() }

      audio.gains.push(node)

      return node
    },
    createOscillator: () => ({ type: 'sine', frequency: param(), connect: vi.fn(), start: vi.fn(), stop: vi.fn() })
  }
}

const importLib = () => import('@/lib/completion-sound')

beforeEach(() => {
  window.localStorage.clear()
  audio.gains.length = 0
  audio.ac = fakeContext()
  setCompletionSoundVariantId(1)
  setCompletionSoundVolume(DEFAULT_COMPLETION_SOUND_VOLUME)
})

afterEach(() => {
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('completion cue master gain', () => {
  it('schedules the stored default, so an untouched install sounds exactly as it did before', async () => {
    const { playCompletionSound } = await importLib()

    playCompletionSound()

    // createGain runs first for the master, then one env gain per voice.
    expect(audio.gains[0].gain.setValueAtTime).toHaveBeenCalledWith(DEFAULT_COMPLETION_SOUND_VOLUME, 0)
  })

  it('follows a changed volume instead of a constant', async () => {
    const { playCompletionSound } = await importLib()

    setCompletionSoundVolume(1.5)
    playCompletionSound()

    expect(audio.gains[0].gain.setValueAtTime).toHaveBeenCalledWith(1.5, 0)
  })

  it('schedules nothing at 0.00', async () => {
    const { playCompletionSound, previewCompletionSound } = await importLib()

    setCompletionSoundVolume(0)
    playCompletionSound()
    previewCompletionSound()

    expect(audio.gains).toHaveLength(0)
  })

  it('does not claim the cross-window cue while silent, so an audible window still gets it', async () => {
    const { playCompletionSound } = await importLib()

    const claim = vi.fn(async () => true)

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { claimAmbientCue: claim }

    setCompletionSoundVolume(0)
    playCompletionSound('session-1')
    expect(claim).not.toHaveBeenCalled()

    setCompletionSoundVolume(1)
    playCompletionSound('session-1')
    await Promise.resolve()
    await Promise.resolve()

    expect(claim).toHaveBeenCalledWith('sound:session-1')
    expect(audio.gains[0].gain.setValueAtTime).toHaveBeenCalledWith(1, 0)
  })
})
