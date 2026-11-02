import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  bargeInTriggerLevels,
  DEFAULT_BARGE_IN_THRESHOLD_MULTIPLIER,
  monitorSpeechDuringPlayback
} from './voice-barge-in'

// Drive the real monitor with a fake mic: `micLevel` is the byte-domain RMS
// level the analyser reports, `playing` is the TTS-flowing flag, and each
// `advance(ms)` runs one animation frame per 16ms of simulated time.
let now = 0
let micLevel = 0
let frames: FrameRequestCallback[] = []

beforeEach(() => {
  now = 1_000_000
  micLevel = 0
  frames = []
  vi.spyOn(Date, 'now').mockImplementation(() => now)
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => frames.push(cb))
  vi.stubGlobal('cancelAnimationFrame', () => undefined)
  vi.stubGlobal(
    'AudioContext',
    class {
      createAnalyser() {
        return {
          fftSize: 256,
          getByteTimeDomainData: (data: Uint8Array) => data.fill(128 + Math.round(micLevel * 42))
        }
      }

      createMediaStreamSource() {
        return { connect: () => undefined }
      }

      close() {
        return Promise.resolve()
      }
    }
  )
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: async () => ({ getTracks: () => [{ stop: () => undefined }] }) }
  })
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

async function flushMicrotasks() {
  for (let i = 0; i < 5; i += 1) {
    await Promise.resolve()
  }
}

function advance(ms: number) {
  for (let elapsed = 0; elapsed < ms; elapsed += 16) {
    now += 16
    const pending = frames
    frames = []
    pending.forEach(cb => cb(now))
  }
}

/** Quiet room, then TTS starts and the user talks over it at `speechLevel`. */
async function talkOverPlayback(speechLevel: number, thresholdMultiplier?: number | null) {
  let playing = false
  const onSpeech = vi.fn()
  const stop = monitorSpeechDuringPlayback({ isPlaying: () => playing, onSpeech, thresholdMultiplier })

  await flushMicrotasks()
  advance(600) // calibrate the quiet floor
  playing = true
  advance(700) // past the playback-onset grace window
  micLevel = speechLevel
  advance(800)
  stop()

  return onSpeech.mock.calls.length > 0
}

describe('monitorSpeechDuringPlayback — voice.barge_in_threshold_multiplier (#117801)', () => {
  // ~0.095: a Bluetooth HFP headset's speech level during playback, below the
  // stock playback clamp.
  const HFP_SPEECH_LEVEL = 0.095

  it('a quiet headset cannot interrupt playback at the stock sensitivity', async () => {
    expect(await talkOverPlayback(HFP_SPEECH_LEVEL)).toBe(false)
    expect(await talkOverPlayback(HFP_SPEECH_LEVEL, DEFAULT_BARGE_IN_THRESHOLD_MULTIPLIER)).toBe(false)
  })

  it('a lower configured multiplier lets the same headset interrupt playback', async () => {
    expect(await talkOverPlayback(HFP_SPEECH_LEVEL, 1.5)).toBe(true)
  })

  it('a higher configured multiplier makes playback harder to interrupt', async () => {
    const loud = bargeInTriggerLevels().playbackMinTrigger + 0.03

    expect(await talkOverPlayback(loud)).toBe(true)
    expect(await talkOverPlayback(loud, DEFAULT_BARGE_IN_THRESHOLD_MULTIPLIER * 2)).toBe(false)
  })

  it('invalid multipliers keep the stock trigger levels', () => {
    for (const value of [null, undefined, 0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(bargeInTriggerLevels(value)).toEqual(bargeInTriggerLevels(DEFAULT_BARGE_IN_THRESHOLD_MULTIPLIER))
    }
  })
})
