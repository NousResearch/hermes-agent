// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/store/haptics', () => ({ $hapticsMuted: { get: vi.fn(() => false) } }))
vi.mock('@/store/ambient', () => ({ ownsAmbientCue: vi.fn(async () => true) }))
vi.mock('@/hermes', () => ({
  getHermesConfigRecord: vi.fn(async () => ({})),
  saveHermesConfig: vi.fn(async () => undefined)
}))

import { $completionSoundVolume } from '@/store/completion-sound-volume'

import { COMPLETION_SOUND_BASE_GAIN, previewCompletionSound } from './completion-sound'

class FakeParam {
  exponentialRampToValueAtTime = vi.fn()
  linearRampToValueAtTime = vi.fn()
  setValueAtTime = vi.fn()
}

class FakeGain {
  connect = vi.fn()
  gain = new FakeParam()
}

class FakeOscillator {
  connect = vi.fn()
  detune = new FakeParam()
  frequency = new FakeParam()
  start = vi.fn()
  stop = vi.fn()
  type = 'sine'
}

class FakeFilter {
  connect = vi.fn()
  frequency = new FakeParam()
  Q = new FakeParam()
  type = 'lowpass'
}

class FakeConvolver {
  buffer: unknown = null
  connect = vi.fn()
}

class FakeSource {
  buffer: unknown = null
  connect = vi.fn()
  start = vi.fn()
  stop = vi.fn()
}

class FakeBuffer {
  private readonly data: Float32Array

  constructor(length: number) {
    this.data = new Float32Array(length)
  }

  getChannelData() {
    return this.data
  }
}

class FakeAudioContext {
  currentTime = 0
  destination = {}
  gains: FakeGain[] = []
  sampleRate = 8
  state = 'running'

  constructor() {
    contexts.push(this)
  }

  createBiquadFilter() {
    return new FakeFilter()
  }

  createBuffer(_channels: number, length: number) {
    return new FakeBuffer(length)
  }

  createBufferSource() {
    return new FakeSource()
  }

  createConvolver() {
    return new FakeConvolver()
  }

  createGain() {
    const gain = new FakeGain()

    this.gains.push(gain)

    return gain
  }

  createOscillator() {
    return new FakeOscillator()
  }

  resume() {
    return Promise.resolve()
  }
}

const contexts: FakeAudioContext[] = []

/** The module caches one AudioContext; the first createGain() of a play is the master. */
function activeContext(): FakeAudioContext {
  const ctx = contexts.at(-1)

  if (!ctx) {
    throw new Error('no AudioContext was created')
  }

  return ctx
}

describe('completion-sound loudness', () => {
  beforeEach(() => {
    vi.stubGlobal('AudioContext', FakeAudioContext)
    $completionSoundVolume.set(1)
    previewCompletionSound(1) // warms the module's cached context
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('keeps the shipped base gain above the previous level', () => {
    // Keep the intentional increase independent of a particular output device.
    expect(COMPLETION_SOUND_BASE_GAIN).toBeGreaterThanOrEqual(2)
  })

  it('scales the master gain by the stored volume', () => {
    const ctx = activeContext()
    const before = ctx.gains.length

    previewCompletionSound(1)

    expect(ctx.gains.length).toBeGreaterThan(before)
    expect(ctx.gains[before].gain.setValueAtTime).toHaveBeenCalledWith(COMPLETION_SOUND_BASE_GAIN, 0)
  })

  it('doubles the master gain at volume 2', () => {
    const ctx = activeContext()
    const before = ctx.gains.length

    $completionSoundVolume.set(2)
    previewCompletionSound(1)

    expect(ctx.gains[before].gain.setValueAtTime).toHaveBeenCalledWith(COMPLETION_SOUND_BASE_GAIN * 2, 0)
  })

  it('goes silent at volume 0 without throwing', () => {
    const ctx = activeContext()
    const before = ctx.gains.length

    $completionSoundVolume.set(0)

    expect(() => previewCompletionSound(1)).not.toThrow()
    expect(ctx.gains[before].gain.setValueAtTime).toHaveBeenCalledWith(0, 0)
  })

  it('never throws when WebAudio is unavailable', async () => {
    vi.stubGlobal('AudioContext', undefined)
    vi.stubGlobal('webkitAudioContext', undefined)
    // Drop the already-warmed AudioContext so this exercises the missing-API path.
    vi.resetModules()
    const fresh = await import('./completion-sound')
    const before = contexts.length

    expect(() => fresh.previewCompletionSound(1)).not.toThrow()
    expect(contexts).toHaveLength(before)
  })
})
