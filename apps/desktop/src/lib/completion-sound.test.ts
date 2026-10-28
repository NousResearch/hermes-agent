// Completion cue behaviour without opening a real audio device: the ownership
// races below only observe scheduling attempts, and the gain/limiter tests drive
// a minimal WebAudio double and read the graph back.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getAudioContext } from '@/lib/audio-context'
import {
  $completionSoundVariantId,
  $completionSoundVolume,
  DEFAULT_COMPLETION_SOUND_VARIANT_ID,
  DEFAULT_COMPLETION_SOUND_VOLUME,
  NO_COMPLETION_SOUND_VARIANT_ID
} from '@/store/completion-sound'
import { $hapticsMuted, setHapticsMuted } from '@/store/haptics'

import { completionMasterGain, playCompletionSound, previewCompletionSound } from './completion-sound'

// Null by default, so the ownership tests never touch a device; the playback
// block swaps in the double below.
vi.mock('@/lib/audio-context', () => ({ getAudioContext: vi.fn(() => null) }))

const original = window.hermesDesktop

afterEach(() => {
  window.hermesDesktop = original
  setHapticsMuted(false)
  $completionSoundVariantId.set(DEFAULT_COMPLETION_SOUND_VARIANT_ID)
  vi.clearAllMocks()
})

it('rechecks mute after ownership resolves, including a failed claim and a later retry', async () => {
  for (const fail of [false, true]) {
    let resolve!: (owns: boolean) => void
    let reject!: (error: Error) => void

    const pending = new Promise<boolean>((yes, no) => {
      resolve = yes
      reject = no
    })

    const claim = vi.fn(() => pending)
    window.hermesDesktop = { ...original, claimAmbientCue: claim }
    setHapticsMuted(false)
    playCompletionSound('session')
    expect(claim).toHaveBeenCalledWith('sound:session')
    setHapticsMuted(true)

    if (fail) {
      reject(new Error('Ownership bridge unavailable'))
    } else {
      resolve(true)
    }

    // Drain the real ownership helper and its completion callback.
    await pending.catch(() => undefined)
    await Promise.resolve()
    expect(getAudioContext).not.toHaveBeenCalled()
    playCompletionSound('session')
    expect(claim).toHaveBeenCalledTimes(1)
  }

  setHapticsMuted(false)
  window.hermesDesktop = { ...original, claimAmbientCue: vi.fn(async () => true) }
  playCompletionSound('session')
  await Promise.resolve()
  await Promise.resolve()
  expect(getAudioContext).toHaveBeenCalledTimes(1)
  expect($hapticsMuted.get()).toBe(false)
})

it('stays silent on None and never claims the cue from an audible peer', () => {
  const claim = vi.fn(async () => true)
  window.hermesDesktop = { ...original, claimAmbientCue: claim }
  $completionSoundVariantId.set(NO_COMPLETION_SOUND_VARIANT_ID)

  playCompletionSound('session')
  previewCompletionSound()

  expect(claim).not.toHaveBeenCalled()
  expect(getAudioContext).not.toHaveBeenCalled()
})

// Minimal WebAudio doubles: enough to read back the master gain the chime was
// built with and to see which node feeds the speakers.
class FakeParam {
  value = 0
  setValueAtTime = vi.fn((value: number) => {
    this.value = value
  })
  exponentialRampToValueAtTime = vi.fn()
}

class FakeNode {
  connect = vi.fn()
}

class FakeGain extends FakeNode {
  gain = new FakeParam()
}

class FakeOscillator extends FakeNode {
  type = 'sine'
  frequency = new FakeParam()
  detune = new FakeParam()
  start = vi.fn()
  stop = vi.fn()
}

class FakeCompressor extends FakeNode {
  threshold = new FakeParam()
  knee = new FakeParam()
  ratio = new FakeParam()
  attack = new FakeParam()
  release = new FakeParam()
}

class FakeFilter extends FakeNode {
  type = 'lowpass'
  frequency = new FakeParam()
  Q = new FakeParam()
}

const destination = {}
let gains: FakeGain[]
let compressors: FakeCompressor[]
let oscillators: FakeOscillator[]

const fakeContext = {
  currentTime: 0,
  sampleRate: 8000,
  destination,
  createGain: () => {
    const node = new FakeGain()
    gains.push(node)

    return node
  },
  createOscillator: () => {
    const node = new FakeOscillator()
    oscillators.push(node)

    return node
  },
  createDynamicsCompressor: () => {
    const node = new FakeCompressor()
    compressors.push(node)

    return node
  },
  createBiquadFilter: () => new FakeFilter(),
  createConvolver: () => new FakeNode(),
  createBuffer: (_channels: number, length: number) => ({ getChannelData: () => new Float32Array(length) })
}

// The master gain is the first node playVariant creates.
const masterGain = () => gains[0]?.gain.value

describe('completionMasterGain', () => {
  it('is silent at zero and rises with every slider step', () => {
    expect(completionMasterGain(0)).toBe(0)

    for (let volume = 1; volume < 100; volume += 1) {
      expect(completionMasterGain(volume + 1)).toBeGreaterThan(completionMasterGain(volume))
    }
  })
})

describe('completion chime playback', () => {
  beforeEach(() => {
    gains = []
    compressors = []
    oscillators = []
    $hapticsMuted.set(false)
    $completionSoundVolume.set(DEFAULT_COMPLETION_SOUND_VOLUME)
    vi.mocked(getAudioContext).mockReturnValue(fakeContext as unknown as AudioContext)
  })

  afterEach(() => {
    $hapticsMuted.set(false)
    $completionSoundVolume.set(DEFAULT_COMPLETION_SOUND_VOLUME)
    vi.mocked(getAudioContext).mockReturnValue(null)
  })

  it('builds the chime louder when the volume slider is higher', () => {
    $completionSoundVolume.set(40)
    previewCompletionSound()
    const quiet = masterGain()

    gains = []

    $completionSoundVolume.set(100)
    previewCompletionSound()

    expect(masterGain()).toBeGreaterThan(quiet as number)
  })

  it('reaches the speakers only through the limiter', () => {
    previewCompletionSound()

    const [limiter] = compressors

    const connectedToDestination = [...gains, ...compressors].filter(node =>
      node.connect.mock.calls.some(([target]) => target === destination)
    )

    expect(connectedToDestination).toEqual([limiter])
  })

  it('previews even when turn-end cues are muted, but the turn-end cue stays silent', () => {
    $hapticsMuted.set(true)

    playCompletionSound()
    expect(oscillators).toHaveLength(0)

    previewCompletionSound()
    expect(oscillators.length).toBeGreaterThan(0)
  })
})
