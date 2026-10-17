import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $completionSoundVariantId,
  DEFAULT_COMPLETION_SOUND_VARIANT_ID,
  NO_COMPLETION_SOUND_VARIANT_ID,
  resolveCompletionSoundVariantId
} from '@/store/completion-sound'
import { $hapticsMuted } from '@/store/haptics'

import { playCompletionSound, previewCompletionSound } from './completion-sound'

const STORAGE_KEY = 'hermes.desktop.completionSoundVariantId'

const loadStore = () => import('@/store/completion-sound')

// Minimal WebAudio doubles (same technique as wake-sound.test.ts): enough to
// record that playCompletionSound wired oscillators to the destination when it
// should, and stayed silent when variant 0 or the mute toggle says so.
class FakeParam {
  setValueAtTime = vi.fn()
  exponentialRampToValueAtTime = vi.fn()
}

class FakeOscillator {
  type = 'sine'
  frequency = new FakeParam()
  connect = vi.fn()
  start = vi.fn()
  stop = vi.fn()
}

class FakeGain {
  gain = new FakeParam()
  connect = vi.fn()
}

class FakeBiquadFilter {
  type: BiquadFilterType = 'lowpass'
  frequency = new FakeParam()
  Q = new FakeParam()
  connect = vi.fn()
}

class FakeAudioBuffer {
  // Nothing reads the samples back; the write loops just need a home.
  getChannelData(): number[] {
    return []
  }
}

class FakeConvolver {
  buffer: FakeAudioBuffer | null = null
  connect = vi.fn()
}

let oscillators: FakeOscillator[]

class FakeAudioContext {
  state = 'running'
  currentTime = 0
  sampleRate = 48_000
  destination = {}
  resume = vi.fn().mockResolvedValue(undefined)

  createOscillator() {
    const osc = new FakeOscillator()
    oscillators.push(osc)

    return osc
  }

  createGain() {
    return new FakeGain()
  }

  createBiquadFilter() {
    return new FakeBiquadFilter()
  }

  createConvolver() {
    return new FakeConvolver()
  }

  createBuffer() {
    return new FakeAudioBuffer()
  }
}

describe('resolveCompletionSoundVariantId', () => {
  it('accepts 0 (none) and every preset id 1..14', () => {
    expect(resolveCompletionSoundVariantId(NO_COMPLETION_SOUND_VARIANT_ID)).toBe(0)

    for (let id = 1; id <= 14; id += 1) {
      expect(resolveCompletionSoundVariantId(id)).toBe(id)
    }
  })

  it('falls back to the default for out-of-range or garbage values', () => {
    for (const bad of [-1, 15, Number.NaN, 1.5]) {
      expect(resolveCompletionSoundVariantId(bad)).toBe(DEFAULT_COMPLETION_SOUND_VARIANT_ID)
    }
  })
})

describe('completion-sound store', () => {
  // The local jsdom/Node env can hand back a half-implemented localStorage
  // (no clear()), so drive the store against a Map-backed double instead —
  // the same stubGlobal technique as layout-persistence.test.ts.
  const makeLocalStorage = () => {
    const values = new Map<string, string>()

    return {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => void values.set(key, value),
      removeItem: (key: string) => void values.delete(key),
      clear: () => values.clear()
    }
  }

  let localStorage: ReturnType<typeof makeLocalStorage>

  beforeEach(() => {
    localStorage = makeLocalStorage()
    vi.stubGlobal('localStorage', localStorage)
    vi.resetModules()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('defaults to variant 1 when nothing is stored', async () => {
    expect((await loadStore()).$completionSoundVariantId.get()).toBe(DEFAULT_COMPLETION_SOUND_VARIANT_ID)
  })

  it('loads a stored "0" as variant 0 (None)', async () => {
    localStorage.setItem(STORAGE_KEY, '0')

    expect((await loadStore()).$completionSoundVariantId.get()).toBe(NO_COMPLETION_SOUND_VARIANT_ID)
  })

  it('persists "0" when the None variant is selected', async () => {
    const store = await loadStore()

    store.setCompletionSoundVariantId(NO_COMPLETION_SOUND_VARIANT_ID)

    expect(store.$completionSoundVariantId.get()).toBe(NO_COMPLETION_SOUND_VARIANT_ID)
    expect(localStorage.getItem(STORAGE_KEY)).toBe('0')
  })

  it('falls back to variant 1 for stored garbage', async () => {
    localStorage.setItem(STORAGE_KEY, 'loud')

    expect((await loadStore()).$completionSoundVariantId.get()).toBe(DEFAULT_COMPLETION_SOUND_VARIANT_ID)
  })
})

describe('playCompletionSound', () => {
  beforeEach(() => {
    oscillators = []
    $hapticsMuted.set(false)
    $completionSoundVariantId.set(DEFAULT_COMPLETION_SOUND_VARIANT_ID)
    vi.stubGlobal('AudioContext', FakeAudioContext)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    $hapticsMuted.set(false)
    $completionSoundVariantId.set(DEFAULT_COMPLETION_SOUND_VARIANT_ID)
  })

  it('plays the default variant audibly', () => {
    playCompletionSound()

    expect(oscillators.length).toBeGreaterThan(0)
  })

  it('stays silent when variant 0 (None) is selected', () => {
    $completionSoundVariantId.set(NO_COMPLETION_SOUND_VARIANT_ID)

    playCompletionSound()

    expect(oscillators).toHaveLength(0)
  })

  it('stays silent when the shared sound-mute toggle is on', () => {
    $hapticsMuted.set(true)

    playCompletionSound()

    expect(oscillators).toHaveLength(0)
  })

  it('previews silently for variant 0', () => {
    previewCompletionSound(NO_COMPLETION_SOUND_VARIANT_ID)

    expect(oscillators).toHaveLength(0)
  })

  // The variant-0 bail must come before the ownsAmbientCue claim, or a silent
  // window would steal the turn-end cue from an audible peer.
  it('claims the ambient cue only when it would be audible', () => {
    const claimAmbientCue = vi.fn(async () => false)

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { claimAmbientCue }

    try {
      $completionSoundVariantId.set(NO_COMPLETION_SOUND_VARIANT_ID)
      playCompletionSound('session-1')
      expect(claimAmbientCue).not.toHaveBeenCalled()

      $completionSoundVariantId.set(DEFAULT_COMPLETION_SOUND_VARIANT_ID)
      playCompletionSound('session-1')
      expect(claimAmbientCue).toHaveBeenCalledTimes(1)
    } finally {
      delete (window as { hermesDesktop?: unknown }).hermesDesktop
    }
  })
})
