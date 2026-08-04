import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/store/haptics', () => ({ $hapticsMuted: { get: vi.fn(() => false) } }))
vi.mock('@/hermes', () => ({
  getHermesConfigRecord: vi.fn(async () => ({})),
  saveHermesConfig: vi.fn(async () => undefined)
}))

import { $hapticsMuted } from '@/store/haptics'
import { $thinkingSoundEnabled } from '@/store/voice-prefs'

import { isThinkingSoundActive, startThinkingSound, stopThinkingSound } from './thinking-sound'

class FakeOscillator {
  type = 'sine'
  frequency = { exponentialRampToValueAtTime: vi.fn(), setValueAtTime: vi.fn() }
  connect = vi.fn()
  start = vi.fn()
  stop = vi.fn()
}

class FakeGain {
  gain = { exponentialRampToValueAtTime: vi.fn(), setValueAtTime: vi.fn() }
  connect = vi.fn()
}

const started: FakeOscillator[] = []

class FakeAudioContext {
  currentTime = 0
  destination = {}
  state = 'running'

  createOscillator() {
    const osc = new FakeOscillator()

    started.push(osc)

    return osc
  }

  createGain() {
    return new FakeGain()
  }

  resume() {
    return Promise.resolve()
  }
}

describe('thinking-sound', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    started.length = 0
    $thinkingSoundEnabled.set(true)
    vi.mocked($hapticsMuted.get).mockReturnValue(false)
    vi.stubGlobal('AudioContext', FakeAudioContext)
  })

  afterEach(() => {
    stopThinkingSound()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('plays repeating blips while active and stops instantly', () => {
    startThinkingSound()
    expect(isThinkingSoundActive()).toBe(true)

    vi.advanceTimersByTime(5_000)
    expect(started.length).toBeGreaterThanOrEqual(3)

    const count = started.length

    stopThinkingSound()
    expect(isThinkingSoundActive()).toBe(false)
    vi.advanceTimersByTime(5_000)
    expect(started.length).toBe(count) // nothing after stop
  })

  it('respects the voice.thinking_sound config gate', () => {
    $thinkingSoundEnabled.set(false)
    startThinkingSound()
    expect(isThinkingSoundActive()).toBe(false)
    vi.advanceTimersByTime(3_000)
    expect(started.length).toBe(0)
  })

  it('stays silent while sounds are muted but keeps the loop alive', () => {
    vi.mocked($hapticsMuted.get).mockReturnValue(true)
    startThinkingSound()
    vi.advanceTimersByTime(3_000)
    expect(started.length).toBe(0)

    // Unmute mid-loop → blips resume without a restart.
    vi.mocked($hapticsMuted.get).mockReturnValue(false)
    vi.advanceTimersByTime(3_000)
    expect(started.length).toBeGreaterThan(0)
  })

  it('start is idempotent', () => {
    startThinkingSound()
    startThinkingSound()
    vi.advanceTimersByTime(1_300)

    // One loop: at most ~2 blips in 1.3s (first at 400ms, next ≥800ms later).
    expect(started.length).toBeLessThanOrEqual(2)
  })

  it('never throws when WebAudio is unavailable', () => {
    vi.stubGlobal('AudioContext', undefined)
    expect(() => {
      startThinkingSound()
      vi.advanceTimersByTime(2_000)
    }).not.toThrow()
    stopThinkingSound()
  })
})
