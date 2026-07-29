import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $hapticsMuted } from '@/store/haptics'

import { playWakeSound } from './wake-sound'

// Minimal WebAudio doubles: enough to record that playWakeSound wired
// oscillators to the destination when it should, and stayed silent when muted.
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

let oscillators: FakeOscillator[]

class FakeAudioContext {
  state = 'running'
  currentTime = 0
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
}

describe('playWakeSound', () => {
  beforeEach(() => {
    oscillators = []
    $hapticsMuted.set(false)
    vi.stubGlobal('AudioContext', FakeAudioContext)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    $hapticsMuted.set(false)
  })

  it('plays a two-note rising chime when sound is on', () => {
    playWakeSound()

    // G5 then C6 — two enveloped oscillators, both routed onward.
    expect(oscillators).toHaveLength(2)
    expect(oscillators[0].frequency.setValueAtTime).toHaveBeenCalledWith(783.99, expect.any(Number))
    expect(oscillators[1].frequency.setValueAtTime).toHaveBeenCalledWith(1046.5, expect.any(Number))
    for (const osc of oscillators) {
      expect(osc.start).toHaveBeenCalled()
      expect(osc.stop).toHaveBeenCalled()
    }
  })

  it('stays silent when the shared sound-mute toggle is on', () => {
    $hapticsMuted.set(true)
    playWakeSound()
    expect(oscillators).toHaveLength(0)
  })

  it('never throws when WebAudio is unavailable', () => {
    vi.stubGlobal('AudioContext', undefined)
    expect(() => playWakeSound()).not.toThrow()
  })
})
