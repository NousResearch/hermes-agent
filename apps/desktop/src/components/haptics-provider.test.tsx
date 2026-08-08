import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { triggerHaptic } from '@/lib/haptics'
import { setHapticsMuted } from '@/store/haptics'

import { HapticsProvider } from './haptics-provider'

// web-haptics computes `WebHaptics.isSupported` (typeof navigator.vibrate)
// once at module load. Desktop Chromium exposes navigator.vibrate as a
// no-op function, so simulate that BEFORE the library module is imported —
// otherwise jsdom's missing Vibration API flips the library onto its
// DOM-only fallback and the mute test below wouldn't exercise the real
// desktop path.
const { vibrateMock } = vi.hoisted(() => {
  const vibrateMock = vi.fn()
  Object.defineProperty(navigator, 'vibrate', { configurable: true, value: vibrateMock })

  return { vibrateMock }
})

class MockAudioContext {
  static constructed = 0

  destination = {}
  sampleRate = 44100
  state = 'running'

  constructor() {
    MockAudioContext.constructed++
  }

  close() {
    return Promise.resolve()
  }

  createBiquadFilter() {
    return { Q: { value: 0 }, connect: () => undefined, frequency: { value: 0 }, type: '' }
  }

  createBuffer(channels: number, length: number) {
    return { getChannelData: () => new Float32Array(length) }
  }

  createBufferSource() {
    return {
      buffer: null,
      connect: () => undefined,
      disconnect: () => undefined,
      onended: null,
      start: () => undefined
    }
  }

  createGain() {
    return { connect: () => undefined, gain: { value: 0 } }
  }

  resume() {
    return Promise.resolve()
  }
}

describe('HapticsProvider', () => {
  beforeEach(() => {
    MockAudioContext.constructed = 0
    setHapticsMuted(false)
    vi.stubGlobal('AudioContext', MockAudioContext)
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('never constructs an AudioContext when a haptic fires', () => {
    render(<HapticsProvider>{null}</HapticsProvider>)

    triggerHaptic('submit')

    expect(MockAudioContext.constructed).toBe(0)
  })

  it('registers a live trigger and keeps the persisted mute control functional', async () => {
    render(<HapticsProvider>{null}</HapticsProvider>)

    triggerHaptic('submit')
    expect(vibrateMock).toHaveBeenCalledTimes(1)

    await act(async () => {
      setHapticsMuted(true)
    })

    triggerHaptic('submit')
    expect(vibrateMock).toHaveBeenCalledTimes(1)
  })

  it('does not warm an AudioContext at idle on mount', () => {
    vi.stubGlobal('requestIdleCallback', (callback: () => void) => {
      callback()

      return 0
    })
    vi.stubGlobal('cancelIdleCallback', () => undefined)

    render(<HapticsProvider>{null}</HapticsProvider>)

    expect(MockAudioContext.constructed).toBe(0)
  })
})
