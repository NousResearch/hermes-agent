// @vitest-environment jsdom
/**
 * The speaking probe already computes per-frame amplitude (`peak`); this pins
 * that the amplitude actually reaches the handlers — normalized to 0..1 and
 * varying with the signal — instead of being discarded.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { VoiceLiveSession } from '@/lib/voice-live'

class FakeAnalyser {
  frequencyBinCount = 4

  fftSize = 512

  samples = new Uint8Array([128, 128, 128, 128])

  getByteTimeDomainData(buffer: Uint8Array): void {
    buffer.set(this.samples)
  }

  disconnect(): void {
    // Nothing to tear down in the double.
  }
}

class FakeAudioContext {
  analyser = new FakeAnalyser()

  createMediaStreamSource(): { connect: () => void } {
    return { connect: () => undefined }
  }

  createAnalyser(): FakeAnalyser {
    return this.analyser
  }

  close(): Promise<void> {
    return Promise.resolve()
  }
}

describe('speaking probe amplitude', () => {
  let originalAudioContext: typeof AudioContext | undefined

  beforeEach(() => {
    vi.useFakeTimers()
    originalAudioContext = (globalThis as { AudioContext?: typeof AudioContext }).AudioContext
    ;(globalThis as { AudioContext?: unknown }).AudioContext = FakeAudioContext
  })

  afterEach(() => {
    vi.useRealTimers()
    ;(globalThis as { AudioContext?: unknown }).AudioContext = originalAudioContext
  })

  it('reports a per-frame level that tracks the real amplitude', () => {
    const changes: Array<{ level: number; speaking: boolean }> = []
    const levels: number[] = []

    const session = new VoiceLiveSession({
      onClosed: () => undefined,
      onDelegation: () => undefined,
      onError: () => undefined,
      onSpeakingChange: (speaking, level) => {
        changes.push({ level, speaking })
      },
      onSpeakingLevel: level => {
        levels.push(level)
      }
    })

    const internals = session as unknown as {
      analyser: FakeAnalyser
      armSpeakingProbe: (stream: MediaStream) => void
    }

    internals.armSpeakingProbe({} as MediaStream)

    // Soft voice: deviation 10 from silence, just above the >6 loud gate.
    internals.analyser.samples = new Uint8Array([138, 118, 138, 118])
    vi.advanceTimersByTime(100)

    // Loud voice: deviation 60.
    internals.analyser.samples = new Uint8Array([188, 68, 188, 68])
    vi.advanceTimersByTime(100)

    session.close()

    expect(levels).toHaveLength(2)
    expect(levels[0]).toBeGreaterThan(0)
    expect(levels[1]).toBeGreaterThan(levels[0]!)
    expect(levels[1]).toBeLessThanOrEqual(1)

    // The boolean speaking edge is unchanged and carries the same amplitude.
    expect(changes).toHaveLength(1)
    expect(changes[0]).toEqual({ level: levels[0], speaking: true })
  })
})
