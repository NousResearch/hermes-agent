import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  armDiagnostics,
  disarmDiagnostics,
  isDiagnosticsArmed,
  type LongFrameEvent,
  readDiagnosticsCapture,
  recordStreamDeltaApplied
} from './capture'

// jsdom ships no PerformanceObserver, and even where it does there is no way to
// provoke a real Long Animation Frame in a test environment. Stand in for the
// engine: count constructions (the "zero observers when disarmed" claim) and
// feed a synthetic LoAF entry shaped exactly like Chromium's.
class FakePerformanceObserver {
  static supportedEntryTypes = ['long-animation-frame', 'longtask']
  static instances: FakePerformanceObserver[] = []

  observed: PerformanceObserverInit[] = []
  disconnected = false

  constructor(private readonly callback: PerformanceObserverCallback) {
    FakePerformanceObserver.instances.push(this)
  }

  observe(options: PerformanceObserverInit) {
    this.observed.push(options)
  }

  disconnect() {
    this.disconnected = true
  }

  takeRecords() {
    return [] as PerformanceEntryList
  }

  emit(entries: unknown[]) {
    this.callback(
      { getEntries: () => entries as PerformanceEntryList } as PerformanceObserverEntryList,
      this as unknown as PerformanceObserver
    )
  }
}

// A 200ms long animation frame whose script attribution names a bundle behind a
// deep path — the path must not survive into the buffer.
const longAnimationFrameEntry = {
  blockingDuration: 154,
  duration: 204,
  scripts: [
    {
      duration: 198,
      invoker: 'TIMER',
      invokerType: 'user-callback',
      sourceFunctionName: 'runFlush',
      sourceURL: 'https://app.local/home/PRIVATE_MARKER/assets/index-abc123.js'
    },
    // Below the 5ms attribution floor: noise in every real frame.
    { duration: 1, invoker: 'IntersectionObserver', invokerType: 'user-callback', sourceURL: 'https://app.local/x.js' }
  ],
  startTime: 1_000,
  styleAndLayoutStart: 1_180
}

const originalPerformanceObserver = globalThis.PerformanceObserver

describe('renderer diagnostics capture', () => {
  beforeEach(() => {
    FakePerformanceObserver.instances = []
    globalThis.PerformanceObserver = FakePerformanceObserver as unknown as typeof PerformanceObserver
  })

  afterEach(() => {
    disarmDiagnostics()
    globalThis.PerformanceObserver = originalPerformanceObserver
    vi.restoreAllMocks()
  })

  it('registers no observers and records nothing while disarmed', () => {
    expect(isDiagnosticsArmed()).toBe(false)

    // The per-delta entry point on the stream flush path: a single boolean read
    // and out, with nowhere for the sample to land.
    expect(recordStreamDeltaApplied({ historyMessages: 40, queuedChars: 12, sessions: 1, writeMs: 3 })).toBeNull()

    expect(FakePerformanceObserver.instances).toHaveLength(0)
    expect(readDiagnosticsCapture()).toBeNull()
  })

  it('registers the LoAF observer on the arm edge and tears it down on disarm', () => {
    armDiagnostics('capture-1', 1_700_000_000_000)

    expect(isDiagnosticsArmed()).toBe(true)
    expect(FakePerformanceObserver.instances).toHaveLength(1)
    expect(FakePerformanceObserver.instances[0].observed).toEqual([{ buffered: true, type: 'long-animation-frame' }])

    disarmDiagnostics()

    expect(isDiagnosticsArmed()).toBe(false)
    expect(FakePerformanceObserver.instances[0].disconnected).toBe(true)
    expect(readDiagnosticsCapture()).toBeNull()
  })

  it('records a 200ms long frame with attribution while armed', () => {
    armDiagnostics('capture-2', 1_700_000_000_000)
    FakePerformanceObserver.instances[0].emit([longAnimationFrameEntry])

    const snapshot = readDiagnosticsCapture()
    expect(snapshot?.captureId).toBe('capture-2')

    const longFrames = (snapshot?.events ?? []).filter((e): e is LongFrameEvent => e.type === 'long_frame')
    expect(longFrames).toHaveLength(1)

    const [frame] = longFrames
    expect(frame.ms).toBe(204)
    expect(frame.blockingMs).toBe(154)
    // startTime + duration - styleAndLayoutStart = the style+layout tail.
    expect(frame.styleMs).toBe(24)
    expect(frame.t).toBeGreaterThanOrEqual(0)
    // Only the >=5ms script is attributed.
    expect(frame.scripts).toHaveLength(1)
    expect(frame.scripts[0].ms).toBe(198)
    expect(frame.scripts[0].invoker).toBe('user-callback:TIMER')
  })

  it('sanitizes attribution at record time — no paths reach the buffer', () => {
    armDiagnostics('capture-3', 1_700_000_000_000)
    FakePerformanceObserver.instances[0].emit([longAnimationFrameEntry])

    const serialized = JSON.stringify(readDiagnosticsCapture()?.events ?? [])

    expect(serialized).not.toContain('PRIVATE_MARKER')
    expect(serialized).not.toContain('https://')
    expect(serialized).not.toContain('/')
    // The bare file name survives — that is the useful, non-identifying part.
    expect(serialized).toContain('index-abc123.js')
  })

  it('records stream-delta samples as counts and durations only', () => {
    armDiagnostics('capture-4', 1_700_000_000_000)

    const event = recordStreamDeltaApplied({ historyMessages: 412, queuedChars: 87, sessions: 2, writeMs: 4.567 })

    expect(event).not.toBeNull()
    expect(event).toMatchObject({
      commitMs: 0,
      historyMessages: 412,
      queuedChars: 87,
      rafGapMs: 0,
      sessions: 2,
      type: 'stream_delta_applied',
      writeMs: 4.57
    })

    // The caller's existing measurement rAF fills the commit half in place —
    // one event per flush, not two.
    event!.commitMs = 61.2
    expect(readDiagnosticsCapture()?.events).toEqual([expect.objectContaining({ commitMs: 61.2 })])
  })

  it('re-arms with a fresh buffer under a new capture id without a reload', () => {
    armDiagnostics('capture-5', 1_700_000_000_000)
    recordStreamDeltaApplied({ historyMessages: 1, queuedChars: 1, sessions: 1, writeMs: 1 })
    expect(readDiagnosticsCapture()?.events).toHaveLength(1)

    armDiagnostics('capture-6', 1_700_000_000_001)

    expect(readDiagnosticsCapture()?.captureId).toBe('capture-6')
    expect(readDiagnosticsCapture()?.events).toHaveLength(0)
    expect(FakePerformanceObserver.instances).toHaveLength(2)
    expect(FakePerformanceObserver.instances[0].disconnected).toBe(true)
  })

  it('samples the heap on a timer while armed and stops on disarm', () => {
    vi.useFakeTimers()

    Object.defineProperty(performance, 'memory', {
      configurable: true,
      value: {
        jsHeapSizeLimit: 4 * 1024 * 1024 * 1024,
        totalJSHeapSize: 900 * 1024 * 1024,
        usedJSHeapSize: 762 * 1024 * 1024
      }
    })

    try {
      armDiagnostics('capture-7', 1_700_000_000_000)

      expect(readDiagnosticsCapture()?.events).toEqual([
        expect.objectContaining({ limitMb: 4096, totalMb: 900, type: 'memory_sample', usedMb: 762 })
      ])

      vi.advanceTimersByTime(15_000)
      expect(readDiagnosticsCapture()?.events).toHaveLength(4)

      disarmDiagnostics()
      expect(vi.getTimerCount()).toBe(0)
    } finally {
      Reflect.deleteProperty(performance, 'memory')
      vi.useRealTimers()
    }
  })
})
