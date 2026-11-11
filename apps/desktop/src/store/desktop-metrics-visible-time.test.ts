import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  bindDesktopMetrics,
  recordAction,
  recordLongFrames,
  resetDesktopMetricsForTests,
  setDesktopMetricsGate,
  tickDesktopMetrics,
  trackDesktopVisibleTime
} from './desktop-metrics'

const MINUTE = 60_000
const DAY = 24 * 60 * MINUTE
const NOON = Date.UTC(2026, 8, 27, 12)
const calls: [string, Record<string, unknown>][] = []
const flush = () => vi.runAllTimersAsync()
const histograms = () => calls.filter(([method]) => method === 'shared_metrics.desktop_long_frames').map(([, params]) => params)
let visibility: DocumentVisibilityState

beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(NOON)
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  calls.length = 0
  visibility = 'visible'
  vi.spyOn(document, 'visibilityState', 'get').mockImplementation(() => visibility)
  bindDesktopMetrics(vi.fn(async (method: string, params?: Record<string, unknown>) => {
    calls.push([method, params ?? {}])

    return { recorded: true } as never
  }))
})

afterEach(() => {
  resetDesktopMetricsForTests()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('visible time ownership', () => {
  it.each(['frames', 'clean', 'clean with earlier roll'] as const)(
    'credits each UTC day before a roll or flush: %s',
    async scenario => {
      const start = Date.UTC(2026, 8, 27, 23, 55)

      vi.setSystemTime(start)
      setDesktopMetricsGate('on')
      trackDesktopVisibleTime(11 * MINUTE)

      if (scenario === 'frames') {
        vi.setSystemTime(start + 3 * MINUTE)
        recordLongFrames([120, 120, 120])
      }

      if (scenario !== 'clean') {
        vi.setSystemTime(start + 6 * MINUTE)
        recordAction('composer.send', 'click')
        tickDesktopMetrics() // Another event can roll and flush before the scheduled clock tick.
        await flush()
      }

      vi.setSystemTime(start + 10 * MINUTE)
      tickDesktopMetrics()
      await flush()
      vi.setSystemTime(start + 10 * MINUTE + DAY)
      tickDesktopMetrics()
      await flush()

      expect(histograms()).toEqual([
        {
          day: '2026-09-27',
          visible_ms: 5 * MINUTE,
          frames: scenario === 'frames' ? [{ bucket: '100ms_to_250ms', count: 3 }] : []
        },
        { day: '2026-09-28', visible_ms: 5 * MINUTE, frames: [] }
      ])
    }
  )

  it.each(['opt-in', 'profile change'] as const)('drops elapsed time before %s', async transition => {
    setDesktopMetricsGate(transition === 'profile change' ? 'on' : 'off')
    trackDesktopVisibleTime(11 * MINUTE)
    vi.setSystemTime(NOON + 9 * MINUTE)

    if (transition === 'profile change') {
      bindDesktopMetrics(vi.fn(async (method: string, params?: Record<string, unknown>) => {
        calls.push([method, params ?? {}])

        return { recorded: true } as never
      }), 'connection|other-profile')
      vi.setSystemTime(NOON + 16 * MINUTE)
    }

    const optedIn = Date.now()

    setDesktopMetricsGate('on')
    vi.setSystemTime(optedIn + 2 * MINUTE)
    tickDesktopMetrics()
    vi.setSystemTime(NOON + DAY)
    tickDesktopMetrics()
    await flush()

    expect(histograms()).toEqual([{ day: '2026-09-27', visible_ms: 2 * MINUTE, frames: [] }])
  })

  it('credits visible spans but neither hidden time nor time after tracking stops', async () => {
    setDesktopMetricsGate('on')
    const stop = trackDesktopVisibleTime(11 * MINUTE)

    vi.setSystemTime(NOON + MINUTE)
    visibility = 'hidden'
    tickDesktopMetrics()
    vi.setSystemTime(NOON + 5 * MINUTE)
    tickDesktopMetrics()
    visibility = 'visible'
    tickDesktopMetrics()
    vi.setSystemTime(NOON + 7 * MINUTE)
    stop()
    vi.setSystemTime(NOON + 9 * MINUTE)
    recordAction('composer.send', 'click')
    vi.setSystemTime(NOON + DAY)
    tickDesktopMetrics()
    await flush()

    expect(histograms()).toEqual([{ day: '2026-09-27', visible_ms: 3 * MINUTE, frames: [] }])
  })

  it('excludes a suspend gap and resumes counting subsequent awake spans', async () => {
    setDesktopMetricsGate('on')
    trackDesktopVisibleTime(11 * MINUTE)
    vi.setSystemTime(NOON + 10 * MINUTE)
    tickDesktopMetrics()
    vi.setSystemTime(NOON + 10 * MINUTE + 8 * 60 * MINUTE)
    tickDesktopMetrics()
    vi.setSystemTime(NOON + 20 * MINUTE + 8 * 60 * MINUTE)
    tickDesktopMetrics()
    vi.setSystemTime(NOON + DAY)
    tickDesktopMetrics()
    await flush()

    expect(histograms()).toEqual([{ day: '2026-09-27', visible_ms: 20 * MINUTE, frames: [] }])
  })

  it.each([false, true])('does not send a purged histogram after consent withdrawal (re-enabled: %s)', async reEnable => {
    bindDesktopMetrics(vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push([method, params ?? {}])

      if (method === 'shared_metrics.desktop_daily') {
        setDesktopMetricsGate('off')

        if (reEnable) {
          setDesktopMetricsGate('on')
        }
      }

      return { recorded: true } as never
    }))
    setDesktopMetricsGate('on')
    trackDesktopVisibleTime(11 * MINUTE)
    recordAction('composer.send', 'click')
    recordLongFrames([120])
    vi.setSystemTime(NOON + MINUTE)
    tickDesktopMetrics()
    vi.setSystemTime(NOON + DAY)
    tickDesktopMetrics()
    await flush()

    expect(calls.filter(([method]) => method.startsWith('shared_metrics.desktop_')).map(([method]) => method)).toEqual([
      'shared_metrics.desktop_daily'
    ])
  })
})
