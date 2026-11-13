import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { pageStorage } from '@/test/page-storage'

import {
  bindDesktopMetrics,
  captureDesktopMetricsGate,
  DESKTOP_METRICS_STATE_KEY,
  recordAction,
  recordLongFrames,
  resetDesktopMetricsForTests,
  tickDesktopMetrics,
  trackDesktopVisibleTime
} from './desktop-metrics'
import { immediateMetricsLocks, indexedDbMetricsLocks, setMetricsGate } from './desktop-metrics-test-locks'

const MINUTE = 60_000
const DAY = 24 * 60 * MINUTE
const NOON = Date.UTC(2026, 8, 27, 12)
const calls: [string, Record<string, unknown>][] = []
const flush = () => vi.runAllTimersAsync()
const histograms = () => calls.filter(([method]) => method === 'shared_metrics.desktop_long_frames').map(([, params]) => params)
let visibility: DocumentVisibilityState

beforeEach(() => {
  immediateMetricsLocks()
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
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('visible time ownership', () => {
  it.each(['frames', 'clean', 'clean with earlier roll'] as const)(
    'credits each UTC day before a roll or flush: %s',
    async scenario => {
      const start = Date.UTC(2026, 8, 27, 23, 55)

      vi.setSystemTime(start)
      await setMetricsGate(captureDesktopMetricsGate, 'on')
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
    await setMetricsGate(captureDesktopMetricsGate, transition === 'profile change' ? 'on' : 'off')
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

    await setMetricsGate(captureDesktopMetricsGate, 'on')
    vi.setSystemTime(optedIn + 2 * MINUTE)
    tickDesktopMetrics()
    vi.setSystemTime(NOON + DAY)
    tickDesktopMetrics()
    await flush()

    expect(histograms()).toEqual([{ day: '2026-09-27', visible_ms: 2 * MINUTE, frames: [] }])
  })

  it('credits visible spans but neither hidden time nor time after tracking stops', async () => {
    await setMetricsGate(captureDesktopMetricsGate, 'on')
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
    await setMetricsGate(captureDesktopMetricsGate, 'on')
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

  it.each([false, true])('does not send old measurements after consent withdrawal (re-enabled: %s)', async reEnable => {
    const afterWithdrawal: [string, Record<string, unknown>][] = []
    let withdrawing = false
    let withdrawn = false

    bindDesktopMetrics(vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push([method, params ?? {}])

      if (withdrawn) {
        afterWithdrawal.push([method, params ?? {}])
      }

      if (!withdrawing && (method === 'shared_metrics.desktop_daily' || method === 'shared_metrics.desktop_long_frames')) {
        withdrawing = true
        await setMetricsGate(captureDesktopMetricsGate, 'off')
        withdrawn = true

        if (reEnable) {
          await setMetricsGate(captureDesktopMetricsGate, 'on')
        }
      }

      return { recorded: true } as never
    }))
    await setMetricsGate(captureDesktopMetricsGate, 'on')
    trackDesktopVisibleTime(11 * MINUTE)
    recordAction('composer.send', 'click')
    recordLongFrames([120])
    vi.setSystemTime(NOON + MINUTE)
    tickDesktopMetrics()
    vi.setSystemTime(NOON + DAY)
    tickDesktopMetrics()
    await flush()
    tickDesktopMetrics()
    await flush()
    expect(withdrawn).toBe(true)
    expect(afterWithdrawal).toEqual([])

    if (reEnable) {
      recordLongFrames([70])
      vi.setSystemTime(NOON + DAY + MINUTE)
      tickDesktopMetrics()
      vi.setSystemTime(NOON + 2 * DAY)
      tickDesktopMetrics()
      await flush()
      expect(afterWithdrawal).toEqual([['shared_metrics.desktop_long_frames', {
        day: '2026-09-28', visible_ms: MINUTE, frames: [{ bucket: '50ms_to_100ms', count: 1 }]
      }]])
    }
  })

  it('migrates a persisted pre-journal histogram without losing or replaying its counts', async () => {
    await setMetricsGate(captureDesktopMetricsGate, 'on')

    const key = Array.from({ length: localStorage.length }, (_, index) => localStorage.key(index)!).find(candidate =>
      candidate.startsWith(`${DESKTOP_METRICS_STATE_KEY}:`) &&
      !candidate.slice(DESKTOP_METRICS_STATE_KEY.length + 1).includes(':'))!

    const legacy = JSON.parse(localStorage.getItem(key)!)

    legacy.today.frames = { '100ms_to_250ms': 7 }
    legacy.today.visibleMs = 3 * MINUTE
    localStorage.setItem(key, JSON.stringify(legacy))
    vi.setSystemTime(NOON + DAY)
    tickDesktopMetrics()
    await flush()
    tickDesktopMetrics()
    await flush()
    expect(histograms()).toEqual([{
      day: '2026-09-27', visible_ms: 3 * MINUTE, frames: [{ bucket: '100ms_to_250ms', count: 7 }]
    }])
  })

  it('waits for a same-profile peer tail before the backend latches midnight', async () => {
    const accepted = new Map<string, Record<string, unknown>>()

    const request = vi.fn(async (method: string, params: Record<string, unknown> = {}) => {
      if (method === 'shared_metrics.desktop_long_frames' && !accepted.has(String(params.day))) {
        accepted.set(String(params.day), structuredClone(params))
      }

      return { recorded: true } as never
    })

    const midnight = Date.UTC(2026, 8, 28)

    vi.setSystemTime(midnight - 4 * MINUTE)
    bindDesktopMetrics(request, 'local|shared')
    await setMetricsGate(captureDesktopMetricsGate, 'on')
    trackDesktopVisibleTime(11 * MINUTE)
    recordLongFrames([120])
    vi.resetModules()
    // A second module load models an independent renderer sharing only localStorage.
    const peer = await import('./desktop-metrics')

    try {
      vi.setSystemTime(midnight - 2 * MINUTE)
      peer.bindDesktopMetrics(request, 'local|shared')
      await setMetricsGate(peer.captureDesktopMetricsGate, 'on')
      peer.trackDesktopVisibleTime(11 * MINUTE)
      peer.recordLongFrames([70, 70])
      vi.setSystemTime(midnight)
      peer.tickDesktopMetrics()
      await flush()
      expect(accepted.size).toBe(0)
      vi.setSystemTime(midnight + MINUTE)
      tickDesktopMetrics()
      peer.tickDesktopMetrics()
      await flush()
      expect([...accepted.values()]).toEqual([{
        day: '2026-09-27',
        visible_ms: 6 * MINUTE,
        frames: [{ bucket: '50ms_to_100ms', count: 2 }, { bucket: '100ms_to_250ms', count: 1 }]
      }])
    } finally {
      peer.resetDesktopMetricsForTests()
    }
  })

  it('retains the legacy payload when its renderer dies before queued adoption can acquire ownership', async () => {
    await setMetricsGate(captureDesktopMetricsGate, 'on')

    const key = Array.from({ length: localStorage.length }, (_, index) => localStorage.key(index)!).find(candidate =>
      candidate.startsWith(`${DESKTOP_METRICS_STATE_KEY}:`) &&
      !candidate.slice(DESKTOP_METRICS_STATE_KEY.length + 1).includes(':'))!

    const legacy = JSON.parse(localStorage.getItem(key)!)

    legacy.today.frames = { '100ms_to_250ms': 7 }
    legacy.today.visibleMs = 3 * MINUTE
    localStorage.setItem(key, JSON.stringify(legacy))
    // This renderer dies while waiting: none of its queued callbacks can ever execute.
    const abandoned = Promise.withResolvers<void>()

    vi.stubGlobal('navigator', { locks: { request: () => abandoned.promise } })
    vi.setSystemTime(NOON + DAY)
    tickDesktopMetrics()
    // Rollover is also queued now: recovery must work whether the legacy day is in today or pending.
    resetDesktopMetricsForTests()
    immediateMetricsLocks()
    vi.resetModules()
    // A fresh module instance models the replacement renderer, not an echo of the old collector.
    const replacement = await import('./desktop-metrics')

    try {
      replacement.bindDesktopMetrics(async (method, params = {}) => {
        calls.push([method, params])

        return { recorded: true } as never
      })
      await setMetricsGate(replacement.captureDesktopMetricsGate, 'on')
      await flush()
      replacement.tickDesktopMetrics()
      await flush()
      expect(histograms()).toEqual([{
        day: '2026-09-27', visible_ms: 3 * MINUTE, frames: [{ bucket: '100ms_to_250ms', count: 7 }]
      }])
    } finally {
      replacement.resetDesktopMetricsForTests()
    }
  })

  it.each([false, true])('requires a fresh positive consent read after peer GC (withdrawn meanwhile: %s)', async withdrawn => {
    await setMetricsGate(captureDesktopMetricsGate, 'on')
    trackDesktopVisibleTime(11 * MINUTE)
    recordLongFrames([6000])
    vi.setSystemTime(NOON + 9 * DAY)
    vi.resetModules()
    // A second renderer performs retention cleanup while the first is frozen.
    const peer = await import('./desktop-metrics')

    try {
      peer.bindDesktopMetrics(async (method, params = {}) => {
        calls.push([method, params])

        return { recorded: true } as never
      })
      await setMetricsGate(peer.captureDesktopMetricsGate, 'on')
      await flush()

      if (withdrawn) {
        await setMetricsGate(peer.captureDesktopMetricsGate, 'off')
        await setMetricsGate(peer.captureDesktopMetricsGate, 'on')
      }

      recordLongFrames([6000]) // A stale collector cannot treat missing ownership as renewed consent.
      // useDesktopMetrics' focus refresh reads backend consent, then calls this same-on setter.
      await setMetricsGate(captureDesktopMetricsGate, 'on')
      vi.setSystemTime(NOON + 9 * DAY + MINUTE)
      recordLongFrames([70])
      vi.setSystemTime(NOON + 10 * DAY)
      tickDesktopMetrics()
      await flush()
      expect(histograms()).toEqual([{
        day: new Date(NOON + 9 * DAY).toISOString().slice(0, 10),
        visible_ms: MINUTE,
        frames: [{ bucket: '50ms_to_100ms', count: 1 }]
      }])
    } finally {
      peer.resetDesktopMetricsForTests()
    }
  })

  it.each(['before purge', 'while purge waits'])('rejects an old off result when a peer replaces its cohort %s', async timing => {
    const request = async (method: string, params: Record<string, unknown> = {}) => {
      calls.push([method, params])

      return { recorded: true } as never
    }

    bindDesktopMetrics(request, 'local|shared')
    await setMetricsGate(captureDesktopMetricsGate, 'on')
    trackDesktopVisibleTime(11 * MINUTE)
    recordLongFrames([6000])
    recordAction('composer.send', 'click')
    const delayedOff = await captureDesktopMetricsGate()

    vi.resetModules()
    const peer = await import('./desktop-metrics')

    try {
      peer.bindDesktopMetrics(request, 'local|shared')
      let withdrawn: Promise<boolean>

      if (timing === 'before purge') {
        await setMetricsGate(peer.captureDesktopMetricsGate, 'off')
        await setMetricsGate(peer.captureDesktopMetricsGate, 'on')
        withdrawn = delayedOff('off')
      } else {
        const held = Promise.withResolvers<void>()
        const acquired = Promise.withResolvers<void>()

        const ownership = navigator.locks.request('hermes-desktop-metrics-coordination', () => {
          acquired.resolve()

          return held.promise
        })

        await acquired.promise
        withdrawn = delayedOff('off')
        const renewed = setMetricsGate(peer.captureDesktopMetricsGate, 'on')

        held.resolve()
        await ownership
        await renewed
      }

      await expect(withdrawn).resolves.toBe(false)
      peer.trackDesktopVisibleTime(11 * MINUTE)
      peer.recordLongFrames([70])
      peer.recordAction('session.new', 'shortcut')
      vi.setSystemTime(NOON + MINUTE)
      peer.tickDesktopMetrics()
      vi.setSystemTime(NOON + DAY)
      peer.tickDesktopMetrics()
      tickDesktopMetrics()
      await flush()
      expect(histograms()).toEqual([{
        day: '2026-09-27', visible_ms: MINUTE, frames: [{ bucket: '50ms_to_100ms', count: 1 }]
      }])
      expect(calls.filter(([method]) => method === 'shared_metrics.desktop_daily').map(([, params]) => params.actions)).toEqual([
        [{ action: 'session.new', count: 1, via: 'shortcut' }]
      ])
    } finally {
      peer.resetDesktopMetricsForTests()
    }
  })

  it('settles a descheduled ordinary writer before rename and old-name recreation', async () => {
    const accepted = new Map<string, Record<string, unknown>>()

    const requester = (owner: string) => async (method: string, params: Record<string, unknown> = {}) => {
      const key = `${owner}|${params.day}`

      if (method === 'shared_metrics.desktop_daily' && !accepted.has(key)) {
        accepted.set(key, structuredClone(params))
      }

      return { recorded: true } as never
    }

    bindDesktopMetrics(requester('old-alpha'), '|alpha')
    await setMetricsGate(captureDesktopMetricsGate, 'on')
    recordAction('composer.send', 'click')
    await flush()

    const key = Array.from({ length: localStorage.length }, (_, index) => localStorage.key(index)!).find(candidate =>
      candidate.startsWith(`${DESKTOP_METRICS_STATE_KEY}:`) &&
      !candidate.slice(DESKTOP_METRICS_STATE_KEY.length + 1).includes(':'))!

    vi.resetModules()
    const peer = await import('./desktop-metrics')
    const storage = pageStorage()
    const setItem = storage.setItem
    let migration: Promise<void> | undefined
    let recreated: Promise<boolean> | undefined
    let interrupted = false

    try {
      peer.bindDesktopMetrics(requester('new-alpha'), '|alpha')
      vi.spyOn(storage, 'setItem').mockImplementation(function (this: Storage, target, value) {
        if (!interrupted && target === key && JSON.parse(value).today?.actions?.['composer.send|click'] === 2) {
          interrupted = true
          // Another renderer runs at the old writer's publication boundary. The shared test
          // coordinator queues nested phases while owned, but executes them immediately if idle.
          migration = peer.migrateDesktopMetricsForProfile('alpha', 'beta')
          recreated = setMetricsGate(peer.captureDesktopMetricsGate, 'on')
        }

        setItem.call(this, target, value)
      })
      recordAction('composer.send', 'click')
      expect(interrupted).toBe(true)
      await migration
      await expect(recreated).resolves.toBe(true)
      peer.recordAction('session.new', 'shortcut')
      recordAction('composer.send', 'click') // The retired original renderer cannot publish again.
      vi.setSystemTime(NOON + DAY)
      peer.tickDesktopMetrics()
      tickDesktopMetrics()
      await flush()
      peer.bindDesktopMetrics(requester('beta'), '|beta')
      await setMetricsGate(peer.captureDesktopMetricsGate, 'on')
      await flush()
      expect(accepted.get('new-alpha|2026-09-27')?.actions).toEqual([
        { action: 'session.new', count: 1, via: 'shortcut' }
      ])
      expect(accepted.get('beta|2026-09-27')?.actions).toEqual([
        { action: 'composer.send', count: 2, via: 'click' }
      ])
      expect(accepted.has('old-alpha|2026-09-27')).toBe(false)
    } finally {
      peer.resetDesktopMetricsForTests()
    }
  })

  it('retires a failed initial IndexedDB track and restores the denominator on positive consent refresh', async () => {
    const coordination = indexedDbMetricsLocks()
    let now = NOON

    // Transaction tasks may advance fake timers; only explicit observations advance usage time.
    vi.spyOn(Date, 'now').mockImplementation(() => now)

    try {
      trackDesktopVisibleTime(11 * MINUTE)
      // Capture, apply validation and initialization succeed; the first tracking transaction aborts.
      coordination.abortNext(3)
      const first = setMetricsGate(captureDesktopMetricsGate, 'on')
      await flush()
      await first
      recordLongFrames([6000]) // The failed collector must not become a frame-only reporter.
      const resumedAt = Date.now()

      const refreshed = setMetricsGate(captureDesktopMetricsGate, 'on')
      await flush()
      await refreshed
      now = resumedAt + MINUTE
      vi.setSystemTime(now)
      recordLongFrames([70])
      await flush()
      now = NOON + DAY
      vi.setSystemTime(now)
      tickDesktopMetrics()
      await flush()
      expect(histograms()).toEqual([{
        day: '2026-09-27', visible_ms: MINUTE, frames: [{ bucket: '50ms_to_100ms', count: 1 }]
      }])
    } finally {
      coordination.close()
    }
  })
})
