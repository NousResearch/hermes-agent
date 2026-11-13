import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { pageStorage } from '@/test/page-storage'

import { DesktopFrameMetrics, type DesktopFrameRevocation } from './desktop-metrics-frames'
import { immediateMetricsLocks, indexedDbMetricsLocks } from './desktop-metrics-test-locks'
import type { SharedMetricsRequester } from './shared-metrics'

const MINUTE = 60_000
const DAY = 24 * 60 * MINUTE
const MIDNIGHT = Date.UTC(2026, 8, 28)
const KEY = 'hermes.desktop.metrics.v1:test-scope'
const current = () => true
const storedKeys = () => Array.from({ length: localStorage.length }, (_, index) => localStorage.key(index)!)
let closeIndexedDb: () => void = () => undefined

function serialLocks(blocked = Promise.resolve()) {
  const tails = new Map<string, Promise<void>>()

  const request = (name: string, run: () => unknown) => {
    const pending = (tails.get(name) ?? blocked).then(run)

    tails.set(name, pending.then(() => undefined, () => undefined))

    return pending
  }

  vi.stubGlobal('navigator', { locks: { request } })

  return { drain: () => Promise.all([...tails.values()]) }
}

// Match the real backend's first-report-wins latch, not an echo that accepts late corrections.
function backend() {
  const accepted = new Map<string, Record<string, unknown>>()
  const attempts: Record<string, unknown>[] = []
  let loseAck = false

  const request: SharedMetricsRequester = async (_method, params = {}) => {
    attempts.push(params)

    if (!accepted.has(String(params.day))) {
      accepted.set(String(params.day), structuredClone(params))
    }

    if (loseAck) {
      loseAck = false
      throw new Error('ack lost after commit')
    }

    return { recorded: true } as never
  }

  return { accepted, attempts, request, loseNextAck: () => { loseAck = true } }
}

beforeEach(() => {
  immediateMetricsLocks()
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(MIDNIGHT - 4 * MINUTE)
  window.localStorage.clear()
  vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
})

afterEach(() => {
  closeIndexedDb()
  closeIndexedDb = () => undefined
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('peer frame day settlement', () => {
  it.each([false, true])('preserves both windows before first-day latch (Web Locks: %s)', async nativeLocks => {
    if (nativeLocks) {
      serialLocks()
    } else {
      closeIndexedDb = indexedDbMetricsLocks().close
    }

    const sink = backend()
    const first = new DesktopFrameMetrics(KEY)

    first.track(11 * MINUTE)
    first.record(['100ms_to_250ms'])
    vi.setSystemTime(MIDNIGHT - 2 * MINUTE)
    const second = new DesktopFrameMetrics(KEY)

    second.track(11 * MINUTE)
    second.record(['50ms_to_100ms', '50ms_to_100ms'])
    vi.setSystemTime(MIDNIGHT)
    second.checkpoint()
    await second.flush(sink.request, current)
    expect(sink.accepted.size).toBe(0)

    vi.setSystemTime(MIDNIGHT + MINUTE)
    first.checkpoint()
    second.checkpoint()
    await Promise.all([first.flush(sink.request, current), second.flush(sink.request, current)])
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-27',
      visible_ms: 6 * MINUTE,
      frames: [{ bucket: '50ms_to_100ms', count: 2 }, { bucket: '100ms_to_250ms', count: 1 }]
    }])
    expect(sink.attempts.every(attempt => JSON.stringify(attempt) === JSON.stringify(sink.attempts[0]))).toBe(true)
    expect(Object.keys(sink.attempts[0]!).sort()).toEqual(['day', 'frames', 'visible_ms'])
  })

  it.each([false, true])('settles a crashed peer only beyond the exact valid-span boundary (Web Locks: %s)', async nativeLocks => {
    if (nativeLocks) {
      serialLocks()
    } else {
      closeIndexedDb = indexedDbMetricsLocks().close
    }

    const sink = backend()
    const crashed = new DesktopFrameMetrics(KEY)

    crashed.track(11 * MINUTE)
    vi.setSystemTime(MIDNIGHT - 3 * MINUTE)
    crashed.record(['100ms_to_250ms']) // Persisted minute survives the renderer disappearing.
    const survivor = new DesktopFrameMetrics(KEY)

    survivor.track(11 * MINUTE)
    await survivor.flush(sink.request, current)
    vi.setSystemTime(MIDNIGHT)
    survivor.checkpoint()
    vi.setSystemTime(MIDNIGHT + 8 * MINUTE)
    survivor.checkpoint()
    await survivor.flush(sink.request, current)
    expect(sink.accepted.size).toBe(0)
    vi.setSystemTime(MIDNIGHT + 8 * MINUTE + 1)
    await survivor.flush(sink.request, current)
    expect(sink.accepted.get('2026-09-27')).toEqual({
      day: '2026-09-27', visible_ms: 4 * MINUTE, frames: [{ bucket: '100ms_to_250ms', count: 1 }]
    })

    // A suspended rather than crashed peer follows the same exclusion rule on wake.
    crashed.checkpoint()
    await crashed.flush(sink.request, current)
    expect(sink.attempts).toHaveLength(1)
  })

  it('settles hidden and gracefully reloaded peers immediately and keeps a clean-day denominator', async () => {
    const sink = backend()
    const first = new DesktopFrameMetrics(KEY)

    first.track(11 * MINUTE)
    vi.setSystemTime(MIDNIGHT - 2 * MINUTE)
    first.stop()
    const reloaded = new DesktopFrameMetrics(KEY)

    reloaded.track(11 * MINUTE)
    vi.setSystemTime(MIDNIGHT - MINUTE)
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden')
    reloaded.checkpoint()
    vi.setSystemTime(MIDNIGHT)
    await reloaded.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{ day: '2026-09-27', visible_ms: 3 * MINUTE, frames: [] }])
  })

  it('retries an identical immutable snapshot after a committed report loses its ack', async () => {
    const sink = backend()
    const collector = new DesktopFrameMetrics(KEY)

    collector.track(11 * MINUTE)
    collector.record(['gte_5s'])
    vi.setSystemTime(MIDNIGHT)
    collector.checkpoint()
    sink.loseNextAck()
    await collector.flush(sink.request, current)
    await collector.flush(sink.request, current)
    await collector.flush(sink.request, current)
    expect(sink.attempts).toHaveLength(2)
    expect(sink.attempts[1]).toEqual(sink.attempts[0])
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: 4 * MINUTE, frames: [{ bucket: 'gte_5s', count: 1 }]
    }])
  })

  it('revokes sleeping and in-flight peers across off/on without settling the replacement epoch', async () => {
    const collector = new DesktopFrameMetrics(KEY)
    const sleeping = new DesktopFrameMetrics(KEY)

    collector.track(11 * MINUTE)
    sleeping.record(['gte_5s'])
    vi.setSystemTime(MIDNIGHT)
    collector.checkpoint()
    const acknowledgement = Promise.withResolvers<{ recorded: boolean }>()
    const request = vi.fn(() => acknowledgement.promise)
    const pending = collector.flush(request as SharedMetricsRequester, current)

    await vi.waitFor(() => expect(request).toHaveBeenCalledTimes(1))
    await DesktopFrameMetrics.purge(KEY)
    const resumedAt = Date.now()
    const replacement = new DesktopFrameMetrics(KEY)

    replacement.track(11 * MINUTE)
    replacement.record(['50ms_to_100ms'])
    sleeping.record(['gte_5s'])
    acknowledgement.resolve({ recorded: true })
    await pending
    const sink = backend()

    await sleeping.flush(sink.request, current)
    expect(sink.attempts).toEqual([])
    vi.setSystemTime(resumedAt + MINUTE)
    replacement.checkpoint()
    vi.setSystemTime(MIDNIGHT + DAY)
    replacement.checkpoint()
    await replacement.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-28', visible_ms: MINUTE, frames: [{ bucket: '50ms_to_100ms', count: 1 }]
    }])
  })

  it('bounds dead journals and resumes after a fresh consent attachment without reopening old days', async () => {
    const stale = new DesktopFrameMetrics(KEY)

    stale.track(11 * MINUTE)
    stale.record(['gte_5s'])
    vi.setSystemTime(MIDNIGHT + 9 * DAY)
    const active = new DesktopFrameMetrics(KEY)
    const sink = backend()

    active.track(11 * MINUTE)
    await active.flush(sink.request, current)
    stale.checkpoint()
    expect(stale.isCurrent()).toBe(false)
    // An accepted positive consent result creates a replacement rather than reviving this journal.
    const resumed = new DesktopFrameMetrics(KEY)

    resumed.track(11 * MINUTE)
    vi.setSystemTime(MIDNIGHT + 9 * DAY + MINUTE)
    resumed.record(['50ms_to_100ms'])
    vi.setSystemTime(MIDNIGHT + 10 * DAY)
    active.checkpoint()
    resumed.checkpoint()
    await active.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-10-07', visible_ms: MINUTE, frames: [{ bucket: '50ms_to_100ms', count: 1 }]
    }])
    expect(storedKeys().filter(key =>
      key.startsWith(`${KEY}:frames:`) && !key.includes(':settled:') && !key.includes(':consent:'))).toHaveLength(2)
    expect(storedKeys().map(key => localStorage.getItem(key)).join('')).not.toContain('2026-09-27')
  })

  it('does not count time twice after a backwards wall-clock correction', async () => {
    const collector = new DesktopFrameMetrics(KEY)
    const sink = backend()

    collector.track(11 * MINUTE)
    vi.setSystemTime(MIDNIGHT - 2 * MINUTE)
    collector.checkpoint()
    vi.setSystemTime(MIDNIGHT - 3 * MINUTE)
    collector.checkpoint()
    vi.setSystemTime(MIDNIGHT)
    collector.checkpoint()
    await collector.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{ day: '2026-09-27', visible_ms: 4 * MINUTE, frames: [] }])
  })

  it('revokes a finalizer queued on a native lock without deleting the new epoch journals', async () => {
    const lock = Promise.withResolvers<void>()

    const locks = serialLocks(lock.promise)
    const sink = backend()
    const stale = new DesktopFrameMetrics(KEY)

    stale.track(11 * MINUTE)
    vi.setSystemTime(MIDNIGHT)
    stale.checkpoint()
    const pending = stale.flush(sink.request, current)

    const revoked = DesktopFrameMetrics.purge(KEY)
    const replacement = new DesktopFrameMetrics(KEY)

    replacement.track(11 * MINUTE)
    replacement.record(['100ms_to_250ms'])
    lock.resolve()
    await pending
    await expect(revoked).resolves.toMatchObject({ complete: true })
    await locks.drain()
    expect(sink.attempts).toEqual([])
    vi.setSystemTime(MIDNIGHT + MINUTE)
    replacement.stop()
    await locks.drain()
    vi.setSystemTime(MIDNIGHT + DAY)
    await replacement.flush(sink.request, current)
    const reloaded = new DesktopFrameMetrics(KEY)

    await reloaded.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-28', visible_ms: MINUTE, frames: [{ bucket: '100ms_to_250ms', count: 1 }]
    }])
  })

  it('adopts the old shared histogram only once even when two windows read it', async () => {
    const first = new DesktopFrameMetrics(KEY)
    const second = new DesktopFrameMetrics(KEY)
    const oldDay = { visibleMs: 2 * MINUTE, frames: { '100ms_to_250ms': 3 } }
    const sink = backend()

    first.adopt('2026-09-27', oldDay)
    second.adopt('2026-09-27', oldDay)
    vi.setSystemTime(MIDNIGHT)
    await Promise.all([first.flush(sink.request, current), second.flush(sink.request, current)])
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: 2 * MINUTE, frames: [{ bucket: '100ms_to_250ms', count: 3 }]
    }])
  })

  it('cannot overwrite a revocation arriving between constructor token read and initialization', async () => {
    const old = new DesktopFrameMetrics(KEY)

    old.track(11 * MINUTE)
    old.record(['gte_5s'])
    expect(old.isCurrent()).toBe(true)
    const getItem = pageStorage().getItem
    let revoke = true

    vi.spyOn(pageStorage(), 'getItem').mockImplementation(function (this: Storage, key: string) {
      const value = getItem.call(this, key)

      if (key === `${KEY}:frames` && revoke) {
        revoke = false
        DesktopFrameMetrics.purge(KEY)
      }

      return value
    })
    const racing = new DesktopFrameMetrics(KEY)
    const sink = backend()

    expect(old.isCurrent()).toBe(false)
    expect(racing.isCurrent()).toBe(false)
    racing.track(11 * MINUTE)
    racing.record(['gte_5s'])
    vi.setSystemTime(MIDNIGHT)
    await old.flush(sink.request, current)
    await racing.flush(sink.request, current)
    expect(sink.attempts).toEqual([])
    expect(storedKeys().filter(key => key.startsWith(`${KEY}:frames:`))).toEqual([])
    expect(new DesktopFrameMetrics(KEY).isCurrent()).toBe(true)
  })

  it('checkpoint and stop at the same instant neither double-credit nor leave a live tail', async () => {
    const collector = new DesktopFrameMetrics(KEY)
    const sink = backend()

    collector.track(11 * MINUTE)
    vi.setSystemTime(MIDNIGHT)
    collector.checkpoint()
    collector.checkpoint()
    collector.stop()
    const reloaded = new DesktopFrameMetrics(KEY)

    await reloaded.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{ day: '2026-09-27', visible_ms: 4 * MINUTE, frames: [] }])
  })

  it.each(['hidden', 'paused'] as const)('closes a %s cursor even while the wall clock is behind', async transition => {
    const collector = new DesktopFrameMetrics(KEY)
    const sink = backend()

    collector.track(11 * MINUTE)
    vi.setSystemTime(MIDNIGHT - 2 * MINUTE)
    collector.checkpoint()
    vi.setSystemTime(MIDNIGHT - 3 * MINUTE)

    if (transition === 'hidden') {
      vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden')
      collector.checkpoint()
    } else {
      collector.track(0)
    }

    vi.setSystemTime(MIDNIGHT - MINUTE)
    collector.checkpoint()
    vi.setSystemTime(MIDNIGHT)
    await collector.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{ day: '2026-09-27', visible_ms: 2 * MINUTE, frames: [] }])
  })

  it.each(['hidden', 'paused'] as const)('retains the high-water mark when %s tracking resumes behind it', async transition => {
    const collector = new DesktopFrameMetrics(KEY)
    const sink = backend()

    collector.track(11 * MINUTE)
    vi.setSystemTime(MIDNIGHT - 2 * MINUTE)
    collector.checkpoint()
    vi.setSystemTime(MIDNIGHT - 3 * MINUTE)

    if (transition === 'hidden') {
      vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden')
      collector.checkpoint()
      vi.setSystemTime(MIDNIGHT - 2.5 * MINUTE)
      vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
      collector.checkpoint()
    } else {
      collector.track(0)
      vi.setSystemTime(MIDNIGHT - 2.5 * MINUTE)
      collector.track(11 * MINUTE)
    }

    vi.setSystemTime(MIDNIGHT - 1.5 * MINUTE)
    collector.track(0)
    vi.setSystemTime(MIDNIGHT)
    await collector.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{ day: '2026-09-27', visible_ms: 2.5 * MINUTE, frames: [] }])
  })

  it('revokes a published writer when quota blocks the token until its journal frees space', async () => {
    const locks = serialLocks()
    const stale = new DesktopFrameMetrics(KEY)

    stale.track(11 * MINUTE)
    stale.record(['gte_5s'])
    await locks.drain()
    const storage = pageStorage()
    const setItem = storage.setItem
    const removeItem = storage.removeItem
    let freed = false

    vi.spyOn(storage, 'setItem').mockImplementation(function (this: Storage, key: string, value: string) {
      if (key === `${KEY}:frames` && !freed) {
        throw new DOMException('full', 'QuotaExceededError')
      }

      setItem.call(this, key, value)
    })
    vi.spyOn(storage, 'removeItem').mockImplementation(function (this: Storage, key: string) {
      removeItem.call(this, key)

      if (key.startsWith(`${KEY}:frames:`) && !key.includes(':consent:')) {
        freed = true
        stale.record(['gte_5s']) // Resume between freeing quota and retrying the scope token.
      }
    })

    await expect(DesktopFrameMetrics.purge(KEY)).resolves.toMatchObject({ complete: true })
    expect(freed).toBe(true)
    expect(stale.isCurrent()).toBe(false)
    const replacement = new DesktopFrameMetrics(KEY)
    const sink = backend()

    replacement.track(11 * MINUTE)
    replacement.record(['50ms_to_100ms'])
    vi.setSystemTime(MIDNIGHT - 3 * MINUTE)
    replacement.track(0)
    vi.setSystemTime(MIDNIGHT)
    stale.checkpoint()
    await stale.flush(sink.request, current)
    await replacement.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: MINUTE, frames: [{ bucket: '50ms_to_100ms', count: 1 }]
    }])
  })

  it('holds the ownership lock across a descheduled checkpoint publication', async () => {
    const locks = serialLocks()
    const sink = backend()

    vi.setSystemTime(MIDNIGHT - 5 * MINUTE)
    const writer = new DesktopFrameMetrics(KEY)

    writer.track(11 * MINUTE)
    writer.record(['100ms_to_250ms'])
    await locks.drain()
    vi.setSystemTime(MIDNIGHT - 2 * MINUTE)
    const reporter = new DesktopFrameMetrics(KEY)

    reporter.track(11 * MINUTE)
    await locks.drain()
    vi.setSystemTime(MIDNIGHT)
    reporter.checkpoint()
    await locks.drain()
    const storage = pageStorage()
    const setItem = storage.setItem
    let finalizing: Promise<void> | undefined
    let interrupted = false

    vi.spyOn(storage, 'setItem').mockImplementation(function (this: Storage, key: string, value: string) {
      const journal = JSON.parse(value)

      if (!interrupted && key.startsWith(`${KEY}:frames:`) &&
        journal.days?.['2026-09-27']?.visibleMs === 5 * MINUTE) {
        interrupted = true
        vi.setSystemTime(MIDNIGHT + 6 * MINUTE + 1)
        finalizing = reporter.flush(sink.request, current)
        expect(sink.attempts).toEqual([])
      }

      setItem.call(this, key, value)
    })
    vi.setSystemTime(MIDNIGHT + 6 * MINUTE - 1)
    writer.checkpoint()
    await locks.drain()
    await finalizing
    expect(interrupted).toBe(true)
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: 7 * MINUTE, frames: [{ bucket: '100ms_to_250ms', count: 1 }]
    }])
  })

  it('checks expiry inside the lock rather than crediting an expired pre-lock sample', async () => {
    const locks = serialLocks()
    const sink = backend()

    vi.setSystemTime(MIDNIGHT - 5 * MINUTE)
    const writer = new DesktopFrameMetrics(KEY)

    writer.track(11 * MINUTE)
    writer.record(['100ms_to_250ms'])
    await locks.drain()
    vi.setSystemTime(MIDNIGHT - 2 * MINUTE)
    const reporter = new DesktopFrameMetrics(KEY)

    reporter.track(11 * MINUTE)
    await locks.drain()
    vi.setSystemTime(MIDNIGHT)
    reporter.checkpoint()
    await locks.drain()
    const held = Promise.withResolvers<void>()

    void navigator.locks.request('hermes-desktop-metrics-coordination', () => held.promise)
    vi.setSystemTime(MIDNIGHT + 6 * MINUTE - 1)
    writer.checkpoint()
    vi.setSystemTime(MIDNIGHT + 6 * MINUTE + 1)
    held.resolve()
    await locks.drain()
    await reporter.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: 2 * MINUTE, frames: [{ bucket: '100ms_to_250ms', count: 1 }]
    }])
  })

  it('keeps queued visibility boundaries instead of counting time hidden behind a lock', async () => {
    const locks = serialLocks()
    const sink = backend()
    const collector = new DesktopFrameMetrics(KEY)

    collector.track(11 * MINUTE)
    await locks.drain()
    const held = Promise.withResolvers<void>()

    void navigator.locks.request('hermes-desktop-metrics-coordination', () => held.promise)
    vi.setSystemTime(MIDNIGHT - 3 * MINUTE)
    collector.checkpoint()
    vi.setSystemTime(MIDNIGHT - 2 * MINUTE)
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden')
    collector.checkpoint()
    vi.setSystemTime(MIDNIGHT + MINUTE)
    held.resolve()
    await locks.drain()
    await collector.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{ day: '2026-09-27', visible_ms: 2 * MINUTE, frames: [] }])
  })

  it('retries a frozen report after an IndexedDB transaction abort without losing its contribution', async () => {
    const coordination = indexedDbMetricsLocks()

    closeIndexedDb = coordination.close
    const sink = backend()
    const collector = new DesktopFrameMetrics(KEY)

    collector.track(11 * MINUTE)
    collector.record(['100ms_to_250ms'])
    await collector.flush(sink.request, current)
    vi.setSystemTime(MIDNIGHT)
    collector.checkpoint()
    await collector.flush(async () => ({ recorded: false }) as never, current)
    coordination.abortNext()
    await collector.flush(sink.request, current)
    expect(sink.attempts).toEqual([])
    await collector.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: 4 * MINUTE, frames: [{ bucket: '100ms_to_250ms', count: 1 }]
    }])
  })

  it('never reassigns a frame observed on an already settled UTC day to the current day', async () => {
    const locks = serialLocks()
    const sink = backend()
    const collector = new DesktopFrameMetrics(KEY)

    collector.track(11 * MINUTE)
    collector.record(['100ms_to_250ms'])
    vi.setSystemTime(MIDNIGHT)
    collector.checkpoint()
    await collector.flush(sink.request, current)
    vi.setSystemTime(MIDNIGHT - MINUTE)
    collector.record(['gte_5s'])
    vi.setSystemTime(MIDNIGHT + MINUTE)
    collector.checkpoint()
    collector.track(0)
    await locks.drain()
    vi.setSystemTime(MIDNIGHT + DAY)
    await collector.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([
      { day: '2026-09-27', visible_ms: 4 * MINUTE, frames: [{ bucket: '100ms_to_250ms', count: 1 }] },
      { day: '2026-09-28', visible_ms: MINUTE, frames: [] }
    ])
  })

  it('releases IndexedDB ownership while awaiting the report RPC and preserves later-day measurements', async () => {
    const coordination = indexedDbMetricsLocks()

    closeIndexedDb = coordination.close
    const sink = backend()
    const collector = new DesktopFrameMetrics(KEY)

    collector.track(11 * MINUTE)
    collector.record(['100ms_to_250ms'])
    await collector.flush(sink.request, current)
    vi.setSystemTime(MIDNIGHT)
    collector.checkpoint()
    const acknowledgement = Promise.withResolvers<void>()

    const request = vi.fn(async (method: string, params: Record<string, unknown> = {}) => {
      const result = await sink.request(method, params)

      await acknowledgement.promise

      return result as never
    })

    const pending = collector.flush(request, current)

    await vi.waitFor(() => expect(request).toHaveBeenCalledTimes(1))
    vi.setSystemTime(MIDNIGHT + MINUTE)
    collector.record(['50ms_to_100ms'])
    await vi.waitFor(() => {
      const journals = storedKeys().map(key => JSON.parse(localStorage.getItem(key)!))

      expect(journals.some(journal => journal.days?.['2026-09-28']?.visibleMs === MINUTE)).toBe(true)
    })
    acknowledgement.resolve()
    await pending
    vi.setSystemTime(MIDNIGHT + DAY)
    collector.checkpoint()
    await collector.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([
      { day: '2026-09-27', visible_ms: 4 * MINUTE, frames: [{ bucket: '100ms_to_250ms', count: 1 }] },
      { day: '2026-09-28', visible_ms: MINUTE, frames: [{ bucket: '50ms_to_100ms', count: 1 }] }
    ])
  })

  it('does not send an in-memory snapshot when quota prevents freezing its durable copy', async () => {
    const collector = new DesktopFrameMetrics(KEY)
    const sink = backend()

    collector.track(11 * MINUTE)
    collector.record(['100ms_to_250ms'])
    vi.setSystemTime(MIDNIGHT)
    collector.checkpoint()
    const storage = pageStorage()
    const setItem = storage.setItem
    let full = true

    vi.spyOn(storage, 'setItem').mockImplementation(function (this: Storage, key: string, value: string) {
      if (full && key.includes(':report:')) {
        throw new DOMException('full', 'QuotaExceededError')
      }

      setItem.call(this, key, value)
    })
    await collector.flush(sink.request, current)
    expect(sink.attempts).toEqual([])
    full = false
    await collector.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: 4 * MINUTE, frames: [{ bucket: '100ms_to_250ms', count: 1 }]
    }])
  })

  it('serializes quota reclamation behind an in-flight writer so it cannot steal the reclaimed space', async () => {
    const locks = serialLocks()
    const stale = new DesktopFrameMetrics(KEY)

    stale.track(11 * MINUTE)
    stale.record(['gte_5s'])
    await locks.drain()
    const storage = pageStorage()
    const setItem = storage.setItem
    const removeItem = storage.removeItem
    let reclaimed = false
    let interrupted = false
    let revoked: Promise<DesktopFrameRevocation> | undefined

    vi.spyOn(storage, 'removeItem').mockImplementation(function (this: Storage, key: string) {
      removeItem.call(this, key)

      if (key.startsWith(`${KEY}:frames:`) && !key.includes(':consent:')) {
        reclaimed = true
      }
    })
    vi.spyOn(storage, 'setItem').mockImplementation(function (this: Storage, key: string, value: string) {
      if (key === `${KEY}:frames` && !reclaimed) {
        throw new DOMException('full', 'QuotaExceededError')
      }

      const journal = JSON.parse(value)

      if (!interrupted && journal.days?.['2026-09-27']?.frames?.gte_5s === 2) {
        interrupted = true
        revoked = DesktopFrameMetrics.purge(KEY)
        expect(reclaimed).toBe(false)
        stale.record(['gte_5s']) // Queued work must observe revocation before allocating.
      }

      setItem.call(this, key, value)
    })
    stale.record(['gte_5s'])
    await locks.drain() // The writer enqueues reclamation while this phase is running.
    await locks.drain() // Drain those newly enqueued phases as well.
    await expect(revoked).resolves.toMatchObject({ complete: true })
    expect(interrupted).toBe(true)
    expect(reclaimed).toBe(true)
    expect(stale.isCurrent()).toBe(false)
    const replacement = new DesktopFrameMetrics(KEY)
    const sink = backend()

    replacement.track(11 * MINUTE)
    replacement.record(['50ms_to_100ms'])
    vi.setSystemTime(MIDNIGHT - 3 * MINUTE)
    replacement.track(0)
    vi.setSystemTime(MIDNIGHT)
    await stale.flush(sink.request, current)
    await replacement.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: MINUTE, frames: [{ bucket: '50ms_to_100ms', count: 1 }]
    }])
  })

  it.each([false, true])('awaits queued source publications before rename (Web Locks: %s)', async nativeLocks => {
    if (nativeLocks) {
      serialLocks()
    } else {
      closeIndexedDb = indexedDbMetricsLocks().close
    }

    const stale = new DesktopFrameMetrics(KEY)
    const destination = `${KEY}-renamed`
    const sink = backend()

    stale.track(11 * MINUTE)
    stale.record(['100ms_to_250ms'])
    vi.setSystemTime(MIDNIGHT - 2 * MINUTE)
    stale.track(0)
    await DesktopFrameMetrics.migrate(KEY, destination)
    stale.record(['gte_5s'])
    expect(stale.isCurrent()).toBe(false)
    const renamed = new DesktopFrameMetrics(destination)

    vi.setSystemTime(MIDNIGHT)
    await stale.flush(sink.request, current)
    await renamed.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: 2 * MINUTE, frames: [{ bucket: '100ms_to_250ms', count: 1 }]
    }])
    expect(storedKeys().filter(key => key.startsWith(`${KEY}:frames:`) && !key.includes(':consent:'))).toEqual([])
  })

  it.each(['journal', 'daily', 'withdrawn'])('isolates the old cohort after an interrupted copy (%s)', async failedPart => {
    const stale = new DesktopFrameMetrics(KEY)
    const destination = `${KEY}-renamed`
    const originalDaily = { today: { day: '2026-09-27', actions: { 'composer.send:click': 7 } } }
    const recreatedDaily = { today: { day: '2026-09-27', actions: { 'composer.send:click': 2 } } }
    const sourceSink = backend()
    const destinationSink = backend()

    stale.track(11 * MINUTE)
    stale.record(['100ms_to_250ms'])
    vi.setSystemTime(MIDNIGHT - 2 * MINUTE)
    stale.track(0)
    localStorage.setItem(KEY, JSON.stringify(originalDaily))
    const captured = DesktopFrameMetrics.capture(KEY)
    const storage = pageStorage()
    const setItem = storage.setItem

    const failingWrite = vi.spyOn(storage, 'setItem').mockImplementation(function (this: Storage, key, value) {
      const journal = key.startsWith(`${destination}:frames:`) && !key.includes(':import:') && !key.includes(':consent:')

      if (failedPart === 'daily' ? key === destination : journal) {
        throw new DOMException('full', 'QuotaExceededError')
      }

      setItem.call(this, key, value)
    })

    await expect(DesktopFrameMetrics.migrate(KEY, destination)).resolves.toBeUndefined()
    expect(stale.isCurrent()).toBe(false)
    expect(localStorage.getItem(KEY)).toBeNull()
    expect(storedKeys().some(key => key.startsWith(`${destination}:frames:import:`))).toBe(true)
    const recreated = new DesktopFrameMetrics(KEY)

    recreated.track(11 * MINUTE)
    recreated.record(['50ms_to_100ms'])
    localStorage.setItem(KEY, JSON.stringify(recreatedDaily))
    vi.setSystemTime(MIDNIGHT - MINUTE)
    recreated.track(0)
    failingWrite.mockRestore()

    if (failedPart === 'journal') {
      await DesktopFrameMetrics.migrate(KEY, destination)
    } else if (failedPart === 'withdrawn') {
      await DesktopFrameMetrics.purge(KEY, captured)
    }

    // A new collector repairs the durable quarantine, including a partially copied daily record.
    const renamed = new DesktopFrameMetrics(destination)

    await renamed.ready
    expect(JSON.parse(localStorage.getItem(KEY)!)).toEqual(recreatedDaily)
    expect(JSON.parse(localStorage.getItem(destination)!)).toEqual(failedPart === 'withdrawn' ? null : originalDaily)
    vi.setSystemTime(MIDNIGHT)
    await stale.flush(destinationSink.request, current)
    await renamed.flush(destinationSink.request, current)
    await recreated.flush(sourceSink.request, current)
    expect([...destinationSink.accepted.values()]).toEqual(failedPart === 'withdrawn' ? [] : [{
      day: '2026-09-27', visible_ms: 2 * MINUTE, frames: [{ bucket: '100ms_to_250ms', count: 1 }]
    }])
    expect([...sourceSink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: MINUTE, frames: [{ bucket: '50ms_to_100ms', count: 1 }]
    }])
  })

  it.each(['token', 'quarantine'])('discards only optional old measurements when the %s cannot be stored', async failedPart => {
    const stale = new DesktopFrameMetrics(KEY)
    const destination = `${KEY}-renamed`
    const sink = backend()

    stale.track(11 * MINUTE)
    stale.record(['gte_5s'])
    const captured = DesktopFrameMetrics.capture(KEY)
    const storage = pageStorage()
    const setItem = storage.setItem

    const failingWrite = vi.spyOn(storage, 'setItem').mockImplementation(function (this: Storage, key, value) {
      if (failedPart === 'token' ? key === `${destination}:frames` : key.startsWith(`${destination}:frames:import:`)) {
        throw new DOMException('full', 'QuotaExceededError')
      }

      setItem.call(this, key, value)
    })

    localStorage.setItem('hermes.desktop.sessionTiles.v2', JSON.stringify({ alpha: ['session-1'] }))
    await expect(DesktopFrameMetrics.migrate(KEY, destination)).resolves.toBeUndefined()
    expect(localStorage.getItem(captured.consent)).toBeNull()
    expect(stale.isCurrent()).toBe(false)
    expect(JSON.parse(localStorage.getItem('hermes.desktop.sessionTiles.v2')!)).toEqual({ alpha: ['session-1'] })
    failingWrite.mockRestore()
    const renamed = new DesktopFrameMetrics(destination)

    renamed.track(11 * MINUTE)
    renamed.record(['50ms_to_100ms'])
    vi.setSystemTime(MIDNIGHT - 3 * MINUTE)
    renamed.track(0)
    vi.setSystemTime(MIDNIGHT)
    await renamed.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: MINUTE, frames: [{ bucket: '50ms_to_100ms', count: 1 }]
    }])
  })

  it('verifies the destination before quota reclamation can remove the source', async () => {
    const stale = new DesktopFrameMetrics(KEY)
    const destination = `${KEY}-renamed`
    const sink = backend()

    stale.track(11 * MINUTE)
    stale.record(['100ms_to_250ms'])
    vi.setSystemTime(MIDNIGHT - 2 * MINUTE)
    stale.track(0)
    const storage = pageStorage()
    const setItem = storage.setItem
    const removeItem = storage.removeItem
    let reclaimed = false

    vi.spyOn(storage, 'setItem').mockImplementation(function (this: Storage, key, value) {
      if (key === `${KEY}:frames` && !reclaimed) {
        throw new DOMException('full', 'QuotaExceededError')
      }

      setItem.call(this, key, value)
    })
    vi.spyOn(storage, 'removeItem').mockImplementation(function (this: Storage, key) {
      if (key.startsWith(`${KEY}:frames:`)) {
        expect(localStorage.getItem(`${destination}:frames`)).not.toBeNull()
        expect(storedKeys().some(candidate => candidate.startsWith(`${destination}:frames:`))).toBe(true)
        reclaimed = true
      }

      removeItem.call(this, key)
    })
    await DesktopFrameMetrics.migrate(KEY, destination)
    expect(reclaimed).toBe(true)
    expect(stale.isCurrent()).toBe(false)
    const renamed = new DesktopFrameMetrics(destination)

    vi.setSystemTime(MIDNIGHT)
    await renamed.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: 2 * MINUTE, frames: [{ bucket: '100ms_to_250ms', count: 1 }]
    }])
  })

  it.each(['destination', 'source'])('keeps queued withdrawal authoritative at the rename %s-token write', async phase => {
    const stale = new DesktopFrameMetrics(KEY)
    const destination = `${KEY}-renamed`
    const sink = backend()

    stale.track(11 * MINUTE)
    stale.record(['gte_5s'])
    vi.setSystemTime(MIDNIGHT - 2 * MINUTE)
    stale.track(0)
    const storage = pageStorage()
    const setItem = storage.setItem
    let withdrew = false

    vi.spyOn(storage, 'setItem').mockImplementation(function (this: Storage, key, value) {
      if (!withdrew && key === `${phase === 'destination' ? destination : KEY}:frames`) {
        withdrew = true
        DesktopFrameMetrics.purge(KEY)
      }

      setItem.call(this, key, value)
    })
    await DesktopFrameMetrics.migrate(KEY, destination)
    expect(withdrew).toBe(true)
    expect(stale.isCurrent()).toBe(false)
    const authorizedAgain = new DesktopFrameMetrics(destination)

    authorizedAgain.track(11 * MINUTE)
    authorizedAgain.record(['50ms_to_100ms'])
    vi.setSystemTime(MIDNIGHT - MINUTE)
    authorizedAgain.track(0)
    vi.setSystemTime(MIDNIGHT)
    await authorizedAgain.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: MINUTE, frames: [{ bucket: '50ms_to_100ms', count: 1 }]
    }])
  })

  it('retires an unpublished old-name collector without stranding a freshly authorized recreation', async () => {
    const stale = new DesktopFrameMetrics(KEY)
    const sink = backend()

    await DesktopFrameMetrics.migrate(KEY, `${KEY}-renamed`)
    stale.track(11 * MINUTE)
    stale.record(['gte_5s'])
    expect(stale.isCurrent()).toBe(false)
    const recreated = new DesktopFrameMetrics(KEY)

    recreated.track(11 * MINUTE)
    recreated.record(['50ms_to_100ms'])
    vi.setSystemTime(MIDNIGHT - 3 * MINUTE)
    recreated.track(0)
    vi.setSystemTime(MIDNIGHT)
    await stale.flush(sink.request, current)
    await recreated.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: MINUTE, frames: [{ bucket: '50ms_to_100ms', count: 1 }]
    }])
  })

  it('does not inherit a deleted destination profile when the renamed source has no metric data', async () => {
    const destination = `${KEY}-renamed`
    const deletedProfile = new DesktopFrameMetrics(destination)
    const sink = backend()

    deletedProfile.track(11 * MINUTE)
    deletedProfile.record(['gte_5s'])
    vi.setSystemTime(MIDNIGHT - 2 * MINUTE)
    deletedProfile.track(0)
    await DesktopFrameMetrics.migrate(KEY, destination)
    expect(deletedProfile.isCurrent()).toBe(false)
    const renamed = new DesktopFrameMetrics(destination)

    renamed.track(11 * MINUTE)
    renamed.record(['50ms_to_100ms'])
    vi.setSystemTime(MIDNIGHT - MINUTE)
    renamed.track(0)
    vi.setSystemTime(MIDNIGHT)
    await renamed.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: MINUTE, frames: [{ bucket: '50ms_to_100ms', count: 1 }]
    }])
  })

  it('discards the old optional cohort when rename coordination aborts without failing the rename', async () => {
    const coordination = indexedDbMetricsLocks()

    closeIndexedDb = coordination.close
    const stale = new DesktopFrameMetrics(KEY)
    const destination = `${KEY}-renamed`
    const sourceSink = backend()
    const destinationSink = backend()

    stale.track(11 * MINUTE)
    stale.record(['gte_5s'])
    vi.setSystemTime(MIDNIGHT - 2 * MINUTE)
    stale.track(0)
    await stale.flush(sourceSink.request, current)
    const captured = DesktopFrameMetrics.capture(KEY)

    coordination.abortNext()
    await expect(DesktopFrameMetrics.migrate(KEY, destination)).resolves.toBeUndefined()
    expect(localStorage.getItem(captured.consent)).toBeNull()
    expect(stale.isCurrent()).toBe(false)
    const recreated = new DesktopFrameMetrics(KEY, await DesktopFrameMetrics.captureCoordinated(KEY))
    const renamed = new DesktopFrameMetrics(destination)

    recreated.track(11 * MINUTE)
    recreated.record(['50ms_to_100ms'])
    vi.setSystemTime(MIDNIGHT - MINUTE)
    recreated.track(0)
    vi.setSystemTime(MIDNIGHT)
    await stale.flush(sourceSink.request, current)
    await recreated.flush(sourceSink.request, current)
    await renamed.flush(destinationSink.request, current)
    expect([...sourceSink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: MINUTE, frames: [{ bucket: '50ms_to_100ms', count: 1 }]
    }])
    expect([...destinationSink.accepted.values()]).toEqual([])
  })

  it('cannot recreate a captured permit removed before a descheduled first initializer runs', async () => {
    const captured = await DesktopFrameMetrics.captureCoordinated(KEY)
    const held = Promise.withResolvers<void>()
    const ownership = serialLocks(held.promise)
    const stale = new DesktopFrameMetrics(KEY, captured)

    stale.track(11 * MINUTE)
    stale.record(['gte_5s'])
    vi.stubGlobal('navigator', { locks: {
      request: () => Promise.reject(new Error('coordinator unavailable'))
    } })
    await expect(DesktopFrameMetrics.purge(KEY, captured)).resolves.toMatchObject({ complete: false })
    expect(localStorage.getItem(captured.consent)).toBeNull()
    immediateMetricsLocks()
    const fresh = new DesktopFrameMetrics(KEY, await DesktopFrameMetrics.captureCoordinated(KEY))
    const sink = backend()

    fresh.track(11 * MINUTE)
    fresh.record(['50ms_to_100ms'])
    held.resolve()
    await ownership.drain()
    expect(stale.isCurrent()).toBe(false)
    vi.setSystemTime(MIDNIGHT - 3 * MINUTE)
    fresh.track(0)
    vi.setSystemTime(MIDNIGHT)
    await stale.flush(sink.request, current)
    await fresh.flush(sink.request, current)
    expect([...sink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: MINUTE, frames: [{ bucket: '50ms_to_100ms', count: 1 }]
    }])
  })

  it('removes only the captured permit after coordinator failure across moves and a recreated old name', async () => {
    const stale = new DesktopFrameMetrics(KEY)
    const middle = `${KEY}-middle`
    const destination = `${KEY}-renamed`
    const sourceSink = backend()
    const destinationSink = backend()

    stale.track(11 * MINUTE)
    stale.record(['gte_5s'])
    const captured = DesktopFrameMetrics.capture(KEY)

    vi.setSystemTime(MIDNIGHT - 2 * MINUTE)
    stale.track(0)
    await DesktopFrameMetrics.migrate(KEY, middle)
    await DesktopFrameMetrics.migrate(middle, destination)
    const recreated = new DesktopFrameMetrics(KEY)

    recreated.track(11 * MINUTE)
    recreated.record(['50ms_to_100ms'])
    const freshSource = DesktopFrameMetrics.capture(KEY)

    const failCoordination = () => vi.stubGlobal('navigator', { locks: {
      request: () => Promise.reject(new Error('coordinator unavailable'))
    } })

    failCoordination()
    await expect(DesktopFrameMetrics.purge(KEY, captured)).resolves.toMatchObject({ complete: false })
    expect(localStorage.getItem(captured.consent)).toBeNull()
    expect(localStorage.getItem(freshSource.consent)).toBe('1')
    immediateMetricsLocks()
    const renamed = new DesktopFrameMetrics(destination, await DesktopFrameMetrics.captureCoordinated(destination))

    renamed.track(11 * MINUTE)
    renamed.record(['100ms_to_250ms'])
    const freshDestination = DesktopFrameMetrics.capture(destination)

    failCoordination()
    await expect(DesktopFrameMetrics.purge(KEY, captured)).resolves.toMatchObject({ complete: false })
    expect(localStorage.getItem(freshDestination.consent)).toBe('1')
    immediateMetricsLocks()
    vi.setSystemTime(MIDNIGHT - MINUTE)
    recreated.track(0)
    renamed.track(0)
    vi.setSystemTime(MIDNIGHT)
    await recreated.flush(sourceSink.request, current)
    await renamed.flush(destinationSink.request, current)
    expect([...sourceSink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: MINUTE, frames: [{ bucket: '50ms_to_100ms', count: 1 }]
    }])
    expect([...destinationSink.accepted.values()]).toEqual([{
      day: '2026-09-27', visible_ms: MINUTE, frames: [{ bucket: '100ms_to_250ms', count: 1 }]
    }])
  })
})
