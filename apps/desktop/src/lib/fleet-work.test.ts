import { describe, expect, it } from 'vitest'

import type { DesktopAgentWork } from '@/global'

import { FLEET_WORK_STALE_MS, FLEET_WORKER_ACTIVE_WINDOW_S, fleetWorkState } from './fleet-work'

const now = 1_800_000_000_000

const work = (lastActiveSeconds: number): DesktopAgentWork => ({
  lastActive: lastActiveSeconds,
  sessionId: 'sess-1',
  source: 'kanban'
})

describe('fleetWorkState', () => {
  it('is idle without a worker claim', () => {
    expect(fleetWorkState(undefined, now, now)).toBe('idle')
  })

  it('is fresh while the heartbeat is inside the liveness window and the roster is recent', () => {
    expect(fleetWorkState(work(now / 1000 - 10), now - 5_000, now)).toBe('fresh')
  })

  it('decays to idle from the heartbeat itself, not from the fetch time', () => {
    // A poll that keeps succeeding cannot keep a dot green: the row stopped
    // heartbeating, so the claim expired on its own clock.
    const staleHeartbeat = now / 1000 - FLEET_WORKER_ACTIVE_WINDOW_S - 1

    expect(fleetWorkState(work(staleHeartbeat), now - 1_000, now)).toBe('idle')
  })

  it('treats a heartbeat exactly at the window edge as expired', () => {
    expect(fleetWorkState(work(now / 1000 - FLEET_WORKER_ACTIVE_WINDOW_S), now, now)).toBe('idle')
  })

  it('marks a fresh heartbeat stale when the roster page is old', () => {
    expect(fleetWorkState(work(now / 1000 - 10), now - FLEET_WORK_STALE_MS - 1, now)).toBe('stale')
  })

  it('marks a fresh heartbeat stale when the roster was never stamped', () => {
    expect(fleetWorkState(work(now / 1000 - 10), 0, now)).toBe('stale')
  })

  it('keeps a claim fresh right up to the stale window', () => {
    expect(fleetWorkState(work(now / 1000 - 10), now - FLEET_WORK_STALE_MS, now)).toBe('fresh')
  })

  it('is idle for a non-finite heartbeat instead of painting a guess', () => {
    expect(fleetWorkState({ lastActive: Number.NaN, sessionId: 'x', source: 'kanban' }, now, now)).toBe('idle')
  })
})
