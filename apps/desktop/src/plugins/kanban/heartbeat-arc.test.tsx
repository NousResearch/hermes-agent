import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { KanbanTask } from './types'
import { arcState, HEARTBEAT_STALE_SECONDS } from './ui'

// A running card must not read "no heartbeat" while its worker is alive. The arc
// used to age the heartbeat against the desktop's own clock with a 120 s window,
// which is inside ordinary jitter (heartbeat cadence 60 s, board refetch 60 s) and
// turned every running card amber whenever the client clock was a few minutes off.

const SERVER_NOW = 1_800_000_000

const card = (over: Partial<KanbanTask> = {}): KanbanTask =>
  ({ id: 't_arc', status: 'running', title: 'a running card', ...over }) as KanbanTask

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('arcState heartbeat freshness', () => {
  it('is not stale for a heartbeat inside the window, aged on the payload clock', () => {
    expect(arcState(card({ last_heartbeat_at: SERVER_NOW - 60, heartbeat_now: SERVER_NOW }), '')).toBe('running')
  })

  it('is not stale two minutes quiet, the old false alarm', () => {
    expect(arcState(card({ last_heartbeat_at: SERVER_NOW - 121, heartbeat_now: SERVER_NOW }), '')).toBe('running')
  })

  it('ignores a client clock that is minutes off when the payload carries its own', () => {
    // The desktop thinks it is ten minutes later than the server, which is exactly the
    // shape that lit every running card amber before the payload clock was used.
    vi.setSystemTime(new Date((SERVER_NOW + 600) * 1000))
    expect(arcState(card({ last_heartbeat_at: SERVER_NOW - 30, heartbeat_now: SERVER_NOW }), '')).toBe('running')
  })

  it('is stale past the window when the server clock says so', () => {
    const past = SERVER_NOW - HEARTBEAT_STALE_SECONDS - 1
    expect(arcState(card({ last_heartbeat_at: past, heartbeat_now: SERVER_NOW }), '')).toBe('stale')
  })

  it('falls back to the local clock when the backend sends no payload stamp', () => {
    vi.setSystemTime(new Date((SERVER_NOW + 400) * 1000))
    expect(arcState(card({ last_heartbeat_at: SERVER_NOW }), '')).toBe('stale')
  })

  it('stays running when no heartbeat was ever recorded', () => {
    expect(arcState(card({ last_heartbeat_at: null, heartbeat_now: SERVER_NOW }), '')).toBe('running')
  })
})
