import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { $connection, $gatewayState, $sessions, setSessions } from '@/store/session'
import { $sessionTiles, type SessionTile } from '@/store/session-states'

import {
  createTileResumeBudget,
  recentTileResumes,
  sessionTileResumeFailure,
  shouldResumeSessionTile,
  startTileBackendIdentityGuard,
  startUnrestoredTileTitleBackfill,
  TILE_RESUME_STORM_LIMIT,
  TILE_RESUME_STORM_WINDOW_MS,
  tileResumeStormed,
  unbindTilesForBackendIdentityChange,
  WRONG_BACKEND_TILE_ERROR
} from './session-tile'

function localConnection(): HermesConnection {
  return {
    baseUrl: 'http://127.0.0.1:9119',
    isFullscreen: false,
    logs: [],
    mode: 'local',
    nativeOverlayWidth: 0,
    token: 'test',
    windowButtonPosition: null,
    wsUrl: 'ws://127.0.0.1:9119'
  }
}

describe('shouldResumeSessionTile', () => {
  const live = {
    gatewayOpen: true,
    removalPending: false,
    resuming: false,
    runtimeId: null,
    tileError: undefined
  }

  it('resumes an unbound tile once the gateway is open', () => {
    expect(shouldResumeSessionTile(live)).toBe(true)
  })

  it('does not resume a session the user is deleting', () => {
    // A 4001 racing the delete unbinds the tile runtime, re-arming the resume
    // effect against an id that is already gone: the resume 404s and latches an
    // error card for a chat that is on its way out.
    expect(shouldResumeSessionTile({ ...live, removalPending: true })).toBe(false)
  })

  it('waits for the gateway, a free slot, and an unbound, unlatched tile', () => {
    expect(shouldResumeSessionTile({ ...live, gatewayOpen: false })).toBe(false)
    expect(shouldResumeSessionTile({ ...live, runtimeId: 'rt-1' })).toBe(false)
    expect(shouldResumeSessionTile({ ...live, tileError: 'boom' })).toBe(false)
    expect(shouldResumeSessionTile({ ...live, resuming: true })).toBe(false)
  })
})

describe('sessionTileResumeFailure', () => {
  it('keeps a confirmed durable session retryable instead of repeating a stale 404', () => {
    expect(sessionTileResumeFailure('session not found', true, true)).toBe(
      'Session is still available — retry resuming it.'
    )
  })

  it('fails safe on an inconclusive durable lookup', () => {
    expect(sessionTileResumeFailure('404', false, true)).toBe('Session unavailable — you can retry resuming it.')
  })

  it('does not overwrite a tile that rebound while the lookup was pending', () => {
    expect(sessionTileResumeFailure('session not found', true, false)).toBeUndefined()
  })

  it('unbinds onto a wrong-backend error when the active backend identity changed', () => {
    expect(sessionTileResumeFailure('session not found', true, true, true)).toBe(
      'Wrong backend — this session lives on another connection. Reconnect to that backend to open it.'
    )
    expect(sessionTileResumeFailure('404', true, true, true)).not.toMatch(/still available/i)
  })
})

describe('unbindTilesForBackendIdentityChange', () => {
  it('clears a runtime binding and latches the wrong-backend error when the active backend is not the tile owner', () => {
    const tiles: SessionTile[] = [
      {
        ownerRoute: { connectionId: '100-106-105-2', profile: 'writer' },
        runtimeId: 'rt-ssh',
        storedSessionId: 'ssh-chat'
      },
      {
        ownerRoute: { connectionId: 'local', profile: 'default' },
        runtimeId: 'rt-local',
        storedSessionId: 'local-chat'
      }
    ]

    const next = unbindTilesForBackendIdentityChange(tiles, { mode: 'local' })

    expect(next[0]).toEqual({
      error: WRONG_BACKEND_TILE_ERROR,
      ownerRoute: { connectionId: '100-106-105-2', profile: 'writer' },
      storedSessionId: 'ssh-chat'
    })
    expect(next[1]?.runtimeId).toBe('rt-local')
    expect(next[1]?.error).toBeUndefined()
  })

  it('leaves a durable same-backend miss retryable', () => {
    const tiles: SessionTile[] = [
      { ownerRoute: { connectionId: 'local', profile: 'default' }, storedSessionId: 'local-chat' }
    ]

    expect(unbindTilesForBackendIdentityChange(tiles, { connectionId: 'local', mode: 'local' })).toBe(tiles)
  })
})

describe('startTileBackendIdentityGuard', () => {
  afterEach(() => {
    $connection.set(null)
    $sessionTiles.set([])
  })

  it('unbinds persisted tiles when an unqualified local boot replaces their owner', () => {
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: '100-106-105-2', profile: 'writer' },
        runtimeId: 'rt-ssh',
        storedSessionId: 'ssh-chat'
      }
    ])

    const stop = startTileBackendIdentityGuard()
    $connection.set(localConnection())

    expect($sessionTiles.get()[0]?.error).toBe(WRONG_BACKEND_TILE_ERROR)
    expect($sessionTiles.get()[0]?.runtimeId).toBeUndefined()
    stop()
  })
})

describe('startUnrestoredTileTitleBackfill (#94167)', () => {
  afterEach(() => {
    $gatewayState.set('idle')
    $sessionTiles.set([])
    setSessions([])
  })

  it('backfills unlisted unrestored tiles by id via their ownerRoute once the gateway opens', async () => {
    const ownerRoute = { connectionId: 'conn-a', profile: 'writer' }
    setSessions([{ id: 'listed', title: 'Already listed' } as never])
    $sessionTiles.set([
      { ownerRoute, storedSessionId: 'old-chat' },
      { storedSessionId: 'listed' },
      { runtimeId: 'rt-live', storedSessionId: 'live' },
      { storedSessionId: 'bot', workspaceTabTitle: 'Bot Chat' }
    ])

    const lookup = vi.fn(async (id: string) => {
      const row = { id, title: 'Quarterly review' } as never
      setSessions(prev => [row, ...prev])

      return row
    })

    const stop = startUnrestoredTileTitleBackfill(lookup as never)
    expect(lookup).not.toHaveBeenCalled()

    $gatewayState.set('open')
    await vi.waitFor(() => expect(lookup).toHaveBeenCalledTimes(1))
    expect(lookup).toHaveBeenCalledWith('old-chat', ownerRoute)
    expect($sessions.get().find(row => row.id === 'old-chat')?.title).toBe('Quarterly review')

    // One-shot: a later reconnect does not re-probe.
    $gatewayState.set('idle')
    $gatewayState.set('open')
    expect(lookup).toHaveBeenCalledTimes(1)
    stop()
  })
})

// #93892: the resume chain had per-step timeouts but no overall budget, so a
// runtime that resumed and was reclaimed over and over spun the loader
// forever. These pin the budget helpers and the pane's use of them.
describe('tile resume storm budget', () => {
  it('counts only resumes inside the window', () => {
    const now = 1_000_000
    const stale = now - TILE_RESUME_STORM_WINDOW_MS
    const fresh = now - TILE_RESUME_STORM_WINDOW_MS + 1

    expect(recentTileResumes([stale, fresh, now], now)).toEqual([fresh, now])
  })

  it('trips once the limit of successful resumes lands inside one window', () => {
    const now = 1_000_000
    const underLimit = Array.from({ length: TILE_RESUME_STORM_LIMIT - 1 }, (_, i) => now - i * 1_000)
    const atLimit = Array.from({ length: TILE_RESUME_STORM_LIMIT }, (_, i) => now - i * 1_000)

    expect(tileResumeStormed(underLimit, now)).toBe(false)
    expect(tileResumeStormed(atLimit, now)).toBe(true)
  })

  it('lets an old storm age out of the window', () => {
    const then = 1_000_000
    const storm = Array.from({ length: TILE_RESUME_STORM_LIMIT }, (_, i) => then - i * 1_000)

    expect(tileResumeStormed(storm, then)).toBe(true)
    expect(tileResumeStormed(storm, then + TILE_RESUME_STORM_WINDOW_MS)).toBe(false)
  })

  it('createTileResumeBudget: successes spend it, the next take past the limit latches, Retry starts clean', () => {
    let now = 1_000_000
    const budget = createTileResumeBudget(() => now)

    // The reclaim loop: each cycle resumes (take + spend) ~20s apart.
    for (let cycle = 0; cycle < TILE_RESUME_STORM_LIMIT; cycle += 1) {
      expect(budget.take()).toBe(true)
      budget.spend()
      now += 20_000
    }

    // One more cycle inside the window: the pane must latch, not dial.
    expect(budget.take()).toBe(false)

    // The user's Retry (or a gateway reopen) gets a clean budget.
    expect(budget.take()).toBe(true)
  })

  it('createTileResumeBudget: failed resumes do not spend it', () => {
    const budget = createTileResumeBudget(() => 1_000_000)

    for (let attempt = 0; attempt < TILE_RESUME_STORM_LIMIT * 2; attempt += 1) {
      expect(budget.take()).toBe(true)
    }
  })
})
