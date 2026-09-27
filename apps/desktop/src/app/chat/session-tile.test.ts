import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { $connection, $gatewayState, $sessions, setSessions } from '@/store/session'
import { $sessionTiles, type SessionTile } from '@/store/session-states'

import {
  sessionTileResumeFailure,
  shouldResumeSessionTile,
  startTileBackendIdentityGuard,
  startUnrestoredTileTitleBackfill,
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
