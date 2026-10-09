import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { getSession } from '@/hermes'
import { clearSessionDraft, stashSessionDraft } from '@/store/composer'
import { $gatewaySwitching } from '@/store/gateway-switch'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $connection, $gatewayState, $sessions, setSessions } from '@/store/session'
import { $sessionTiles, openSessionTile, reopenLastClosedTile, type SessionTile } from '@/store/session-states'

import {
  sessionTileResumeFailure,
  shouldResumeSessionTile,
  startTileBackendIdentityGuard,
  startUnrestoredTileTitleBackfill,
  unbindTilesForBackendIdentityChange,
  WRONG_BACKEND_TILE_ERROR
} from './session-tile'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getSession: vi.fn()
}))

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

describe('startUnrestoredTileTitleBackfill retires dead tiles (#125678)', () => {
  const NOT_FOUND = '404: {"detail":"Session not found"}'
  const get = vi.mocked(getSession)
  let stop: (() => void) | undefined

  beforeEach(() => {
    $gatewayState.set('idle')
    $activeGatewayProfile.set('default')
    // Connection first: a connection change re-scopes the profile inventory.
    $connection.set({ connectionId: 'local', mode: 'local' } as never)
    $profiles.set([{ name: 'default' }, { name: 'writer' }] as never)
    get.mockReset()
  })

  afterEach(() => {
    stop?.()
    $gatewayState.set('idle')
    $gatewaySwitching.set(false)
    $sessionTiles.set([])
    $profiles.set([])
    $connection.set(null)
    setSessions([])
    clearSessionDraft('dead-chat')
    window.localStorage.clear()
  })

  it('drops a restored tile once every profile answered 404 in calm conditions, off the reopen stack', async () => {
    openSessionTile('dead-chat')
    get.mockRejectedValue(new Error(NOT_FOUND))

    stop = startUnrestoredTileTitleBackfill()
    $gatewayState.set('open')

    await vi.waitFor(() => expect($sessionTiles.get()).toEqual([]))
    expect(get.mock.calls.map(call => call[1])).toEqual(['default', 'writer'])
    expect(window.localStorage.getItem('hermes.desktop.sessionTiles.v2') ?? '').not.toContain('dead-chat')
    reopenLastClosedTile()
    expect($sessionTiles.get()).toEqual([])
  })

  it.each([
    '500 on one profile',
    'network failure on one profile',
    'gateway switch in flight',
    'profile A→B→A while the probes are out',
    'connection switch while the probes are out',
    'stashed draft text',
    'profile inventory not loaded'
  ])('keeps the tile when absence is not conclusive: %s', async reason => {
    openSessionTile('dead-chat')

    if (reason === 'gateway switch in flight') {
      $gatewaySwitching.set(true)
    }

    if (reason === 'stashed draft text') {
      stashSessionDraft('dead-chat', 'keep my words', [])
    }

    if (reason === 'profile inventory not loaded') {
      $profiles.set([])
    }

    let settleFirst!: (error: Error) => void

    const first = new Promise<never>((_resolve, reject) => {
      settleFirst = reject
    })

    get.mockRejectedValue(new Error(NOT_FOUND)).mockImplementationOnce(() => first)

    stop = startUnrestoredTileTitleBackfill()
    $gatewayState.set('open')
    await vi.waitFor(() => expect(get).toHaveBeenCalled())

    if (reason === 'profile A→B→A while the probes are out') {
      $activeGatewayProfile.set('writer')
      $activeGatewayProfile.set('default')
    }

    if (reason === 'connection switch while the probes are out') {
      $connection.set({ connectionId: 'remote', mode: 'remote' } as never)
      $connection.set({ connectionId: 'local', mode: 'local' } as never)
    }

    settleFirst(
      new Error(
        reason === '500 on one profile'
          ? '500: {"detail":"Session not found"}'
          : reason === 'network failure on one profile'
            ? 'net::ERR_CONNECTION_REFUSED'
            : NOT_FOUND
      )
    )

    // Let the ladder finish (and the retire branch run) before asserting. A
    // connection switch re-scopes the inventory, so its ladder stops at one rung.
    const rungs = ['profile inventory not loaded', 'connection switch while the probes are out'].includes(reason)
      ? 1
      : 2

    await vi.waitFor(() => expect(get).toHaveBeenCalledTimes(rungs))
    await new Promise(resolve => setTimeout(resolve, 0))
    expect($sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['dead-chat'])
  })
})
