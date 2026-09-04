import { afterEach, describe, expect, it, vi } from 'vitest'

import { $gatewayState, $sessions, setSessions } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

import { sessionTileResumeFailure, shouldResumeSessionTile, startUnrestoredTileTitleBackfill } from './session-tile'

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
