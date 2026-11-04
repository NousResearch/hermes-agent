import { beforeEach, describe, expect, it } from 'vitest'

import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import type { SessionTile } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { ownerRouteFromKey, ownerRouteKey, tileOwnerRoute } from './session-tile-owner'

const row = (over: Partial<SessionInfo>): SessionInfo => over as SessionInfo

const tile = (over: Partial<SessionTile> & Pick<SessionTile, 'storedSessionId'>): SessionTile => over as SessionTile

describe('tileOwnerRoute', () => {
  beforeEach(() => {
    _resetSessionOwnerHintsForTests()
  })

  it('prefers the tile own explicit route', () => {
    const route = tileOwnerRoute(
      [tile({ ownerRoute: { connectionId: 'pandora', profile: 'work' }, storedSessionId: 's1' })],
      [row({ connection_id: 'other-box', id: 's1', profile: 'default' })],
      's1'
    )

    expect(route).toEqual({ connectionId: 'pandora', profile: 'work' })
  })

  it('falls back to the session row owner when the tile carries no route', () => {
    // How a branch child is opened: openSessionTile with no workspaceScope, so
    // the tile route alone leaves the owner undefined and every RPC drops to
    // the ambient socket.
    const route = tileOwnerRoute(
      [tile({ storedSessionId: 's1' })],
      [row({ connection_id: 'rigremote', id: 's1', profile: 'default' })],
      's1'
    )

    expect(route).toEqual({ connectionId: 'rigremote', profile: 'default' })
  })

  it('falls back to the owner hint when neither tile nor row is tagged', () => {
    setSessionOwnerHint('s1', { connectionId: 'pandora', profile: 'work' })

    expect(tileOwnerRoute([tile({ storedSessionId: 's1' })], [], 's1')).toMatchObject({ connectionId: 'pandora' })
  })

  it('carries a targetProfile through, and omits it when absent', () => {
    const routed = tileOwnerRoute(
      [tile({ ownerRoute: { connectionId: 'pandora', profile: 'work', targetProfile: 'ceo' }, storedSessionId: 's1' })],
      [],
      's1'
    )

    expect(routed).toEqual({ connectionId: 'pandora', profile: 'work', targetProfile: 'ceo' })
    expect(
      tileOwnerRoute([tile({ ownerRoute: { connectionId: 'p', profile: 'w' }, storedSessionId: 's1' })], [], 's1')
    ).not.toHaveProperty('targetProfile')
  })

  it('narrows a bare profile owner away', () => {
    // knownSessionOwner returns a bare profile string for a row that names a
    // profile but no connection. It carries no backend identity, so handing it
    // on as a route would resolve against whichever connection is active.
    expect(tileOwnerRoute([], [row({ id: 's1', profile: 'work' })], 's1')).toBeUndefined()
  })

  it('is undefined for an untagged session, preserving ambient routing', () => {
    expect(tileOwnerRoute([], [row({ id: 's1' })], 's1')).toBeUndefined()
    expect(tileOwnerRoute([], [], 'missing')).toBeUndefined()
  })
})

describe('tileOwnerRoute key stability under session-list churn', () => {
  it('keeps the current tile key when an UNRELATED row changes (sessions.changed tick)', () => {
    const tiles = [tile({ storedSessionId: 'mine' })]
    const mine = { connection_id: 'local', id: 'mine', profile: 'default' }
    // The only difference is an unrelated row's last_active moving — exactly
    // what a sessions.changed broadcast republishes for every profile.
    const rowsBefore: SessionInfo[] = [row({ id: 'other-session', last_active: 100 }), row(mine)]
    const rowsAfter: SessionInfo[] = [row({ id: 'other-session', last_active: 9_999 }), row(mine)]

    expect(ownerRouteKey(tileOwnerRoute(tiles, rowsBefore, 'mine'))).toBe(
      ownerRouteKey(tileOwnerRoute(tiles, rowsAfter, 'mine'))
    )
  })

  it('changes the key when THIS tile owner actually re-homes', () => {
    const tiles = [tile({ storedSessionId: 'mine' })]

    expect(
      ownerRouteKey(tileOwnerRoute(tiles, [row({ connection_id: 'local', id: 'mine', profile: 'default' })], 'mine'))
    ).not.toBe(
      ownerRouteKey(tileOwnerRoute(tiles, [row({ connection_id: 'remote', id: 'mine', profile: 'default' })], 'mine'))
    )
    // A target profile is part of the owner identity too, and an unresolved
    // owner has no key at all.
    expect(ownerRouteKey({ connectionId: 'local', profile: 'default' })).not.toBe(
      ownerRouteKey({ connectionId: 'local', profile: 'default', targetProfile: 'tech-review' })
    )
    expect(ownerRouteKey(undefined)).toBeNull()

    // The component rebuilds the route from the key; encoder and decoder must round-trip.
    for (const route of [
      { connectionId: 'local', profile: 'default' },
      { connectionId: 'local', profile: 'default', targetProfile: 'tech-review' }
    ]) {
      expect(ownerRouteFromKey(ownerRouteKey(route))).toEqual(route)
    }

    expect(ownerRouteFromKey(null)).toBeUndefined()
  })
})
