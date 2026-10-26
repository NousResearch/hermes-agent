import { beforeEach, describe, expect, it } from 'vitest'

import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import type { SessionTile } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { tileOwnerRoute } from './session-tile-owner'

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
