import { beforeEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $sessionStates,
  $sessionTiles,
  clearAllSessionStates,
  dropSessionState,
  liveSessionScopes,
  openTileGatewayScopes,
  publishSessionState,
  recordSessionEventScope
} from '@/store/session-states'

/**
 * The (connectionId, profile) half of the gateway keep-set. Working/attention
 * ids are profile-blind, and every registered source exposes a 'default'
 * profile — so registry-sourced live work must surface as composite
 * backendScopeKey scopes, while untagged local/primary events contribute
 * nothing (their liveness keeps flowing through bare profile names).
 */

const state = (patch: Partial<ReturnType<typeof createClientSessionState>> = {}) => ({
  ...createClientSessionState('stored-1'),
  ...patch
})

beforeEach(() => {
  clearAllSessionStates()
  $sessionStates.set({})
})

describe('liveSessionScopes', () => {
  it('maps a registry-tagged busy session to its composite scope', () => {
    recordSessionEventScope({ connectionId: 'homelab', profile: 'default', session_id: 'rt-1' })
    publishSessionState('rt-1', state({ busy: true }))

    expect(liveSessionScopes()).toEqual(new Set(['conn:homelab::default']))
  })

  it('keeps an explicit local registry session on its composite scope', () => {
    recordSessionEventScope({ connectionId: 'local', profile: 'default', session_id: 'rt-local' })
    publishSessionState('rt-local', state({ busy: true }))

    expect(liveSessionScopes()).toEqual(new Set(['conn:local::default']))
  })

  it('includes needs-input sessions and drops settled ones', () => {
    recordSessionEventScope({ connectionId: 'homelab', profile: 'default', session_id: 'rt-1' })
    publishSessionState('rt-1', state({ busy: false, needsInput: true }))

    expect(liveSessionScopes()).toEqual(new Set(['conn:homelab::default']))

    publishSessionState('rt-1', state({ busy: false, needsInput: false }))

    expect(liveSessionScopes()).toEqual(new Set())
  })

  it('ignores untagged (local/primary) events — no connectionId, no scope', () => {
    recordSessionEventScope({ profile: 'default', session_id: 'rt-1' })
    publishSessionState('rt-1', state({ busy: true }))

    expect(liveSessionScopes()).toEqual(new Set())
  })

  it("keeps two sources' same-named 'default' profiles distinct", () => {
    recordSessionEventScope({ connectionId: 'homelab', profile: 'default', session_id: 'rt-a' })
    recordSessionEventScope({ connectionId: 'spark', profile: 'default', session_id: 'rt-b' })
    publishSessionState('rt-a', state({ busy: true }))
    publishSessionState('rt-b', state({ busy: true }))

    expect(liveSessionScopes()).toEqual(new Set(['conn:homelab::default', 'conn:spark::default']))
  })

  it('forgets a dropped runtime session', () => {
    recordSessionEventScope({ connectionId: 'homelab', profile: 'default', session_id: 'rt-1' })
    publishSessionState('rt-1', state({ busy: true }))
    dropSessionState('rt-1')
    publishSessionState('rt-1', state({ busy: true }))

    expect(liveSessionScopes()).toEqual(new Set())
  })
})

describe('openTileGatewayScopes', () => {
  beforeEach(() => {
    $sessionTiles.set([])
  })

  afterEach(() => {
    $sessionTiles.set([])
  })

  it('keeps a local bot tile on both the bare profile and the explicit local registry scope', () => {
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'local', mode: 'local', profile: 'berry' },
        storedSessionId: 'bot-chat-berry'
      }
    ])

    expect(openTileGatewayScopes()).toEqual(new Set(['berry', 'conn:local::berry']))
  })

  it('keeps a remote tile on its composite scope only', () => {
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'homelab', profile: 'default' },
        storedSessionId: 'bot-chat-homelab'
      }
    ])

    expect(openTileGatewayScopes()).toEqual(new Set(['conn:homelab::default']))
    expect(openTileGatewayScopes().has('default')).toBe(false)
  })

  it('keys the keep-set on route.profile, not a remapped targetProfile', () => {
    // openGatewayForAgent dials (connectionId, profile). targetProfile only
    // rewrites RPC params — using it here would miss the live socket.
    $sessionTiles.set([
      {
        ownerRoute: {
          connectionId: 'barry',
          profile: 'oxcoder',
          targetProfile: 'backend-oxcoder'
        },
        storedSessionId: 'bot-chat-oxcoder'
      }
    ])

    expect(openTileGatewayScopes()).toEqual(new Set(['conn:barry::oxcoder']))
  })

  it('ignores tiles without an owner route', () => {
    $sessionTiles.set([{ storedSessionId: 'plain' }])

    expect(openTileGatewayScopes()).toEqual(new Set())
  })
})
