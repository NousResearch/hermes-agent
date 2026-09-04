import { describe, expect, it } from 'vitest'

import {
  findStoredIdForRuntimeId,
  resolveReplayHydrationTarget,
  resolveRoutingSessionId,
  resolveSessionRpcOwner
} from './wiring-routing'

describe('resolveReplayHydrationTarget', () => {
  const rows = [
    { id: 'same-id', profile: 'default' },
    { connection_id: 'background-b', id: 'same-id', profile: 'default' }
  ]

  it('pins a background replay to its exact connection instead of the ambient source', () => {
    expect(
      resolveReplayHydrationTarget(rows, {
        connectionId: 'background-b',
        profile: 'default'
      })
    ).toEqual({
      row: rows[1],
      scope: { connectionId: 'background-b', profile: 'default' }
    })
  })

  it('fails closed when the known row owner disagrees with the replay source', () => {
    expect(
      resolveReplayHydrationTarget([{ connection_id: 'active-a', id: 'same-id', profile: 'default' }], {
        connectionId: 'background-b',
        profile: 'default'
      })
    ).toBeUndefined()
  })
})

describe('findStoredIdForRuntimeId', () => {
  it('reverse-resolves a runtime id to its stored id', () => {
    const bindings = new Map([
      ['stored-a', 'runtime-a'],
      ['stored-b', 'runtime-b']
    ])

    expect(findStoredIdForRuntimeId(bindings, 'runtime-b')).toBe('stored-b')
  })

  it('returns undefined for an unknown runtime id', () => {
    expect(findStoredIdForRuntimeId(new Map([['stored-a', 'runtime-a']]), 'runtime-x')).toBeUndefined()
    expect(findStoredIdForRuntimeId(new Map(), 'anything')).toBeUndefined()
  })
})

describe('resolveRoutingSessionId', () => {
  const never = (): string | undefined => undefined

  it('routes by the RPC target session, not the focused tile (the Bot Mode misroute)', () => {
    // A bot chat is a background tile: focused/selected point at the DEFAULT
    // chat, but the RPC targets the bot. Routing must follow the RPC's target.
    const routing = resolveRoutingSessionId({
      focusedStoredSessionId: 'default-chat',
      paramSessionId: 'runtime-bot',
      selectedStoredSessionId: 'default-chat',
      storedIdForRuntime: runtimeId => (runtimeId === 'runtime-bot' ? 'stored-bot' : undefined)
    })

    expect(routing).toBe('stored-bot')
  })

  it('treats an unresolved session_id as already a stored id', () => {
    // Several RPCs pass stored ids directly; a runtime miss must not drop back
    // to the focused tile (that reintroduces the misroute).
    const routing = resolveRoutingSessionId({
      focusedStoredSessionId: 'default-chat',
      paramSessionId: 'stored-bot-direct',
      selectedStoredSessionId: 'default-chat',
      storedIdForRuntime: never
    })

    expect(routing).toBe('stored-bot-direct')
  })

  it('falls back to focused then selected when the RPC carries no session_id', () => {
    expect(
      resolveRoutingSessionId({
        focusedStoredSessionId: 'focused',
        paramSessionId: undefined,
        selectedStoredSessionId: 'selected',
        storedIdForRuntime: never
      })
    ).toBe('focused')

    expect(
      resolveRoutingSessionId({
        focusedStoredSessionId: null,
        paramSessionId: undefined,
        selectedStoredSessionId: 'selected',
        storedIdForRuntime: never
      })
    ).toBe('selected')

    expect(
      resolveRoutingSessionId({
        focusedStoredSessionId: null,
        paramSessionId: undefined,
        selectedStoredSessionId: null,
        storedIdForRuntime: never
      })
    ).toBeNull()
  })
})

describe('resolveSessionRpcOwner', () => {
  const none = () => undefined
  const omar = { connectionId: 'local', mode: 'local' as const, profile: 'omar' }
  const homelab = { connectionId: 'homelab', mode: 'remote' as const, profile: 'worker', targetProfile: 'w' }

  it('returns undefined for an RPC with no session (ambient chrome)', () => {
    expect(
      resolveSessionRpcOwner({
        routingSessionId: null,
        sessionOwnerHint: none,
        sessionRowOwner: none,
        tileOwnerRoute: none
      })
    ).toBeUndefined()
  })

  it('prefers the persisted tile owner route over the hint and the row', () => {
    const owner = resolveSessionRpcOwner({
      routingSessionId: 'stored-bot',
      sessionOwnerHint: () => omar,
      sessionRowOwner: () => 'default',
      tileOwnerRoute: () => homelab
    })

    expect(owner).toEqual(homelab)
  })

  it('prefers the exact unique owner hint over the session row profile', () => {
    // The row is presentation state: an optimistic row minted while the
    // ambient profile stayed `default` reads `default` even though the create
    // ran on local::omar. The hint recorded at create time is exact.
    const owner = resolveSessionRpcOwner({
      routingSessionId: 'stored-omar',
      sessionOwnerHint: id => (id === 'stored-omar' ? omar : undefined),
      sessionRowOwner: () => 'default',
      tileOwnerRoute: none
    })

    expect(owner).toEqual(omar)
  })

  it('reconstructs the EXACT owner from a connection-tagged row when the hint is gone (evicted / relaunch)', () => {
    // The bounded hint map is transient. A row tagged with its owning
    // connection (optimistic create row, unified-list splice, or a tag
    // mergeSessionPage carried across a refresh) names the same registry
    // entry, so the second turn still dials the socket that holds the runtime.
    expect(
      resolveSessionRpcOwner({
        routingSessionId: 'stored-omar',
        sessionOwnerHint: none,
        sessionRowOwner: () => ({ connectionId: 'local', profile: 'omar' }),
        tileOwnerRoute: none
      })
    ).toEqual({ connectionId: 'local', profile: 'omar' })

    // The hint still outranks the row when both exist.
    expect(
      resolveSessionRpcOwner({
        routingSessionId: 'stored-omar',
        sessionOwnerHint: () => omar,
        sessionRowOwner: () => ({ connectionId: 'homelab', profile: 'omar' }),
        tileOwnerRoute: none
      })
    ).toEqual(omar)
  })

  it('falls back to the session row profile, then to undefined for the probe', () => {
    expect(
      resolveSessionRpcOwner({
        routingSessionId: 'stored-1',
        sessionOwnerHint: none,
        sessionRowOwner: () => 'coder',
        tileOwnerRoute: none
      })
    ).toBe('coder')

    expect(
      resolveSessionRpcOwner({
        routingSessionId: 'stored-1',
        sessionOwnerHint: none,
        sessionRowOwner: () => '  ',
        tileOwnerRoute: none
      })
    ).toBeUndefined()

    expect(
      resolveSessionRpcOwner({
        routingSessionId: 'stored-1',
        sessionOwnerHint: none,
        sessionRowOwner: () => null,
        tileOwnerRoute: none
      })
    ).toBeUndefined()
  })
})
