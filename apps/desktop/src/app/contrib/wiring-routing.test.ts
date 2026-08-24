import { describe, expect, it } from 'vitest'

import { findStoredIdForRuntimeId, resolveRoutingSessionId } from './wiring-routing'

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
