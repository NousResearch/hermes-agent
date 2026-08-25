import { afterEach, describe, expect, it } from 'vitest'

import { $sessionTiles, storedSessionIdForRuntimeId } from '@/store/session-states'

// #92687-adjacent Bot Mode misroute: a session RPC (prompt.submit et al.)
// carries its target as a RUNTIME id, while tile owner routes key on the
// STORED id. requestGateway (contrib/wiring) routes by the RPC's own target
// session — which requires this translation. Before the fix it routed by the
// WINDOW's focused tile, so a background bot chat's submit dispatched on
// whatever backend the focused pane owned: the bot ran on the default
// backend, or 4001'd when default didn't hold the session.

describe('storedSessionIdForRuntimeId', () => {
  afterEach(() => {
    $sessionTiles.set([])
  })

  it('maps a runtime id to the stored id of the tile bound to it', () => {
    $sessionTiles.set([
      { runtimeId: 'rt-default', storedSessionId: 'stored-default' },
      { runtimeId: 'rt-developer', storedSessionId: 'stored-developer' }
    ])

    expect(storedSessionIdForRuntimeId('rt-developer')).toBe('stored-developer')
    expect(storedSessionIdForRuntimeId('rt-default')).toBe('stored-default')
  })

  it('passes a stored id through unchanged (callers may hold either identity)', () => {
    $sessionTiles.set([{ runtimeId: 'rt-a', storedSessionId: 'stored-a' }])

    expect(storedSessionIdForRuntimeId('stored-a')).toBe('stored-a')
  })

  it('returns null for an unknown id so the caller falls back to ambient routing', () => {
    $sessionTiles.set([{ runtimeId: 'rt-a', storedSessionId: 'stored-a' }])

    expect(storedSessionIdForRuntimeId('rt-unknown')).toBeNull()
    expect(storedSessionIdForRuntimeId('')).toBeNull()
  })

  it('ignores tiles with no runtime binding instead of matching undefined ids', () => {
    // A drafted/never-resumed tile has no runtimeId. Looking up an undefined-ish
    // id must not accidentally claim that tile.
    $sessionTiles.set([{ storedSessionId: 'stored-unbound' }, { runtimeId: 'rt-b', storedSessionId: 'stored-b' }])

    expect(storedSessionIdForRuntimeId('rt-b')).toBe('stored-b')
    expect(storedSessionIdForRuntimeId('undefined')).toBeNull()
  })

  it('prefers the stored-id identity when one tile is stored-matched and another is runtime-matched', () => {
    // Pathological but possible after a stale rebind: some other tile's dead
    // runtimeId equals a live tile's storedSessionId. The stored-id claim is
    // authoritative (durable identity wins).
    $sessionTiles.set([
      { runtimeId: 'collision', storedSessionId: 'stored-other' },
      { runtimeId: 'rt-live', storedSessionId: 'collision' }
    ])

    expect(storedSessionIdForRuntimeId('collision')).toBe('collision')
  })
})
