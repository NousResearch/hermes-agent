import { afterEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $profiles } from '@/store/profile'
import { _resetSessionOwnerHintsForTests, setSessionOwnerHint, setSessions } from '@/store/session'
import { isSessionOwnerResolutionError } from '@/store/session-owner-resolution'
import {
  $sessionTiles,
  clearAllSessionStates,
  dropSessionState,
  knownOwnerForSession,
  publishSessionState,
  recordSessionEventScope,
  requestForOwnedSession,
  storedSessionIdForRuntimeId
} from '@/store/session-states'
import { makeSessionInfo } from '@/test/session-info'

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

  it('maps a MAIN-PANE runtime id through the per-runtime state mirror (no tile involved)', () => {
    // approval.respond from a native notification, a queued send: the caller
    // holds the runtime id of the primary thread, which no tile knows. The
    // state mirror carries the stored id the wiring cache bound.
    publishSessionState('rt-main', createClientSessionState('stored-main'))

    expect(storedSessionIdForRuntimeId('rt-main')).toBe('stored-main')
    // A detached runtime (null stored id) is still unknown.
    publishSessionState('rt-detached', createClientSessionState(null))
    expect(storedSessionIdForRuntimeId('rt-detached')).toBeNull()

    clearAllSessionStates()
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

describe('knownOwnerForSession / requestForOwnedSession', () => {
  afterEach(() => {
    $sessionTiles.set([])
    clearAllSessionStates()
    setSessions([])
    $profiles.set([])
    _resetSessionOwnerHintsForTests({ storage: true })
  })

  it('resolves a main-pane runtime id to its EXACT owner via the mirror + the hint, then the tagged row', () => {
    publishSessionState('rt-main', createClientSessionState('stored-main'))
    setSessionOwnerHint('stored-main', { connectionId: 'local', profile: 'omar' })

    expect(knownOwnerForSession('rt-main')).toEqual({ connectionId: 'local', profile: 'omar' })

    _resetSessionOwnerHintsForTests()
    setSessions([makeSessionInfo({ connection_id: 'local', id: 'stored-main', profile: 'omar' })])
    expect(knownOwnerForSession('rt-main')).toEqual({ connectionId: 'local', profile: 'omar' })

    setSessions([makeSessionInfo({ id: 'stored-main', profile: 'coder' })])
    expect(knownOwnerForSession('rt-main')).toBe('coder')
  })

  it('fails closed with an explicit owner-resolution error instead of the ambient socket', async () => {
    // Somewhere to misroute to: two profiles exist.
    $profiles.set([{ name: 'default' }, { name: 'omar' }] as never)
    const ambient = vi.fn(async () => ({ ok: true }))

    await expect(
      requestForOwnedSession('rt-orphan', ambient as never, 'approval.respond', { session_id: 'rt-orphan' })
    ).rejects.toSatisfy(isSessionOwnerResolutionError)
    expect(ambient).not.toHaveBeenCalled()

    // Legacy single backend: the ambient gateway IS the owner.
    $profiles.set([{ name: 'default' }] as never)
    await expect(
      requestForOwnedSession('rt-orphan', ambient as never, 'approval.respond', { session_id: 'rt-orphan' })
    ).resolves.toEqual({ ok: true })
    expect(ambient).toHaveBeenCalledWith('approval.respond', { session_id: 'rt-orphan' })
  })

  it('routes a connection-tagged orphan runtime through the owner its inbound event recorded (#97511)', () => {
    // Registry topology, multiple profiles, no tile/hint/row binding for the
    // runtime — the approval.request event itself proved the exact owner.
    $profiles.set([{ name: 'default' }, { name: 'omar' }] as never)
    recordSessionEventScope({ connectionId: 'homelab', profile: 'omar', session_id: 'rt-unbound' })

    expect(knownOwnerForSession('rt-unbound')).toEqual({ connectionId: 'homelab', profile: 'omar' })

    // An event without a profile tag still records the 'default' convention
    // every other owner source uses.
    recordSessionEventScope({ connectionId: 'homelab', session_id: 'rt-unprofiled' })
    expect(knownOwnerForSession('rt-unprofiled')).toEqual({ connectionId: 'homelab', profile: 'default' })
  })

  it('still prefers the durable stored owner when a stale runtime ledger entry collides with a stored id (#97511)', () => {
    // Pathological collision: some dead runtime's id equals a live stored id.
    // The persisted hint (durable identity) must outrank the ledger entry.
    setSessionOwnerHint('stored-live', { connectionId: 'local', profile: 'omar' })
    recordSessionEventScope({ connectionId: 'spark', profile: 'default', session_id: 'stored-live' })

    expect(knownOwnerForSession('stored-live')).toEqual({ connectionId: 'local', profile: 'omar' })
  })

  it('keeps failing closed for untagged or unknown runtimes in multi-profile topology (#97511)', () => {
    $profiles.set([{ name: 'default' }, { name: 'omar' }] as never)
    // Untagged events carry no connectionId and record nothing.
    recordSessionEventScope({ profile: 'omar', session_id: 'rt-untagged' })

    expect(knownOwnerForSession('rt-untagged')).toBeUndefined()
    expect(knownOwnerForSession('rt-never-seen')).toBeUndefined()
  })

  it('drops the recorded event owner together with the runtime state (#97511)', () => {
    recordSessionEventScope({ connectionId: 'homelab', profile: 'omar', session_id: 'rt-dropped' })
    expect(knownOwnerForSession('rt-dropped')).toEqual({ connectionId: 'homelab', profile: 'omar' })

    dropSessionState('rt-dropped')
    expect(knownOwnerForSession('rt-dropped')).toBeUndefined()
  })
})
