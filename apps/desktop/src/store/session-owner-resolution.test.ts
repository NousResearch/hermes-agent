import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connectionsRegistry } from './connections'
import { $profiles } from './profile'
import {
  ambientGatewayOwnsEverySession,
  assertSessionOwnerResolved,
  sessionOwnerIsKnown
} from './session-owner-resolution'

const registry = (...ids: string[]) =>
  ({
    connections: ids.map(id => ({ id })),
    lastUsed: ids[0] ?? null,
    launchMode: 'primary',
    primary: ids[0] ?? null
  }) as never

beforeEach(() => {
  $connectionsRegistry.set(null)
  $profiles.set([])
})

afterEach(() => {
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('session owner topology', () => {
  it('fails closed while the modern registry bridge is present but its async cache is not loaded', () => {
    ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = {
      connections: { list: vi.fn(async () => Promise.reject(new Error('ipc unavailable'))) }
    }
    $connectionsRegistry.set(null)
    $profiles.set([{ name: 'default' }] as never)

    expect(sessionOwnerIsKnown('default')).toBe(true)
    expect(ambientGatewayOwnsEverySession()).toBe(false)
    expect(() =>
      assertSessionOwnerResolved('default', { method: 'session.resume', sessionId: 'registry-loading' })
    ).not.toThrow()
    expect(() => assertSessionOwnerResolved(null, { method: 'session.resume', sessionId: 'registry-loading' })).toThrow(
      /could not be resolved/i
    )
  })

  it('fails closed on an unknown owner in registry topology while preserving legacy profile routes', () => {
    // A connection registry means the ambient gateway is never provably the
    // sole backend, even with one profile listed: an unknown owner fails
    // closed. A bare profile still names a backend — the legacy profile door
    // (a pick on the primary / explicit `local` source) mints sessions owned
    // by that profile's pool socket in every topology.
    $connectionsRegistry.set(registry('local'))
    $profiles.set([{ name: 'default' }] as never)

    expect(sessionOwnerIsKnown('default')).toBe(true)
    expect(ambientGatewayOwnsEverySession()).toBe(false)
    expect(() =>
      assertSessionOwnerResolved('default', { method: 'session.resume', sessionId: 'registry-profile' })
    ).not.toThrow()
    expect(() => assertSessionOwnerResolved(null, { method: 'session.resume', sessionId: 'unknown-owner' })).toThrow(
      /could not be resolved/i
    )

    $connectionsRegistry.set(registry('local', 'homelab'))
    expect(sessionOwnerIsKnown(null)).toBe(false)
    expect(ambientGatewayOwnsEverySession()).toBe(false)
    expect(() => assertSessionOwnerResolved(null, { method: 'session.resume', sessionId: 'unknown-owner' })).toThrow(
      /could not be resolved/i
    )

    $connectionsRegistry.set(null)
    expect(sessionOwnerIsKnown('default')).toBe(true)
    expect(ambientGatewayOwnsEverySession()).toBe(true)
    expect(() =>
      assertSessionOwnerResolved(null, { method: 'session.resume', sessionId: 'legacy-single-profile' })
    ).not.toThrow()

    $profiles.set([{ name: 'default' }, { name: 'loki' }] as never)
    expect(sessionOwnerIsKnown('loki')).toBe(true)
    expect(ambientGatewayOwnsEverySession()).toBe(false)
    expect(() =>
      assertSessionOwnerResolved('loki', { method: 'session.resume', sessionId: 'legacy-profile-owner' })
    ).not.toThrow()
  })
})
