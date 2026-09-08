import { describe, expect, it, vi } from 'vitest'

import { BackendDialClaims } from './backend-dial-claim'
import { runBackendDial } from './backend-dial-routing'
import { parseBackendScopeKey } from './connection-registry'

describe('BackendDialClaims (#90812)', () => {
  it('coalesces two concurrent dials for the same (connectionId, profile) onto ONE backend spawn', async () => {
    const claims = new BackendDialClaims()
    let spawns = 0
    let resolveSpawn: ((value: { baseUrl: string }) => void) | undefined

    const dial = vi.fn(() => {
      spawns += 1

      return new Promise<{ baseUrl: string }>(resolve => {
        resolveSpawn = resolve
      })
    })

    // Two renderer windows race the same reconnect: reconnectGateway()'s
    // in-flight lock is per-renderer, so BOTH invoke the main-process dial.
    const first = claims.run('conn:office-ssh::default', dial)
    const second = claims.run('conn:office-ssh::default', dial)

    expect(spawns).toBe(1)

    resolveSpawn?.({ baseUrl: 'http://127.0.0.1:53150' })

    const [firstResult, secondResult] = await Promise.all([first, second])

    // The second caller receives the FIRST dial's result, not its own spawn.
    expect(firstResult).toBe(secondResult)
    expect(firstResult).toEqual({ baseUrl: 'http://127.0.0.1:53150' })
    expect(dial).toHaveBeenCalledTimes(1)
  })

  it('scopes claims by key: different (connectionId, profile) pairs dial independently', async () => {
    const claims = new BackendDialClaims()
    const dialA = vi.fn(async () => 'a')
    const dialB = vi.fn(async () => 'b')

    const [a, b] = await Promise.all([
      claims.run('conn:office-ssh::default', dialA),
      claims.run('conn:office-ssh::work', dialB)
    ])

    expect(a).toBe('a')
    expect(b).toBe('b')
    expect(dialA).toHaveBeenCalledTimes(1)
    expect(dialB).toHaveBeenCalledTimes(1)
  })

  it('releases the claim once the dial settles so a later reconnect can dial again (bounded, not latched)', async () => {
    const claims = new BackendDialClaims()
    const dial = vi.fn(async () => 'fresh')

    await claims.run('default', dial)
    expect(claims.inFlight('default')).toBe(false)

    await claims.run('default', dial)
    expect(dial).toHaveBeenCalledTimes(2)
  })

  it('propagates a failed dial to every coalesced waiter and never caches the rejection', async () => {
    const claims = new BackendDialClaims()
    let rejectSpawn: ((error: Error) => void) | undefined

    const failingDial = vi.fn(
      () =>
        new Promise<never>((_resolve, reject) => {
          rejectSpawn = reject
        })
    )

    const first = claims.run('conn:office-ssh::default', failingDial)
    const second = claims.run('conn:office-ssh::default', failingDial)
    expect(failingDial).toHaveBeenCalledTimes(1)

    rejectSpawn?.(new Error('ssh dial failed'))

    await expect(first).rejects.toThrow('ssh dial failed')
    await expect(second).rejects.toThrow('ssh dial failed')

    // Fail closed but not latched: the NEXT dial attempt runs fresh.
    const recovered = vi.fn(async () => 'recovered')
    await expect(claims.run('conn:office-ssh::default', recovered)).resolves.toBe('recovered')
    expect(recovered).toHaveBeenCalledTimes(1)
  })

  it('a synchronously-throwing dial rejects the claim instead of escaping the coalescing seam', async () => {
    const claims = new BackendDialClaims()

    await expect(
      claims.run('default', () => {
        throw new Error('spawn refused')
      })
    ).rejects.toThrow('spawn refused')

    expect(claims.inFlight('default')).toBe(false)
  })
})

describe('parseBackendScopeKey (#90812/#93910)', () => {
  it('round-trips the composite pool key back to (connectionId, profile)', () => {
    expect(parseBackendScopeKey('conn:office-ssh::default')).toEqual({
      connectionId: 'office-ssh',
      profile: 'default'
    })
    expect(parseBackendScopeKey('conn:office-ssh::work')).toEqual({ connectionId: 'office-ssh', profile: 'work' })
  })

  it('treats a bare profile key as the local/primary scope', () => {
    expect(parseBackendScopeKey('default')).toEqual({ connectionId: null, profile: 'default' })
    expect(parseBackendScopeKey('work')).toEqual({ connectionId: null, profile: 'work' })
  })
})

describe('backend dial routing (#90812)', () => {
  it('uses the composite registry scope for a registry backend', async () => {
    const claims = new BackendDialClaims()
    const dial = vi.fn(async () => 'registry')
    const scopeKey = vi.fn((connectionId: string | null, profile: string | null | undefined) =>
      `conn:${connectionId ?? 'local'}::${profile ?? 'default'}`)

    await expect(runBackendDial({ claims, scopeKey }, 'office-ssh', 'work', dial)).resolves.toBe('registry')
    expect(scopeKey).toHaveBeenCalledWith('office-ssh', 'work')
    expect(dial).toHaveBeenCalledTimes(1)
  })

  it('uses the local profile scope for a local backend', async () => {
    const claims = new BackendDialClaims()
    const dial = vi.fn(async () => 'local')
    const scopeKey = vi.fn((connectionId: string | null, profile: string | null | undefined) =>
      `conn:${connectionId ?? 'local'}::${profile ?? 'default'}`)

    await expect(runBackendDial({ claims, scopeKey }, null, 'default', dial)).resolves.toBe('local')
    expect(scopeKey).toHaveBeenCalledWith(null, 'default')
    expect(dial).toHaveBeenCalledTimes(1)
  })
})
