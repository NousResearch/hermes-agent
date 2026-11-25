import { describe, expect, it, vi } from 'vitest'

import { BackendDialClaims, DIAL_ROUTE_REPLACED } from './backend-dial-claim'
import { backendScopeKey, parseBackendScopeKey } from './connection-registry'
import { resolveDesktopConnectionRequest } from './desktop-profile'

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

describe('BackendDialClaims.forget', () => {
  it('the next caller dials again, and the callers of the forgotten dial are told to ask again', async () => {
    const claims = new BackendDialClaims()
    const old = claims.run('work', () => new Promise<string>(() => undefined))
    const joined = claims.run('work', async () => 'never dialed')

    claims.forget('work')
    expect(claims.inFlight('work')).toBe(false)

    // No three-minute wait on an address nothing serves any more.
    await expect(old).rejects.toThrow(DIAL_ROUTE_REPLACED)
    await expect(joined).rejects.toThrow(DIAL_ROUTE_REPLACED)

    const fresh = vi.fn(async () => 'new route')

    await expect(claims.run('work', fresh)).resolves.toBe('new route')
    expect(fresh).toHaveBeenCalledTimes(1)
  })

  it('a forgotten dial that settles later does not release the dial that replaced it', async () => {
    const claims = new BackendDialClaims()
    let failOld: (error: Error) => void = () => undefined

    const old = claims.run(
      'work',
      () =>
        new Promise<string>((_resolve, reject) => {
          failOld = reject
        })
    )

    claims.forget('work')

    // Dialed in the same tick, before the forgotten claim's own release has run.
    const replacement = claims.run('work', () => new Promise<string>(() => undefined))

    await expect(old).rejects.toThrow(DIAL_ROUTE_REPLACED)
    // The old wait runs out long after: nobody is told, and the newer dial keeps its claim.
    failOld(new Error('old route gone'))
    await Promise.resolve()

    const third = vi.fn(async () => 'third')

    expect(claims.inFlight('work')).toBe(true)
    expect(claims.run('work', third)).toBe(replacement)
    expect(third).not.toHaveBeenCalled()
  })

  it('a dial nobody forgot answers as before, and forgetting a key with no dial in flight does nothing', async () => {
    const claims = new BackendDialClaims()

    expect(() => claims.forget('work')).not.toThrow()
    expect(claims.inFlight('work')).toBe(false)

    const answered = claims.run('work', async () => 'this computer')

    await expect(answered).resolves.toBe('this computer')
    expect(claims.inFlight('work')).toBe(false)
    expect(() => claims.forget('work')).not.toThrow()
  })

  it('a dial nobody forgot gives up its claim the moment it settles, so the next caller dials again', async () => {
    const claims = new BackendDialClaims()
    let settle: (value: string) => void = () => undefined

    const first = claims.run(
      'work',
      () =>
        new Promise<string>(resolve => {
          settle = resolve
        })
    )

    settle('first')
    // One microtask after the dial settled: its claim lasts only while the dial itself is unsettled.
    await Promise.resolve()
    expect(claims.inFlight('work')).toBe(false)

    const again = vi.fn(async () => 'second')

    await expect(claims.run('work', again)).resolves.toBe('second')
    expect(again).toHaveBeenCalledTimes(1)
    await expect(first).resolves.toBe('first')
  })

  it('a forgotten or failed dial whose caller walked away is not an unhandled rejection', async () => {
    const unhandled = vi.fn()

    process.on('unhandledRejection', unhandled)

    try {
      const claims = new BackendDialClaims()

      void claims.run('work', () => new Promise<string>(() => undefined))
      claims.forget('work')
      void claims.run('default', () => Promise.reject(new Error('dial failed')))
      // Node reports an unhandled rejection once the microtasks have drained.
      await new Promise(resolve => setTimeout(resolve, 0))
      await new Promise(resolve => setImmediate(resolve))

      expect(unhandled).not.toHaveBeenCalled()
    } finally {
      process.off('unhandledRejection', unhandled)
    }
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

describe('resolved window routes share one dial claim (#90812)', () => {
  it.each([null, 'office-ssh'])(
    'coalesces resolved window routes without absorbing a same-named source (%s)',
    async connectionId => {
      const claims = new BackendDialClaims()
      const source = { connectionId, profile: 'work', registryScoped: connectionId !== null }
      const route = resolveDesktopConnectionRequest(undefined, source, 'default')
      const key = backendScopeKey(route.connectionId, route.profile)
      const dial = vi.fn(async () => ({ baseUrl: 'http://localhost:53150' }))
      const other = vi.fn(async () => ({ baseUrl: 'http://localhost:53151' }))

      const [first, second, separate] = await Promise.all([
        claims.run(key, dial),
        claims.run(key, dial),
        claims.run(backendScopeKey('another-source', route.profile), other)
      ])

      expect(first).toBe(second)
      expect(first).not.toBe(separate)
      expect(dial).toHaveBeenCalledTimes(1)
      expect(other).toHaveBeenCalledTimes(1)
    }
  )
})
