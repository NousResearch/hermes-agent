import { describe, expect, it, vi } from 'vitest'

import {
  connectionApplyAffectsPoolProfile,
  createProfileAsyncQueue,
  ensureCompatiblePoolEntry,
  gatewayRequestForcesLocal,
  selectBackendSelection
} from './profile-request-routing'

describe('profile request routing', () => {
  it('makes gatewayId=local authoritative over ambient remote routing', () => {
    expect(gatewayRequestForcesLocal('local')).toBe(true)
    expect(
      selectBackendSelection({
        explicitRemote: true,
        forceLocal: gatewayRequestForcesLocal('local'),
        globalRemote: true,
        primaryProfile: 'default',
        profile: 'default'
      })
    ).toEqual({ route: 'local', target: 'pool' })
  })

  it('fails closed for unsupported gateway ids', () => {
    expect(() => gatewayRequestForcesLocal('invented-peer')).toThrow('Unknown gateway target: invented-peer')
  })

  it('keeps omitted targets on current ambient primary routing', () => {
    expect(gatewayRequestForcesLocal(undefined)).toBe(false)
    expect(
      selectBackendSelection({ globalRemote: true, primaryProfile: 'default', profile: 'default' })
    ).toEqual({ route: 'remote-global', target: 'primary' })
    expect(
      selectBackendSelection({ globalRemote: true, primaryProfile: 'default', profile: 'work' })
    ).toEqual({ route: 'remote-global', target: 'primary' })
  })

  it('keeps an explicit Local primary on primary and isolates non-primary overrides', () => {
    expect(
      selectBackendSelection({
        explicitLocal: true,
        globalRemote: true,
        primaryProfile: 'work',
        profile: 'work'
      })
    ).toEqual({ route: 'local', target: 'primary' })
    expect(
      selectBackendSelection({
        explicitLocal: true,
        globalRemote: true,
        primaryProfile: 'default',
        profile: 'work'
      })
    ).toEqual({ route: 'local', target: 'pool' })
    expect(
      selectBackendSelection({
        explicitRemote: true,
        globalRemote: true,
        primaryProfile: 'default',
        profile: 'work'
      })
    ).toEqual({ route: 'remote-profile', target: 'pool' })
  })

  it('reuses compatible routes and replaces incompatible pool entries', async () => {
    type Entry = { id: string; routeIdentity: 'local' | 'remote-global' }
    let current: Entry | undefined = { id: 'stale', routeIdentity: 'remote-global' }
    const teardown = vi.fn(async (entry: Entry) => {
      if (current === entry) {
        current = undefined
      }
    })
    const create = vi.fn(() => (current = { id: 'replacement', routeIdentity: 'local' }))

    const replacement = await ensureCompatiblePoolEntry({ create, get: () => current, route: 'local', teardown })
    const reused = await ensureCompatiblePoolEntry({ create, get: () => current, route: 'local', teardown })

    expect(replacement).toBe(reused)
    expect(teardown).toHaveBeenCalledOnce()
    expect(create).toHaveBeenCalledOnce()
  })

  it('reuses a compatible replacement installed while stale teardown awaits', async () => {
    type Entry = { id: string; routeIdentity: 'local' | 'remote-global' }
    let current: Entry | undefined = { id: 'stale', routeIdentity: 'remote-global' }
    let releaseTeardown!: () => void
    const teardownGate = new Promise<void>(resolve => {
      releaseTeardown = resolve
    })
    let teardownStarted!: () => void
    const started = new Promise<void>(resolve => {
      teardownStarted = resolve
    })
    const create = vi.fn(() => (current = { id: 'replacement', routeIdentity: 'local' }))
    const teardown = vi.fn(async (entry: Entry) => {
      if (current === entry) {
        current = undefined
      }

      teardownStarted()
      await teardownGate
    })

    const first = ensureCompatiblePoolEntry({ create, get: () => current, route: 'local', teardown })
    await started
    const second = await ensureCompatiblePoolEntry({ create, get: () => current, route: 'local', teardown })
    releaseTeardown()

    expect(await first).toBe(second)
    expect(create).toHaveBeenCalledOnce()
    expect(teardown).toHaveBeenCalledOnce()
  })

  it('serializes Apply teardown before a same-profile replacement is created', async () => {
    const queue = createProfileAsyncQueue()
    const events: string[] = []
    let release!: () => void
    const waiting = new Promise<void>(resolve => {
      release = resolve
    })
    const apply = queue.run('work', async () => {
      events.push('apply:start')
      await waiting
      events.push('apply:end')
    })
    const request = queue.run('work', () => {
      events.push('request:create')
    })

    await Promise.resolve()
    expect(events).toEqual(['apply:start'])
    release()
    await Promise.all([apply, request])
    expect(events).toEqual(['apply:start', 'apply:end', 'request:create'])
  })

  it('invalidates only pool routes affected by Apply', () => {
    const affected = (profile: string, appliedProfile: null | string, explicit = false) =>
      connectionApplyAffectsPoolProfile({
        appliedProfile,
        hasExplicitProfileRoute: explicit,
        primaryProfile: 'default',
        profile
      })

    expect(affected('work', 'work', true)).toBe(true)
    expect(affected('review', 'work', true)).toBe(false)
    expect(affected('work', null, false)).toBe(true)
    expect(affected('work', null, true)).toBe(false)
  })
})
