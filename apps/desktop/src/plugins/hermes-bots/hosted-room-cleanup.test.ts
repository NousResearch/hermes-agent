import { beforeEach, describe, expect, it, vi } from 'vitest'

import { pluginSdkMock, scriptedStorage } from './group-test-utils'

const mocks = vi.hoisted(() => ({
  host: {} as Record<string, unknown>,
  requestProfile: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async () => pluginSdkMock(mocks.host))

const reconnectOperation = () => ({
  operationId: 'reconnect:room-1:builder:grant',
  setupId: 'reconnect:room-1:builder',
  kind: 'peer-reconnect' as const,
  connectionId: 'peer',
  profile: 'builder',
  grant: 'private-grant',
  grantSha256: 'a'.repeat(64),
  expectedGrantSha256: 'c'.repeat(64),
  roomId: 'room-1',
  cancelId: null,
  homeConnectionId: 'home',
  homeProfile: 'default',
  memberId: 'builder',
  targetUrl: 'https://peer.example.test:19445/p/builder',
  catalog: {
    catalog_digest: 'digest:peer',
    installation_id: 'install:peer'
  }
})

async function loadCleanup() {
  vi.resetModules()

  return import('./hosted-room-cleanup')
}

function expireCleanupOwners(durable: Map<string, unknown>) {
  const cleanup = durable.get('hosted-room-cleanup-v1') as {
    operations?: Array<Record<string, unknown>>
  }

  durable.set('hosted-room-cleanup-v1', {
    version: 1,
    operations: (cleanup?.operations || []).map(operation => ({
      ...operation,
      ownerLeaseUntil: 0
    }))
  })
}

function testLockManager() {
  const held = new Set<string>()
  const tails = new Map<string, Promise<void>>()

  return {
    request<T>(name: string, options: { ifAvailable?: boolean }, callback: (lock: null | object) => Promise<T> | T) {
      if (options.ifAvailable) {
        if (held.has(name)) {
          return Promise.resolve(callback(null))
        }

        held.add(name)
        return Promise.resolve(callback({})).finally(() => held.delete(name))
      }

      const previous = tails.get(name) || Promise.resolve()
      const result = previous.then(async () => {
        held.add(name)
        try {
          return await callback({})
        } finally {
          held.delete(name)
        }
      })

      tails.set(
        name,
        result.then(
          () => undefined,
          () => undefined
        )
      )
      return result
    }
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  Object.assign(mocks.host, {
    profileRoutes: async () => [
      { connectionId: 'home', mode: 'remote', profile: 'default', targetProfile: 'default' },
      { connectionId: 'peer', mode: 'remote', profile: 'builder', targetProfile: 'builder' }
    ],
    requestProfile: mocks.requestProfile
  })
})

describe('hosted Group Chat cleanup journal', () => {
  it('preserves overlapping writes from separate Desktop windows', async () => {
    const durable = new Map<string, unknown>()
    const storage = scriptedStorage(durable).storage
    const originalLocks = Object.getOwnPropertyDescriptor(globalThis.navigator, 'locks')

    Object.defineProperty(globalThis.navigator, 'locks', {
      configurable: true,
      value: testLockManager()
    })

    try {
      const first = await loadCleanup()

      await first.startHostedRoomCleanup(storage)

      const second = await loadCleanup()

      await second.startHostedRoomCleanup(storage)

      await Promise.all([
        first.addHostedRoomCleanup(reconnectOperation()),
        second.addHostedRoomCleanup({
          ...reconnectOperation(),
          operationId: 'reconnect:room-2:builder:grant',
          setupId: 'reconnect:room-2:builder',
          roomId: 'room-2'
        })
      ])

      expect((durable.get('hosted-room-cleanup-v1') as { operations: Array<{ roomId: string }> }).operations).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ roomId: 'room-1' }),
          expect.objectContaining({ roomId: 'room-2' })
        ])
      )
      await second.dispatchHostedRoomCleanup()
      expect(mocks.requestProfile).not.toHaveBeenCalled()

      const persisted = durable.get('hosted-room-cleanup-v1') as {
        operations: Array<{ ownerLeaseUntil: number; roomId: string }>
      }
      durable.set('hosted-room-cleanup-v1', {
        version: 1,
        operations: persisted.operations.map(operation =>
          operation.roomId === 'room-1' ? { ...operation, ownerLeaseUntil: 0 } : operation
        )
      })
      mocks.requestProfile.mockResolvedValue({ registered: true })
      await second.dispatchHostedRoomCleanup()
      expect(mocks.requestProfile).not.toHaveBeenCalled()

      first.stopHostedRoomCleanup()
      await second.dispatchHostedRoomCleanup()

      expect(mocks.requestProfile).toHaveBeenCalledWith(
        expect.objectContaining({ connectionId: 'home' }),
        'groups.peer.register',
        expect.objectContaining({ room_id: 'room-1' })
      )
      expect((durable.get('hosted-room-cleanup-v1') as { operations: Array<{ roomId: string }> }).operations).toEqual([
        expect.objectContaining({ roomId: 'room-2' })
      ])
      second.stopHostedRoomCleanup()
    } finally {
      if (originalLocks) {
        Object.defineProperty(globalThis.navigator, 'locks', originalLocks)
      } else {
        Reflect.deleteProperty(globalThis.navigator, 'locks')
      }
    }
  })

  it('rejects a cleanup write that production storage cannot read back', async () => {
    const cleanup = await loadCleanup()
    const storage = {
      get: vi.fn(async () => null),
      set: vi.fn(() => undefined)
    }

    await cleanup.startHostedRoomCleanup(storage as never)
    await expect(cleanup.addHostedRoomCleanup(reconnectOperation())).rejects.toThrow(
      'did not persist Group Chat cleanup'
    )
    expect(cleanup.$hostedRoomCleanup.get().operations).toEqual([])
  })

  it('journals an invitation response that arrives after runtime stop', async () => {
    const durable = new Map<string, unknown>()
    const storage = scriptedStorage(durable).storage
    const cleanup = await loadCleanup()

    await cleanup.startHostedRoomCleanup(storage)
    cleanup.stopHostedRoomCleanup()
    await cleanup.addHostedRoomCleanup(reconnectOperation())
    await cleanup.armHostedRoomCleanup(reconnectOperation().setupId)
    await cleanup.dispatchHostedRoomCleanup()

    expect(mocks.requestProfile).not.toHaveBeenCalled()
    expect((durable.get('hosted-room-cleanup-v1') as { operations: Array<{ armed: boolean }> }).operations).toEqual([
      expect.objectContaining({ armed: true })
    ])
  })

  it('replays a reconnect registration after the Desktop dies post-invite', async () => {
    const durable = new Map<string, unknown>()
    const storage = scriptedStorage(durable).storage
    const first = await loadCleanup()

    await first.startHostedRoomCleanup(storage)
    await first.addHostedRoomCleanup(reconnectOperation())
    first.stopHostedRoomCleanup()
    expireCleanupOwners(durable)

    mocks.requestProfile.mockImplementation(async (_route, method) => {
      if (method === 'groups.state') {
        return {
          driver_status: {
            peer_routes: [{ member_id: 'builder', status: 'needs_reauthorization', grant_sha256: 'c'.repeat(64) }]
          }
        }
      }

      if (method === 'groups.peer.register') {
        return { registered: true }
      }

      throw new Error(`unexpected method: ${method}`)
    })

    const restarted = await loadCleanup()

    await restarted.startHostedRoomCleanup(storage)

    expect(mocks.requestProfile).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'home' }),
      'groups.peer.register',
      expect.objectContaining({
        expected_grant_sha256: 'c'.repeat(64),
        grant: 'private-grant',
        member_id: 'builder',
        room_id: 'room-1'
      })
    )
    expect((durable.get('hosted-room-cleanup-v1') as { operations: unknown[] }).operations).toEqual([])
  })

  it('replays owner-tagged cleanup after a same-process stop and start', async () => {
    const durable = new Map<string, unknown>()
    const storage = scriptedStorage(durable).storage
    const cleanup = await loadCleanup()

    await cleanup.startHostedRoomCleanup(storage)
    await cleanup.addHostedRoomCleanup(reconnectOperation())
    cleanup.stopHostedRoomCleanup()
    mocks.requestProfile.mockImplementation(async (_route, method) => {
      if (method === 'groups.peer.register') {
        return { registered: true }
      }

      throw new Error(`unexpected method: ${method}`)
    })

    await cleanup.startHostedRoomCleanup(storage)

    expect(mocks.requestProfile).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'home' }),
      'groups.peer.register',
      expect.anything()
    )
    expect((durable.get('hosted-room-cleanup-v1') as { operations: unknown[] }).operations).toEqual([])
  })

  it('keeps a matching route pending until registration is positively revalidated', async () => {
    const durable = new Map<string, unknown>()
    const storage = scriptedStorage(durable).storage
    const first = await loadCleanup()

    await first.startHostedRoomCleanup(storage)
    await first.addHostedRoomCleanup(reconnectOperation())
    first.stopHostedRoomCleanup()
    expireCleanupOwners(durable)

    mocks.requestProfile.mockImplementation(async (_route, method) => {
      if (method === 'groups.peer.register') {
        throw new Error('response lost')
      }

      if (method === 'groups.state') {
        return {
          driver_status: {
            peer_routes: [{ member_id: 'builder', status: 'ready', grant_sha256: 'a'.repeat(64) }]
          }
        }
      }

      if (method === 'groups.peer.revoke_exact') {
        throw new Error('must not revoke a committed grant')
      }

      throw new Error(`unexpected method: ${method}`)
    })

    const restarted = await loadCleanup()

    await restarted.startHostedRoomCleanup(storage)

    expect(mocks.requestProfile).not.toHaveBeenCalledWith(
      expect.anything(),
      'groups.peer.revoke_exact',
      expect.anything()
    )
    expect((durable.get('hosted-room-cleanup-v1') as { operations: unknown[] }).operations).toHaveLength(1)

    mocks.requestProfile.mockImplementation(async (_route, method) => {
      if (method === 'groups.state') {
        return {
          driver_status: {
            peer_routes: [{ member_id: 'builder', status: 'ready', grant_sha256: 'a'.repeat(64) }]
          }
        }
      }

      if (method === 'groups.peer.register') {
        return { registered: true }
      }

      throw new Error(`unexpected method: ${method}`)
    })
    await restarted.dispatchHostedRoomCleanup()
    expect((durable.get('hosted-room-cleanup-v1') as { operations: unknown[] }).operations).toEqual([])
  })

  it('keeps a failed revocation durable and retries it later', async () => {
    const durable = new Map<string, unknown>()
    const storage = scriptedStorage(durable).storage
    const first = await loadCleanup()

    await first.startHostedRoomCleanup(storage)
    await first.addHostedRoomCleanup({
      operationId: 'revoke:room-1:builder',
      setupId: 'revoke:room-1:builder',
      kind: 'peer-revoke',
      connectionId: 'peer',
      profile: 'builder',
      grant: 'private-grant',
      roomId: null,
      cancelId: null,
      homeConnectionId: null,
      homeProfile: null,
      memberId: null,
      targetUrl: null,
      catalog: null
    })
    first.stopHostedRoomCleanup()
    expireCleanupOwners(durable)

    mocks.requestProfile.mockRejectedValueOnce(new Error('peer offline'))
    const restarted = await loadCleanup()

    await restarted.startHostedRoomCleanup(storage)
    expect((durable.get('hosted-room-cleanup-v1') as { operations: unknown[] }).operations).toHaveLength(1)

    mocks.requestProfile.mockResolvedValueOnce({ revoked: true })
    await restarted.dispatchHostedRoomCleanup()
    expect((durable.get('hosted-room-cleanup-v1') as { operations: unknown[] }).operations).toEqual([])
  })

  it('keeps reconnect cleanup pending when home state omits driver status', async () => {
    const durable = new Map<string, unknown>()
    const storage = scriptedStorage(durable).storage
    const first = await loadCleanup()

    await first.startHostedRoomCleanup(storage)
    await first.addHostedRoomCleanup(reconnectOperation())
    first.stopHostedRoomCleanup()
    expireCleanupOwners(durable)

    mocks.requestProfile.mockImplementation(async (_route, method) => {
      if (method === 'groups.peer.register') {
        throw new Error('home unreachable')
      }

      if (method === 'groups.state') {
        return { room: { room_id: 'room-1' } }
      }

      if (method === 'groups.peer.revoke_exact') {
        throw new Error('must not revoke an ambiguous grant')
      }

      throw new Error(`unexpected method: ${method}`)
    })
    const restarted = await loadCleanup()

    await restarted.startHostedRoomCleanup(storage)

    expect(mocks.requestProfile).not.toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'peer' }),
      'groups.peer.revoke_exact',
      expect.anything()
    )
    expect((durable.get('hosted-room-cleanup-v1') as { operations: unknown[] }).operations).toHaveLength(1)

    mocks.requestProfile.mockImplementation(async (_route, method) => {
      if (method === 'groups.peer.register') {
        return { registered: true }
      }

      throw new Error(`unexpected method: ${method}`)
    })
    await restarted.dispatchHostedRoomCleanup()
    expect((durable.get('hosted-room-cleanup-v1') as { operations: unknown[] }).operations).toEqual([])
  })

  it('keeps a committed but unavailable route pending after a lost reply', async () => {
    const durable = new Map<string, unknown>()
    const storage = scriptedStorage(durable).storage
    const first = await loadCleanup()

    await first.startHostedRoomCleanup(storage)
    await first.addHostedRoomCleanup(reconnectOperation())
    first.stopHostedRoomCleanup()
    expireCleanupOwners(durable)

    mocks.requestProfile.mockImplementation(async (_route, method) => {
      if (method === 'groups.peer.register') {
        throw new Error('response lost')
      }

      if (method === 'groups.state') {
        return {
          driver_status: {
            peer_routes: [{ member_id: 'builder', status: 'unavailable', grant_sha256: 'a'.repeat(64) }]
          }
        }
      }

      if (method === 'groups.peer.revoke_exact') {
        throw new Error('must not revoke a transiently unavailable route')
      }

      throw new Error(`unexpected method: ${method}`)
    })
    const restarted = await loadCleanup()

    await restarted.startHostedRoomCleanup(storage)

    expect(mocks.requestProfile).not.toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'peer' }),
      'groups.peer.revoke_exact',
      expect.anything()
    )
    expect((durable.get('hosted-room-cleanup-v1') as { operations: unknown[] }).operations).toHaveLength(1)
  })

  it('exact-revokes only the losing grant when another Desktop wins', async () => {
    const durable = new Map<string, unknown>()
    const storage = scriptedStorage(durable).storage
    const first = await loadCleanup()

    await first.startHostedRoomCleanup(storage)
    await first.addHostedRoomCleanup(reconnectOperation())
    first.stopHostedRoomCleanup()
    expireCleanupOwners(durable)

    mocks.requestProfile.mockImplementation(async (_route, method) => {
      if (method === 'groups.peer.register') {
        throw new Error('response lost')
      }

      if (method === 'groups.state') {
        return {
          driver_status: {
            peer_routes: [{ member_id: 'builder', status: 'ready', grant_sha256: 'b'.repeat(64) }]
          }
        }
      }

      if (method === 'groups.peer.revoke_exact') {
        return { revoked: true }
      }

      throw new Error(`unexpected method: ${method}`)
    })
    const restarted = await loadCleanup()

    await restarted.startHostedRoomCleanup(storage)

    expect(mocks.requestProfile).not.toHaveBeenCalledWith(expect.anything(), 'groups.peer.register', expect.anything())
    expect(mocks.requestProfile).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'peer' }),
      'groups.peer.revoke_exact',
      { grant: 'private-grant', profile: 'builder' }
    )
    expect((durable.get('hosted-room-cleanup-v1') as { operations: unknown[] }).operations).toEqual([])
  })

  it('retries a failed revoke after a definitive reconnect rejection', async () => {
    const durable = new Map<string, unknown>()
    const storage = scriptedStorage(durable).storage
    const first = await loadCleanup()

    await first.startHostedRoomCleanup(storage)
    await first.addHostedRoomCleanup(reconnectOperation())
    first.stopHostedRoomCleanup()
    expireCleanupOwners(durable)

    let revokeFails = true
    mocks.requestProfile.mockImplementation(async (_route, method) => {
      if (method === 'groups.peer.register') {
        throw new Error('grant rejected')
      }

      if (method === 'groups.state') {
        return {
          driver_status: {
            peer_routes: [{ member_id: 'builder', status: 'needs_reauthorization' }]
          }
        }
      }

      if (method === 'groups.peer.revoke_exact') {
        if (revokeFails) {
          throw new Error('peer offline')
        }

        return { revoked: true }
      }

      throw new Error(`unexpected method: ${method}`)
    })
    const restarted = await loadCleanup()

    await restarted.startHostedRoomCleanup(storage)
    expect((durable.get('hosted-room-cleanup-v1') as { operations: unknown[] }).operations).toHaveLength(1)

    revokeFails = false
    await restarted.dispatchHostedRoomCleanup()
    expect((durable.get('hosted-room-cleanup-v1') as { operations: unknown[] }).operations).toEqual([])
  })
})
