import { describe, expect, it, vi } from 'vitest'

import { recycleOwnedBackend, recycleOwnedBackendTarget, recyclePinnedBackend } from './backend-recycle'
import { backendScopeKey, type ConnectionRegistry, type ResolvedConnectionDescriptor } from './connection-registry'
import { createPoolStopper } from './pool-stop'

const registry: ConnectionRegistry = {
  version: 2,
  primary: 'ssh-a',
  lastUsed: 'ssh-a',
  launchMode: 'primary',
  connections: [
    { id: 'local', label: 'Device', kind: 'local' },
    { id: 'ssh-a', label: 'SSH A', kind: 'ssh', host: 'a.invalid' },
    { id: 'ssh-b', label: 'SSH B', kind: 'ssh', host: 'b.invalid' },
    { id: 'url', label: 'URL', kind: 'remote', url: 'https://remote.invalid' }
  ]
}

it('pinned recovery follows the real local/registry pool keys and leaves same-name owners running', async () => {
  for (const [connectionId, globalRemote, key] of [
    ['local', false, 'worker'],
    ['local', true, 'conn:local::worker'],
    ['ssh-a', false, backendScopeKey('ssh-a', 'worker')]
  ] as const) {
    const descriptor: ResolvedConnectionDescriptor =
      connectionId === 'local' ? { mode: 'local' } : { mode: 'remote', remoteKind: 'ssh', connectionId }

    const process = { key }

    const pool = new Map([
      [key, { process, connectionPromise: Promise.resolve(descriptor) }],
      [
        backendScopeKey('ssh-b', 'worker'),
        { process: {}, connectionPromise: Promise.resolve({ mode: 'remote' as const }) }
      ]
    ])

    const events: string[] = []

    const stopper = createPoolStopper({
      pool,
      stopChild: child => {
        expect(child).toBe(process)
        events.push('child')
      },
      waitForExit: async () => {
        events.push('exit')
      }
    })

    const ssh = { remotePlatform: 'Linux' }
    const primary = vi.fn()
    await recyclePinnedBackend(
      { connectionId, profile: 'worker' },
      {
        registry,
        routeOptions: { globalRemote, primaryProfile: 'default' },
        effectiveSshFingerprint: async () => 'fixture',
        primarySshKey: '',
        primaryPromise: () => Promise.resolve({ mode: 'local' }),
        pool,
        sshState: () => (connectionId === 'local' ? undefined : ssh),
        teardownSsh: async scope => {
          expect(scope).toBe(key)
          events.push('ssh')
        },
        teardownPool: stopper.stop,
        teardownPrimary: primary,
        notifyApplied: vi.fn()
      }
    )
    expect(events).toEqual(connectionId === 'local' ? ['child', 'exit'] : ['ssh', 'child', 'exit'])
    expect(pool.has(key)).toBe(false)
    expect(pool.has(backendScopeKey('ssh-b', 'worker'))).toBe(true)
    expect(primary).not.toHaveBeenCalled()
  }
})

it('primary pins preserve recovery but reject unmanaged and changed owners before teardown', async () => {
  let resolve!: (value: ResolvedConnectionDescriptor) => void

  let promise = new Promise<ResolvedConnectionDescriptor>(yes => {
    resolve = yes
  })

  const ssh = { remotePlatform: 'Linux' }
  const teardownSsh = vi.fn(async () => {})
  const teardownPrimary = vi.fn(async () => {})

  const deps = {
    registry,
    routeOptions: { primaryProfile: 'default' },
    effectiveSshFingerprint: async () => 'fixture',
    primarySshKey: '',
    primaryPromise: () => promise,
    pool: new Map(),
    sshState: () => ssh,
    teardownSsh,
    teardownPool: vi.fn(async () => {}),
    teardownPrimary,
    notifyApplied: vi.fn()
  }

  const owner = { connectionId: 'ssh-a', profile: 'default' }
  const run = recyclePinnedBackend(owner, deps)
  promise = Promise.resolve({ mode: 'local' })
  resolve({ mode: 'remote', remoteKind: 'ssh', connectionId: 'ssh-a', ssh: { effectiveConfigFingerprint: 'fixture' } })
  await expect(run).rejects.toThrow('changed')
  await expect(recyclePinnedBackend({ connectionId: 'url', profile: 'default' }, deps)).rejects.toThrow('not managed')
  expect(teardownSsh).not.toHaveBeenCalled()
  expect(teardownPrimary).not.toHaveBeenCalled()
  promise = Promise.resolve({
    mode: 'remote',
    remoteKind: 'ssh',
    connectionId: 'ssh-a',
    ssh: { effectiveConfigFingerprint: 'fixture' }
  })
  await expect(recyclePinnedBackend(owner, deps)).resolves.toBe('primary')
  expect(teardownPrimary).toHaveBeenCalledOnce()
  expect(deps.notifyApplied).toHaveBeenCalledOnce()
})

it('legacy SSH recovery validates the descriptor before the existing ordered teardown', async () => {
  const descriptor = {
    mode: 'remote' as const,
    remoteKind: 'ssh' as const,
    baseUrl: 'http://127.0.0.1:1234',
    token: 'fixture'
  }

  let promise = Promise.resolve(descriptor)
  const events: string[] = []
  const ssh = { remotePlatform: 'Linux' }

  const deps = {
    registry,
    routeOptions: { globalRemote: true, primaryProfile: 'default' },
    primarySshKey: '',
    effectiveSshFingerprint: async () => '',
    primaryPromise: () => promise,
    pool: new Map(),
    sshState: () => ssh,
    teardownSsh: async () => {
      events.push('ssh')
    },
    teardownPool: vi.fn(async () => {}),
    teardownPrimary: async () => {
      events.push('primary')
    },
    notifyApplied: () => {
      events.push('applied')
    }
  }

  const owner = { connectionId: null, profile: 'worker', legacyConnection: descriptor }
  await expect(recyclePinnedBackend(owner, deps)).resolves.toBe('primary')
  expect(events).toEqual(['ssh', 'primary', 'applied'])
  events.length = 0
  promise = Promise.resolve({ ...descriptor, baseUrl: 'http://127.0.0.1:4321' })
  await expect(recyclePinnedBackend(owner, deps)).rejects.toThrow('Backend changed')
  expect(events).toEqual([])
  expect(deps.teardownPool).not.toHaveBeenCalled()
})

describe('recycleOwnedBackendTarget', () => {
  it('treats an empty or matching profile as the primary backend', () => {
    expect(recycleOwnedBackendTarget(undefined, 'default')).toBe('primary')
    expect(recycleOwnedBackendTarget('', 'default')).toBe('primary')
    expect(recycleOwnedBackendTarget('default', 'default')).toBe('primary')
  })

  it('treats any other named profile as a pooled backend', () => {
    expect(recycleOwnedBackendTarget('paid-ads', 'default')).toBe('pool')
  })
})

describe('recycleOwnedBackend', () => {
  it('kills the owned SSH serve before the primary child, then notifies apply', async () => {
    const events: string[] = []

    const target = await recycleOwnedBackend({
      notifyApplied: () => events.push('applied'),
      primaryProfile: 'default',
      profile: undefined,
      teardownPool: async () => {
        events.push('pool')
      },
      teardownPrimary: async () => {
        events.push('primary')
      },
      teardownSsh: async profile => {
        events.push(`ssh:${profile}`)
      }
    })

    expect(target).toBe('primary')
    expect(events).toEqual(['ssh:', 'primary', 'applied'])
  })

  it('recycles a pooled profile without tearing down the primary', async () => {
    const events: string[] = []

    const target = await recycleOwnedBackend({
      notifyApplied: () => events.push('applied'),
      primaryProfile: 'default',
      profile: 'paid-ads',
      teardownPool: async profile => {
        events.push(`pool:${profile}`)
      },
      teardownPrimary: async () => {
        events.push('primary')
      },
      teardownSsh: async profile => {
        events.push(`ssh:${profile}`)
      }
    })

    expect(target).toBe('pool')
    expect(events).toEqual(['ssh:paid-ads', 'pool:paid-ads'])
  })

  it('awaits SSH teardown before the local child even when SSH is slow', async () => {
    const events: string[] = []
    let releaseSsh!: () => void

    const sshGate = new Promise<void>(resolve => {
      releaseSsh = resolve
    })

    const run = recycleOwnedBackend({
      notifyApplied: () => events.push('applied'),
      primaryProfile: 'default',
      teardownPool: vi.fn(),
      teardownPrimary: async () => {
        events.push('primary')
      },
      teardownSsh: async () => {
        events.push('ssh-start')
        await sshGate
        events.push('ssh-done')
      }
    })

    await Promise.resolve()
    expect(events).toEqual(['ssh-start'])

    releaseSsh()
    await run

    expect(events).toEqual(['ssh-start', 'ssh-done', 'primary', 'applied'])
  })
})
