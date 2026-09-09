import { describe, expect, it, vi } from 'vitest'

import { createScopedBackendRecycler } from './backend-recycle'
import { normalizeRegistry } from './connection-registry'

function fixture() {
  const pool = new Map()

  const snapshot = {
    registry: normalizeRegistry({
      connections: [
        { id: 'other', kind: 'remote', label: 'Other', url: 'https://other.invalid' },
        { id: 'cloud', kind: 'cloud', label: 'Cloud', url: 'https://cloud.invalid' },
        { id: 'ssh', kind: 'ssh', label: 'SSH', host: 'ssh.invalid' }
      ]
    }),
    routeOptions: {
      primaryProfile: 'coder',
      globalRemote: false,
      primaryRemoteActive: false,
      profileRemoteOverride: false
    },
    primary: {
      process: { killed: false, exitCode: null },
      connectionPromise: Promise.resolve({ mode: 'local', profile: 'coder' })
    },
    pool
  }

  const stopPrimary = vi.fn(async () => {})
  const stopPool = vi.fn(async (_key: string) => {})
  const api = createScopedBackendRecycler({ readState: () => snapshot, stopPrimary, stopPool })

  return { api, snapshot, stopPrimary, stopPool }
}

describe('registry-scoped backend recycling', () => {
  it('retires only the selected local owner and waits for exit, without global apply', async () => {
    const { api, snapshot, stopPrimary, stopPool } = fixture()
    const target = { connectionId: 'local', profile: 'coder' }
    let exited!: () => void

    const exit = new Promise<void>(resolve => {
      exited = resolve
    })

    stopPrimary.mockImplementation(() => exit)

    expect(await api.capability(target)).toEqual({ supported: true })
    expect(stopPrimary).not.toHaveBeenCalled()
    const pending = api.recycle(target)
    let settled = false
    void pending.then(() => {
      settled = true
    })
    await vi.waitFor(() => expect(stopPrimary).toHaveBeenCalledOnce())
    expect(settled).toBe(false)
    exited()
    expect(await pending).toEqual({ status: 'recycled', ...target })
    expect(stopPool).not.toHaveBeenCalled()

    const child = {
      process: { killed: false, exitCode: null },
      connectionPromise: Promise.resolve({ mode: 'local', profile: 'writer' })
    }

    snapshot.pool.set('writer', child)
    expect(await api.recycle({ connectionId: 'local', profile: 'writer' })).toEqual({
      status: 'recycled',
      connectionId: 'local',
      profile: 'writer'
    })
    expect(stopPool).toHaveBeenLastCalledWith('writer')

    // Migrated remote primary: the bare coder slot belongs to another gateway.
    snapshot.routeOptions.globalRemote = true
    snapshot.routeOptions.primaryRemoteActive = true
    snapshot.pool.set('coder', { ...child, connectionPromise: Promise.resolve({ mode: 'remote', profile: 'coder' }) })
    snapshot.pool.set('conn:local::coder', {
      ...child,
      connectionPromise: Promise.resolve({ mode: 'local', profile: 'coder' })
    })
    expect(await api.recycle(target)).toEqual({ status: 'recycled', ...target })
    expect(stopPool.mock.calls).toEqual([['writer'], ['conn:local::coder']])
    expect(stopPrimary).toHaveBeenCalledOnce()
  })

  it('rejects local aliases to remote routes and descriptors that do not prove the exact owner', async () => {
    for (const change of [
      (s: ReturnType<typeof fixture>['snapshot']) => {
        s.routeOptions.profileRemoteOverride = true
      },
      (s: ReturnType<typeof fixture>['snapshot']) => {
        s.routeOptions.primaryRemoteActive = true
      },
      (s: ReturnType<typeof fixture>['snapshot']) => {
        s.primary.process.killed = true
      },
      (s: ReturnType<typeof fixture>['snapshot']) => {
        s.primary.connectionPromise = Promise.resolve({ mode: 'remote', profile: 'coder' })
      },
      (s: ReturnType<typeof fixture>['snapshot']) => {
        s.primary.connectionPromise = Promise.resolve({ mode: 'local', profile: 'other' })
      },
      (s: ReturnType<typeof fixture>['snapshot']) => {
        s.primary.connectionPromise = Promise.resolve({ mode: 'local', profile: 'coder', connectionId: 'other' })
      }
    ]) {
      const { api, snapshot, stopPool, stopPrimary } = fixture()
      change(snapshot)
      expect(await api.recycle({ connectionId: 'local', profile: 'coder' })).toEqual({
        status: 'unsupported',
        reason: 'not-owned'
      })
      expect(stopPool).not.toHaveBeenCalled()
      expect(stopPrimary).not.toHaveBeenCalled()
    }
  })

  it('revalidates the exact process and route after asynchronous ownership resolution', async () => {
    for (const change of [
      (s: ReturnType<typeof fixture>['snapshot']) => {
        s.primary.process = { killed: false, exitCode: null }
      },
      (s: ReturnType<typeof fixture>['snapshot']) => {
        s.routeOptions.globalRemote = true
      },
      (s: ReturnType<typeof fixture>['snapshot']) => {
        s.routeOptions.profileRemoteOverride = true
      }
    ]) {
      const { api, snapshot, stopPool, stopPrimary } = fixture()
      let ready!: (value: { mode: string; profile: string }) => void
      snapshot.primary.connectionPromise = new Promise(resolve => {
        ready = resolve
      })
      const pending = api.recycle({ connectionId: 'local', profile: 'coder' })
      change(snapshot)
      ready({ mode: 'local', profile: 'coder' })
      expect(await pending).toEqual({ status: 'unsupported', reason: 'target-changed' })
      expect(stopPool).not.toHaveBeenCalled()
      expect(stopPrimary).not.toHaveBeenCalled()
    }
  })

  it('does not start a replacement when the selected route changes during child shutdown', async () => {
    const { api, snapshot, stopPrimary } = fixture()
    const startLocal = vi.fn(async () => ({}))
    stopPrimary.mockImplementation(async () => {
      snapshot.routeOptions.globalRemote = true
    })

    await expect(api.restart({ connectionId: 'local', profile: 'coder' }, startLocal)).rejects.toThrow('target-changed')
    expect(startLocal).not.toHaveBeenCalled()
  })

  it('fails closed before lifecycle side effects for invalid and externally owned targets', async () => {
    const { api, stopPrimary, stopPool } = fixture()

    const cases = [
      [undefined, 'invalid-target'],
      [{ profile: 'coder' }, 'invalid-target'],
      [{ connectionId: '', profile: 'coder' }, 'invalid-target'],
      [{ connectionId: 'local', profile: '' }, 'invalid-target'],
      [{ connectionId: ' local', profile: 'coder' }, 'invalid-target'],
      [{ connectionId: 'local', profile: '../coder' }, 'invalid-target'],
      [{ connectionId: 'gone', profile: 'coder' }, 'unknown-connection'],
      [{ connectionId: 'other', profile: 'coder' }, 'externally-managed'],
      [{ connectionId: 'cloud', profile: 'coder' }, 'externally-managed'],
      [{ connectionId: 'ssh', profile: 'coder' }, 'ssh-ownership-unverified']
    ] as const

    for (const [target, reason] of cases) {
      expect(await api.capability(target)).toEqual({ supported: false, reason })
      expect(await api.recycle(target)).toEqual({ status: 'unsupported', reason })
    }

    expect(stopPrimary).not.toHaveBeenCalled()
    expect(stopPool).not.toHaveBeenCalled()
  })
})
