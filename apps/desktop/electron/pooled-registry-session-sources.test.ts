import { describe, expect, it, vi } from 'vitest'

import { backendScopeKey, backendScopePrefix } from './connection-registry'
import { createPooledRegistrySessionSources, PENDING_BACKEND_BUDGET_MS } from './pooled-registry-session-sources'

type Entry = { connectionPromise?: unknown }

const resolved = (descriptor: unknown): Entry => ({ connectionPromise: Promise.resolve(descriptor) })

function build(
  connections: Array<{ id: string; kind: string }>,
  pool: Array<[string, Entry]>,
  probeInstallId = vi.fn(async (connectionId: string) => `install-${connectionId}`)
) {
  const sources = createPooledRegistrySessionSources({
    pool: new Map(pool),
    probeInstallId,
    readRegistry: () => ({ connections })
  })

  return { probeInstallId, sources }
}

const LOCAL = { id: 'local', kind: 'local' }
const HOMELAB = { id: 'homelab', kind: 'ssh' }
const CLOUD = { id: 'cloud', kind: 'cloud' }

describe('createPooledRegistrySessionSources', () => {
  it('reads only pooled backends: a connection with no pooled entry is not a source', async () => {
    const { sources } = build([LOCAL, HOMELAB, CLOUD], [[backendScopeKey('cloud', 'default'), resolved({ id: 'c' })]])

    expect((await sources()).map(source => source.connectionId)).toEqual(['cloud'])
  })

  it('leaves the local connection out of the legacy unpinned read', async () => {
    const { sources } = build([LOCAL, CLOUD], [
      [`${backendScopePrefix('local')}default`, resolved({ id: 'l' })],
      [backendScopeKey('cloud', 'default'), resolved({ id: 'c' })]
    ])

    expect((await sources()).map(source => source.connectionId)).toEqual(['cloud'])
  })

  it('opts the local connection in for a registry-pinned read, one backend per profile', async () => {
    const { sources } = build([LOCAL], [
      [`${backendScopePrefix('local')}default`, resolved({ id: 'l-default' })],
      [`${backendScopePrefix('local')}work`, resolved({ id: 'l-work' })]
    ])

    const [local] = await sources(true)

    expect(local.connectionId).toBe('local')
    expect(local.backends.map(backend => backend.profileLabel).sort()).toEqual(['default', 'work'])
  })

  it('gives an ssh connection one labelled backend per pooled profile scope', async () => {
    const { sources } = build([HOMELAB], [
      [backendScopeKey('homelab', 'default'), resolved({ id: 'h-default' })],
      [backendScopeKey('homelab', 'omer'), resolved({ id: 'h-omer' })]
    ])

    const [homelab] = await sources()

    expect(homelab.kind).toBe('ssh')
    expect(homelab.backends.map(backend => backend.profileLabel).sort()).toEqual(['default', 'omer'])
  })

  it('treats a remote/cloud host as one shared backend with no profile label', async () => {
    const { sources } = build([CLOUD], [
      [backendScopeKey('cloud', 'default'), resolved({ id: 'c-default' })],
      [backendScopeKey('cloud', 'other'), resolved({ id: 'c-other' })]
    ])

    const [cloud] = await sources()

    expect(cloud.backends).toHaveLength(1)
    expect(cloud.backends[0].profileLabel).toBeNull()
  })

  it('stamps each source with its gateway install_id', async () => {
    const { probeInstallId, sources } = build([CLOUD], [[backendScopeKey('cloud', 'default'), resolved({ id: 'c' })]])

    const [cloud] = await sources()

    expect(cloud.backendId).toBe('install-cloud')
    expect(probeInstallId).toHaveBeenCalledWith('cloud', { id: 'c' })
  })

  it('skips a still-dialing backend within the sidebar budget instead of waiting on it', async () => {
    vi.useFakeTimers()

    try {
      const { sources } = build([HOMELAB, CLOUD], [
        [backendScopeKey('homelab', 'default'), { connectionPromise: new Promise(() => undefined) }],
        [backendScopeKey('cloud', 'default'), resolved({ id: 'c' })]
      ])

      const pending = sources()
      await vi.advanceTimersByTimeAsync(PENDING_BACKEND_BUDGET_MS)

      expect((await pending).map(source => source.connectionId)).toEqual(['cloud'])
    } finally {
      vi.useRealTimers()
    }
  })

  it('skips a backend whose connection promise rejected', async () => {
    const dead: Entry = { connectionPromise: Promise.reject(new Error('down')) }

    const { sources } = build([HOMELAB, CLOUD], [
      [backendScopeKey('homelab', 'default'), dead],
      [backendScopeKey('cloud', 'default'), resolved({ id: 'c' })]
    ])

    expect((await sources()).map(source => source.connectionId)).toEqual(['cloud'])
  })

  it('ignores a pool entry with no connection promise', async () => {
    const { sources } = build([CLOUD], [[backendScopeKey('cloud', 'default'), {}]])

    expect(await sources()).toEqual([])
  })
})
