import { describe, expect, it } from 'vitest'

import { createPooledRegistrySessionSources, createRemoteProfileSessionMerger } from './session-list-owner-routing'

describe('createPooledRegistrySessionSources', () => {
  it('keeps resolved SSH profile owners and skips local unless requested', async () => {
    const descriptor = { baseUrl: 'http://127.0.0.1:9001', mode: 'remote' } as const

    const pooled = createPooledRegistrySessionSources({
      backendIdForConnection: async connectionId => `install-${connectionId}`,
      backendPool: new Map([
        ['conn:ssh-a::worker', { connectionPromise: Promise.resolve(descriptor) }],
        ['conn:local::default', { connectionPromise: Promise.resolve(descriptor) }]
      ]),
      primaryBackend: () => null,
      readRegistry: () => ({
        connections: [
          { id: 'local', kind: 'local', label: 'This device', name: 'This device' },
          { host: 'worker@example', id: 'ssh-a', kind: 'ssh', label: 'Worker', name: 'Worker' }
        ],
        lastUsed: 'local',
        launchMode: 'primary',
        primary: 'local',
        version: 2
      })
    })

    await expect(pooled()).resolves.toEqual([
      {
        backendId: 'install-ssh-a',
        backends: [{ descriptor, profileLabel: 'worker' }],
        connectionId: 'ssh-a',
        kind: 'ssh'
      }
    ])
  })

  it('keeps a primary SSH source when its default backend is reused instead of pooled', async () => {
    const descriptor = {
      connectionId: 'ssh-a',
      mode: 'remote',
      remoteKind: 'ssh',
      ssh: { host: 'worker@example' }
    } as const

    const pooled = createPooledRegistrySessionSources({
      backendIdForConnection: async connectionId => `install-${connectionId}`,
      backendPool: new Map(),
      primaryBackend: () => Promise.resolve(descriptor as never),
      readRegistry: () => ({
        connections: [{ host: 'worker@example', id: 'ssh-a', kind: 'ssh', label: 'Worker', name: 'Worker' }],
        lastUsed: 'ssh-a',
        launchMode: 'primary',
        primary: 'ssh-a',
        version: 2
      }),
      sharedProfiles: () => ['default', 'worker']
    })

    await expect(pooled()).resolves.toEqual([
      {
        backendId: 'install-ssh-a',
        backends: [{ descriptor, profileLabel: 'default' }],
        connectionId: 'ssh-a',
        kind: 'ssh',
        sharedProfiles: ['default', 'worker']
      }
    ])
  })
})

describe('createRemoteProfileSessionMerger', () => {
  it('keeps primary rows on their owner when another gateway is pinned', async () => {
    const merge = createRemoteProfileSessionMerger({
      fetchJsonForProfile: async () => ({
        profile_totals: { laptop: 1 },
        sessions: [{ id: 'local-chat', profile: 'laptop' }],
        total: 1
      }),
      getJsonForBackend: async () => ({}),
      pooledRegistrySessionSources: async () => [],
      primaryBackend: async () => ({ mode: 'local' }),
      primaryBackendId: async () => null,
      remoteSessionList: async () => ({ sessions: [], total: 0 })
    })

    const result = (await merge(new URLSearchParams({ profile: 'all' }), [], null, 'remote-a')) as any

    expect(result.sessions[0]).toMatchObject({ connection_id: 'local', id: 'local-chat' })
  })
})
