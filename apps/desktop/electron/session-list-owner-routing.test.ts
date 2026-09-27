import { describe, expect, it } from 'vitest'

import { createPooledRegistrySessionSources } from './session-list-owner-routing'

describe('createPooledRegistrySessionSources', () => {
  it('keeps resolved SSH profile owners and skips local unless requested', async () => {
    const descriptor = { baseUrl: 'http://127.0.0.1:9001', mode: 'remote' } as const

    const pooled = createPooledRegistrySessionSources({
      backendIdForConnection: async connectionId => `install-${connectionId}`,
      backendPool: new Map([
        ['conn:ssh-a::worker', { connectionPromise: Promise.resolve(descriptor) }],
        ['conn:local::default', { connectionPromise: Promise.resolve(descriptor) }]
      ]),
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
})
