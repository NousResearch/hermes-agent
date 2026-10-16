import { describe, expect, it } from 'vitest'

import { pathForRegistryBackendRequest } from './connection-config'
import type { ConnectionRegistry, RegistryConnection, ResolvedConnectionDescriptor } from './connection-registry'
import { reuseMatchingPrimaryRemoteBackend } from './connection-registry'

interface LiveDescriptor extends ResolvedConnectionDescriptor {
  baseUrl: string
  wsUrl: string
}

function registryFor(source: RegistryConnection): ConnectionRegistry {
  return {
    version: 2,
    primary: source.id,
    launchMode: 'primary',
    lastUsed: source.id,
    connections: [{ id: 'local', kind: 'local', label: 'This device' }, source]
  }
}

describe('primary-remote descriptor reuse keeps profile scope', (): void => {
  it.each(['remote', 'cloud'] as const)(
    'reuses %s identity and routes each caller without mutating the primary',
    async (kind: 'remote' | 'cloud'): Promise<void> => {
      const source: RegistryConnection = {
        id: 'shared',
        kind,
        label: 'Shared',
        url: 'https://gateway.example',
        authMode: 'token'
      }

      const registry: ConnectionRegistry = registryFor(source)

      const descriptor: LiveDescriptor = {
        mode: 'remote',
        remoteKind: kind === 'remote' ? 'url' : 'cloud',
        baseUrl: source.url!,
        wsUrl: 'wss://gateway.example/ws?ticket=live',
        authMode: 'token'
      }

      const profiles: Array<null | string | undefined> = []

      for (const profile of ['acme', 'research', null]) {
        const result = await reuseMatchingPrimaryRemoteBackend({
          connectionId: source.id,
          profile,
          registry,
          source,
          ensurePrimary: async (requested: null | string | undefined): Promise<LiveDescriptor> => {
            profiles.push(requested)

            return descriptor
          }
        })

        expect(result).toMatchObject({
          ...descriptor,
          profile: profile ?? 'default',
          connectionId: source.id
        })

        if (!result) {
          throw new Error('Expected a reused descriptor')
        }

        expect(pathForRegistryBackendRequest('/api/skills', result.profile, result)).toBe(
          `/api/skills?profile=${profile ?? 'default'}`
        )
        expect(pathForRegistryBackendRequest('/api/skills?profile=explicit', result.profile, result)).toBe(
          '/api/skills?profile=explicit'
        )
      }

      expect(profiles).toEqual(['acme', 'research', null])
      expect(descriptor).not.toHaveProperty('sharedRemote')
      expect(descriptor).not.toHaveProperty('profile')
      expect(pathForRegistryBackendRequest('/api/skills', 'acme', { sharedRemote: false, remoteProfile: null })).toBe(
        '/api/skills'
      )
    }
  )

  it('does not boot ineligible primaries or reuse a different live identity', async (): Promise<void> => {
    const source: RegistryConnection = {
      id: 'shared',
      kind: 'remote',
      label: 'Shared',
      url: 'https://gateway.example',
      authMode: 'token',
      token: 'right-account'
    }

    const registry: ConnectionRegistry = registryFor(source)
    let calls: number = 0

    const descriptor: LiveDescriptor = {
      mode: 'remote',
      remoteKind: 'url',
      baseUrl: 'https://gateway.example',
      wsUrl: 'wss://gateway.example/ws',
      authMode: 'token',
      token: 'wrong-account',
      headers: {}
    }

    const ensurePrimary = async (): Promise<LiveDescriptor> => {
      calls += 1

      return descriptor
    }

    for (const candidate of [
      { source, connectionId: 'secondary' },
      { source: { ...source, kind: 'local' as const }, connectionId: source.id },
      { source: { ...source, kind: 'ssh' as const }, connectionId: source.id }
    ]) {
      expect(
        await reuseMatchingPrimaryRemoteBackend({ ...candidate, profile: 'acme', registry, ensurePrimary })
      ).toBeNull()
    }

    expect(calls).toBe(0)
    expect(
      await reuseMatchingPrimaryRemoteBackend({
        source,
        connectionId: source.id,
        profile: 'acme',
        registry,
        ensurePrimary
      })
    ).toBeNull()
    expect(calls).toBe(1)
  })
})
