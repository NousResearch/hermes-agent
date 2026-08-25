import { describe, expect, it, vi } from 'vitest'

import {
  buildOpaqueProfileRoutes,
  buildRegistryProfileRoutes,
  isLocalEnumerationFailure,
  localRouteFallbackProfiles,
  type ProfileRouteConfig,
  registryGatewayWsUrl,
  undialedSshRouteSeeds
} from './plugin-profile-routes'

function config(overrides: Partial<ProfileRouteConfig> = {}): ProfileRouteConfig {
  return {
    cloudOrg: '',
    mode: 'local',
    remoteUrl: '',
    sshHost: '',
    sshPort: null,
    sshRemoteHermesPath: '',
    sshRemoteProfile: '',
    sshUser: '',
    ...overrides
  }
}

describe('buildOpaqueProfileRoutes', () => {
  it('groups SSH aliases by their effective route without exposing endpoint data', async () => {
    const configs = new Map([
      [
        'research',
        config({
          mode: 'ssh',
          sshHost: 'lab-a',
          sshRemoteHermesPath: '~/.hermes',
          sshRemoteProfile: 'remote-research'
        })
      ],
      [
        'writing',
        config({
          mode: 'ssh',
          sshHost: 'lab-b',
          sshRemoteHermesPath: '~/.hermes',
          sshRemoteProfile: 'remote-writing'
        })
      ]
    ])

    const resolveSsh = vi.fn(async () => ({ hostname: 'gateway.example', port: 22, user: 'hermes' }))

    const routes = await buildOpaqueProfileRoutes({
      getProfileConfig: profile => configs.get(profile) ?? config(),
      globalConfig: config(),
      installationId: 'install-a-secret',
      primaryProfile: 'default',
      profileNames: ['default', 'research', 'writing'],
      resolveSsh
    })

    expect(
      routes.map(route => ({ mode: route.mode, profile: route.profile, targetProfile: route.targetProfile }))
    ).toEqual([
      { mode: 'local', profile: 'default', targetProfile: 'default' },
      { mode: 'remote', profile: 'research', targetProfile: 'remote-research' },
      { mode: 'remote', profile: 'writing', targetProfile: 'remote-writing' }
    ])
    expect(routes[1].connectionId).toBe(routes[2].connectionId)
    expect(routes[0].connectionId).not.toBe(routes[1].connectionId)
    expect(JSON.stringify(routes)).not.toContain('gateway.example')
    expect(JSON.stringify(routes)).not.toContain('lab-a')
    expect(JSON.stringify(routes)).not.toContain('.hermes')
  })

  it('changes opaque IDs when the effective SSH destination changes', async () => {
    const options = {
      getProfileConfig: () => config({ mode: 'ssh', sshHost: 'lab', sshRemoteHermesPath: '~/.hermes' }),
      globalConfig: config(),
      installationId: 'install-a-secret',
      primaryProfile: 'default',
      profileNames: ['default', 'worker']
    }

    const before = await buildOpaqueProfileRoutes({
      ...options,
      resolveSsh: async () => ({ hostname: 'old.example', port: 22, user: 'hermes' })
    })

    const after = await buildOpaqueProfileRoutes({
      ...options,
      resolveSsh: async () => ({ hostname: 'new.example', port: 22, user: 'hermes' })
    })

    expect(before[1].connectionId).not.toBe(after[1].connectionId)
  })

  it('keys IDs to the Desktop installation', async () => {
    const options = {
      getProfileConfig: () => config({ mode: 'remote', remoteUrl: 'https://gateway.example' }),
      globalConfig: config(),
      primaryProfile: 'default',
      profileNames: ['default', 'worker'],
      resolveSsh: vi.fn()
    }

    const first = await buildOpaqueProfileRoutes({ ...options, installationId: 'install-a-secret' })
    const second = await buildOpaqueProfileRoutes({ ...options, installationId: 'install-b-secret' })

    expect(first[1].connectionId).not.toBe(second[1].connectionId)
  })

  it('isolates an SSH resolution failure to its configured route', async () => {
    const options = {
      getProfileConfig: (profile: string) =>
        profile === 'broken'
          ? config({ mode: 'ssh', sshHost: 'unreachable', sshPort: 2222, sshUser: 'hermes' })
          : config(),
      globalConfig: config(),
      installationId: 'install-a-secret',
      primaryProfile: 'default',
      profileNames: ['default', 'broken']
    }

    const routes = await buildOpaqueProfileRoutes({
      ...options,
      resolveSsh: async route => {
        if (route.sshHost === 'unreachable') {
          throw new Error('ssh -G timed out')
        }

        return { hostname: route.sshHost, port: route.sshPort, user: route.sshUser }
      }
    })

    expect(routes).toHaveLength(2)
    expect(routes.find(route => route.profile === 'broken')).toMatchObject({ mode: 'remote', profile: 'broken' })
  })

  it('inherits the global remote gateway and deduplicates profile names', async () => {
    const routes = await buildOpaqueProfileRoutes({
      getProfileConfig: () => config(),
      globalConfig: config({ mode: 'remote', remoteUrl: 'https://gateway.example/' }),
      installationId: 'install-a-secret',
      primaryProfile: 'default',
      profileNames: ['default', 'alpha', 'beta', 'alpha'],
      resolveSsh: vi.fn()
    })

    expect(routes.map(route => route.profile)).toEqual(['default', 'alpha', 'beta'])
    expect(routes.map(route => route.targetProfile)).toEqual(['default', 'alpha', 'beta'])
    expect(routes.every(route => route.mode === 'remote')).toBe(true)
    expect(new Set(routes.map(route => route.connectionId))).toHaveLength(1)
  })

  it('reports the backend root for a per-profile URL alias', async () => {
    const routes = await buildOpaqueProfileRoutes({
      getProfileConfig: profile =>
        profile === 'barry' ? config({ mode: 'remote', remoteUrl: 'https://tower.example' }) : config(),
      globalConfig: config(),
      installationId: 'install-a-secret',
      primaryProfile: 'default',
      profileNames: ['default', 'barry'],
      resolveSsh: vi.fn()
    })

    expect(routes[1]).toMatchObject({ profile: 'barry', targetProfile: 'default' })
  })

  it('reports an explicit backend profile inherited from global SSH', async () => {
    const routes = await buildOpaqueProfileRoutes({
      getProfileConfig: () => config(),
      globalConfig: config({
        mode: 'ssh',
        sshHost: 'gateway',
        sshRemoteProfile: 'remote-primary'
      }),
      installationId: 'install-a-secret',
      primaryProfile: 'default',
      profileNames: ['default', 'desktop-alias'],
      resolveSsh: vi.fn(async () => ({ hostname: 'gateway.example', port: 22, user: 'hermes' }))
    })

    expect(routes.map(route => route.targetProfile)).toEqual(['remote-primary', 'remote-primary'])
  })

  it('keeps cloud organizations on one service URL in distinct groups', async () => {
    const routes = await buildOpaqueProfileRoutes({
      getProfileConfig: profile =>
        profile === 'org-a' || profile === 'org-b'
          ? config({ cloudOrg: profile, mode: 'cloud', remoteUrl: 'https://cloud.example' })
          : config(),
      globalConfig: config(),
      installationId: 'install-a-secret',
      primaryProfile: 'default',
      profileNames: ['default', 'org-a', 'org-b'],
      resolveSsh: vi.fn()
    })

    expect(new Set(routes.map(route => route.connectionId))).toHaveLength(3)
    expect(JSON.stringify(routes.map(({ connectionId, mode }) => ({ connectionId, mode })))).not.toContain('org-a')
  })
})

describe('buildRegistryProfileRoutes', () => {
  it('keeps duplicate profile names distinct by registry connection without exposing source details', () => {
    const routes = buildRegistryProfileRoutes({
      agents: [
        { connectionId: 'local', profile: 'research' },
        { connectionId: 'homelab', profile: 'research' }
      ],
      legacyRoutes: [{ connectionId: 'legacy-hash', mode: 'local', profile: 'research', targetProfile: 'research' }],
      sources: [
        { id: 'local', kind: 'local', label: 'This device' },
        {
          authMode: 'token',
          host: 'private.lan',
          id: 'homelab',
          kind: 'ssh',
          keyPath: '/secret/id_ed25519',
          label: 'Homelab',
          remoteProfile: 'remote-research',
          token: 'encrypted-secret'
        }
      ]
    })

    expect(routes).toEqual([
      { connectionId: 'local', mode: 'local', profile: 'research', targetProfile: 'research' },
      { connectionId: 'homelab', mode: 'remote', profile: 'research', targetProfile: 'remote-research' }
    ])
    expect(JSON.stringify(routes)).not.toContain('private.lan')
    expect(JSON.stringify(routes)).not.toContain('id_ed25519')
    expect(JSON.stringify(routes)).not.toContain('encrypted-secret')
    expect(new Set(routes.map(route => `${route.connectionId}/${route.profile}`))).toHaveLength(2)
  })

  it('keeps the registry local source genuinely local when legacy v1 routing is remote', () => {
    const routes = buildRegistryProfileRoutes({
      agents: [{ connectionId: 'local', profile: 'barry' }],
      legacyRoutes: [{ connectionId: 'legacy-hash', mode: 'remote', profile: 'barry', targetProfile: 'default' }],
      sources: [{ id: 'local', kind: 'local', label: 'This device' }]
    })

    expect(routes).toEqual([{ connectionId: 'local', mode: 'local', profile: 'barry', targetProfile: 'barry' }])
  })

  it('scopes registry-shared remote websocket URLs to the requested profile', () => {
    expect(
      registryGatewayWsUrl({ profile: 'research', sharedRemote: true }, 'wss://gateway.example/api/ws?token=secret')
    ).toBe('wss://gateway.example/api/ws?token=secret&profile=research')
    expect(registryGatewayWsUrl({ profile: 'research' }, 'ws://127.0.0.1:5151/api/ws?token=local')).toBe(
      'ws://127.0.0.1:5151/api/ws?token=local'
    )
  })
})

describe('isLocalEnumerationFailure', () => {
  it('does not treat an intentionally deferred local enumeration as a failure', () => {
    expect(isLocalEnumerationFailure('connect-on-demand')).toBe(false)
  })

  it('treats any other enumeration error as a failure', () => {
    expect(isLocalEnumerationFailure('ECONNREFUSED')).toBe(true)
  })

  it('treats a missing error as no failure', () => {
    expect(isLocalEnumerationFailure(undefined)).toBe(false)
  })
})

describe('localRouteFallbackProfiles', () => {
  it('restores failed local profiles when another source returned agents', () => {
    const agents = [{ connectionId: 'cloud-prod', profile: 'default' }]

    expect(localRouteFallbackProfiles(agents, 'local', ['default', 'venture'], true)).toEqual(['default', 'venture'])
  })

  it('does not synthesize local routes after a successful local enumeration', () => {
    expect(localRouteFallbackProfiles([], 'local', ['default'], false)).toEqual([])
  })
})

describe('undialedSshRouteSeeds', () => {
  it('keeps a stable default route while retaining the configured backend target', () => {
    expect(
      undialedSshRouteSeeds(
        [],
        [
          { id: 'homelab', kind: 'ssh', remoteProfile: 'venture' },
          { id: 'cloud-prod', kind: 'cloud' }
        ]
      )
    ).toEqual([{ connectionId: 'homelab', profile: 'default' }])
  })

  it('does not add a speculative seed after the source has roster agents', () => {
    expect(
      undialedSshRouteSeeds(
        [{ connectionId: 'homelab', profile: 'research' }],
        [{ id: 'homelab', kind: 'ssh', remoteProfile: 'venture' }]
      )
    ).toEqual([])
  })
})
