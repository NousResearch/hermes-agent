import { describe, expect, it, vi } from 'vitest'

import { buildOpaqueProfileRoutes, type ProfileRouteConfig } from './plugin-profile-routes'

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

    expect(routes.map(route => ({ mode: route.mode, profile: route.profile, targetProfile: route.targetProfile }))).toEqual([
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
        profile === 'barry'
          ? config({ mode: 'remote', remoteUrl: 'https://tower.example' })
          : config(),
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
