import { describe, expect, it } from 'vitest'

import type { DesktopAgentRoster, DesktopRegistryConnection } from '@/global'

import { buildRestGroups, countRestAgents, fleetRouteKey } from './fleet-rail'

const connections: DesktopRegistryConnection[] = [
  { id: 'pandora', kind: 'remote', label: 'Pandora', url: 'https://pandora.example' },
  { id: 'local', kind: 'local', label: 'This device' },
  { id: 'vps', kind: 'ssh', label: 'VPS', host: 'vps.example' }
] as DesktopRegistryConnection[]

const roster: DesktopAgentRoster = {
  agents: [
    {
      connectionId: 'pandora',
      connectionKind: 'remote',
      connectionLabel: 'Pandora',
      profile: 'default',
      handle: 'hermes-pandora'
    },
    {
      connectionId: 'pandora',
      connectionKind: 'remote',
      connectionLabel: 'Pandora',
      profile: 'scout',
      handle: 'scout'
    },
    {
      connectionId: 'pandora',
      connectionKind: 'remote',
      connectionLabel: 'Pandora',
      profile: 'omer',
      handle: 'omer-pandora'
    },
    {
      connectionId: 'local',
      connectionKind: 'local',
      connectionLabel: 'This device',
      profile: 'default',
      handle: 'hermes'
    },
    {
      connectionId: 'local',
      connectionKind: 'local',
      connectionLabel: 'This device',
      profile: 'omer',
      handle: 'omer-this-device'
    }
  ],
  sources: [
    { connectionId: 'pandora', kind: 'remote', label: 'Pandora', reachable: true },
    { connectionId: 'local', kind: 'local', label: 'This device', reachable: true },
    { connectionId: 'vps', kind: 'ssh', label: 'VPS', reachable: false, error: 'ssh: connect timed out' }
  ]
}

describe('buildRestGroups', () => {
  it('lists every gateway except the active one, in switcher order, regardless of which is active', () => {
    const fromPandora = buildRestGroups({ activeConnectionId: 'pandora', connections, roster })
    const fromLocal = buildRestGroups({ activeConnectionId: 'local', connections, roster })

    // This device first (switcher order), then by label — never "active first".
    expect(fromPandora.map(group => group.connectionId)).toEqual(['local', 'vps'])
    expect(fromLocal.map(group => group.connectionId)).toEqual(['pandora', 'vps'])
  })

  it('carries each gateway default as its own square plus named profiles alphabetically', () => {
    const [local] = buildRestGroups({ activeConnectionId: 'pandora', connections, roster })

    expect(local.defaultAgent).toMatchObject({
      connectionId: 'local',
      profile: 'default',
      isDefault: true,
      handle: 'hermes'
    })
    expect(local.named.map(agent => agent.profile)).toEqual(['omer'])
    expect(local.named[0]).toMatchObject({
      connectionLabel: 'This device',
      handle: 'omer-this-device',
      isDefault: false
    })

    const [pandora] = buildRestGroups({ activeConnectionId: 'local', connections, roster })
    expect(pandora.named.map(agent => agent.profile)).toEqual(['omer', 'scout'])
  })

  it('keeps an unreachable gateway on the strip with its default square and marks it', () => {
    const groups = buildRestGroups({ activeConnectionId: 'pandora', connections, roster })
    const vps = groups.find(group => group.connectionId === 'vps')

    expect(vps).toBeDefined()
    expect(vps?.reachable).toBe(false)
    expect(vps?.defaultAgent.profile).toBe('default')
    expect(vps?.named).toEqual([])
  })

  it('keeps a connect-on-demand gateway at rest — reachable, with its seeded squares — not marked unreachable', () => {
    // This device while the registry primary is remote: Electron does not
    // dial it, but inventories its profiles from disk and tags the source
    // connect-on-demand (same as an undialed SSH box). That is a sleeping
    // source, not a failed one — no amber dot, real named squares.
    const onDemand: DesktopAgentRoster = {
      agents: roster.agents,
      sources: roster.sources.map(source =>
        source.connectionId === 'local' ? { ...source, error: 'connect-on-demand' } : source
      )
    }

    const [local] = buildRestGroups({ activeConnectionId: 'pandora', connections, roster: onDemand })

    expect(local.connectionId).toBe('local')
    expect(local.reachable).toBe(true)
    expect(local.named.map(agent => agent.profile)).toEqual(['omer'])

    // A desktop with no local runtime at all: nothing to seed (profiles null →
    // reachable false from Electron) but the deferral is still not a failure.
    const noRuntime: DesktopAgentRoster = {
      agents: roster.agents.filter(agent => agent.connectionId !== 'local'),
      sources: roster.sources.map(source =>
        source.connectionId === 'local' ? { ...source, reachable: false, error: 'connect-on-demand' } : source
      )
    }

    const [bare] = buildRestGroups({ activeConnectionId: 'pandora', connections, roster: noRuntime })
    expect(bare.reachable).toBe(true)
    expect(bare.named).toEqual([])
  })

  it('shows every gateway with just its default before the roster has loaded', () => {
    const groups = buildRestGroups({ activeConnectionId: 'pandora', connections, roster: null })

    expect(groups.map(group => [group.connectionId, group.reachable, group.named.length])).toEqual([
      ['local', true, 0],
      ['vps', true, 0]
    ])
  })

  it('skips a registration the roster collapsed into another (same backend, two addresses)', () => {
    const twin: DesktopRegistryConnection = {
      id: 'pandora-lan',
      kind: 'remote',
      label: 'Pandora LAN',
      url: 'http://10.0.0.2'
    } as DesktopRegistryConnection

    const groups = buildRestGroups({ activeConnectionId: 'local', connections: [...connections, twin], roster })

    expect(groups.map(group => group.connectionId)).toEqual(['pandora', 'vps'])
  })

  it('counts every at-rest square for the condensed threshold', () => {
    const groups = buildRestGroups({ activeConnectionId: 'pandora', connections, roster })

    // local: default + omer; vps: default
    expect(countRestAgents(groups)).toBe(3)
    expect(fleetRouteKey('local', 'omer')).toBe('local::omer')
  })
})
