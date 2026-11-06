import { describe, expect, it } from 'vitest'

import type { DesktopAgentRoster, DesktopRegistryConnection } from '@/global'

import { buildRestGroups, countRestAgents, countWorkingAgents } from './fleet-rail'

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

  it('keeps an expired Cloud source visible but not reachable even with cached profiles', () => {
    const expired: DesktopAgentRoster = {
      ...roster,
      sources: roster.sources.map(source =>
        source.connectionId === 'pandora'
          ? { ...source, reachable: false, error: 'OAuth expired', needsSignIn: true }
          : source
      )
    }

    const groups = buildRestGroups({ activeConnectionId: 'local', connections, roster: expired })
    const cloud = groups.find(group => group.connectionId === 'pandora')
    expect(cloud).toMatchObject({ reachable: false, error: 'OAuth expired', needsSignIn: true })
    expect(cloud?.named.map(agent => agent.profile)).toEqual(['omer', 'scout'])
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
  })

  it('carries a roster worker claim onto the square it belongs to', () => {
    const working: DesktopAgentRoster = {
      ...roster,
      agents: roster.agents.map(agent =>
        agent.connectionId === 'pandora' && agent.profile === 'scout'
          ? { ...agent, work: { lastActive: 1_800_000_000, sessionId: 'sess-9', source: 'kanban' } }
          : agent
      )
    }

    const [pandora] = buildRestGroups({ activeConnectionId: 'local', connections, roster: working })
    const scout = pandora.named.find(agent => agent.profile === 'scout')

    expect(scout?.work).toEqual({ lastActive: 1_800_000_000, sessionId: 'sess-9', source: 'kanban' })
    // A square with no claim carries no `work` key at all, so the rail can tell
    // "idle" from "unknown".
    expect(pandora.named.find(agent => agent.profile === 'omer')).not.toHaveProperty('work')
  })
})

describe('countWorkingAgents', () => {
  const now = 1_800_000_000_000

  const agent = (lastActive?: number) => ({
    ...(lastActive === undefined ? {} : { work: { lastActive, sessionId: `s-${lastActive}`, source: 'kanban' } })
  })

  it('counts nothing when no square carries a claim', () => {
    expect(countWorkingAgents([agent(), agent()], now, now)).toEqual({ stale: false, working: 0 })
  })

  it('counts fresh claims and ignores expired heartbeats', () => {
    const count = countWorkingAgents([agent(now / 1000 - 10), agent(now / 1000 - 400), agent()], now - 5_000, now)

    expect(count).toEqual({ stale: false, working: 1 })
  })

  it('flags the count as stale only when every claim rests on an old roster page', () => {
    const oldRoster = now - 200_000

    expect(countWorkingAgents([agent(now / 1000 - 10)], oldRoster, now)).toEqual({ stale: true, working: 1 })
    expect(countWorkingAgents([agent(now / 1000 - 10), agent(now / 1000 - 20)], now - 5_000, now)).toEqual({
      stale: false,
      working: 2
    })
  })

  it('does not call a mixed fresh/stale count stale', () => {
    // Both claims sit on the same roster page, so they age together: the flag
    // is about the page, not the heartbeat.
    expect(countWorkingAgents([agent(now / 1000 - 10)], 0, now)).toEqual({ stale: true, working: 1 })
  })
})
