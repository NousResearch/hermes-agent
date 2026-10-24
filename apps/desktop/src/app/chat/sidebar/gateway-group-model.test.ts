import { describe, expect, it } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'
import { makeSessionInfo } from '@/test/session-info'
import type { ProfileInfo } from '@/types/hermes'

import { buildGatewaySessionGroups, ownerProfileLabeler, scopeGatewaySessionGroups } from './gateway-group-model'

const registry = {
  version: 2,
  primary: 'local',
  secureTokenStorage: true,
  connections: [
    { id: 'local', label: 'This computer', kind: 'local', tokenSet: false, tokenPreview: null },
    { id: 'remote-1', label: 'Homelab', kind: 'remote', tokenSet: false, tokenPreview: null }
  ]
} as DesktopConnectionsRegistry

const rows = [
  makeSessionInfo({ id: 'a', connection_id: 'local', profile: 'default' }),
  makeSessionInfo({ id: 'b', connection_id: 'remote-1', profile: 'default' }),
  makeSessionInfo({ id: 'c', profile: 'default' }),
  makeSessionInfo({ id: 'd', connection_id: 'local', profile: undefined })
]

const members = (groups: ReturnType<typeof buildGatewaySessionGroups>) =>
  Object.fromEntries(groups.map(group => [group.id, group.sessions.map(session => session.id)]))

describe('buildGatewaySessionGroups', () => {
  it('keys groups by exact owner, so one profile name on two gateways stays two groups', () => {
    const groups = buildGatewaySessionGroups(rows, registry, {})

    expect(members(groups)).toEqual({
      [JSON.stringify(['local', 'default'])]: ['a', 'd'],
      [JSON.stringify(['remote-1', 'default'])]: ['b'],
      [JSON.stringify([null, 'default'])]: ['c']
    })

    for (const group of groups) {
      expect(group.sessions.every(session => (session.connection_id || null) === group.connectionId)).toBe(true)
    }
  })

  it('leaves legacy rows without a connection unassigned instead of guessing a gateway', () => {
    const legacy = buildGatewaySessionGroups(rows, registry, {}).find(group => group.connectionId === null)!

    expect(legacy.label).toBe(legacy.profile)
    expect(registry.connections.some(connection => legacy.label.includes(connection.label))).toBe(false)
  })
})

describe('scopeGatewaySessionGroups', () => {
  it('namespaces preference ids and keeps the gateway in labels only when owners mix gateways', () => {
    const recents = buildGatewaySessionGroups(rows, registry, {})
    const mixed = scopeGatewaySessionGroups(recents, 'messaging:telegram')
    const single = scopeGatewaySessionGroups(recents.slice(0, 1), 'messaging:telegram')

    expect(mixed.map(group => group.id).filter(id => recents.some(group => group.id === id))).toEqual([])
    expect(new Set(mixed.map(group => group.label)).size).toBe(mixed.length)
    expect(single[0].label).toBe(single[0].profile)
    expect(mixed.map(group => group.sessions)).toEqual(recents.map(group => group.sessions))
  })
})

describe('owner group display names', () => {
  const localList = [{ name: 'default', display_name: 'NEON', is_default: true }, { name: 'work' }] as ProfileInfo[]
  const remoteList = [{ name: 'default', display_name: 'Homelab main', is_default: true }] as ProfileInfo[]

  const byOwner = (groups: ReturnType<typeof buildGatewaySessionGroups>) =>
    Object.fromEntries(groups.map(group => [group.id, group.label]))

  it('labels a renamed profile by its display name on its own backend while identity keeps the key', () => {
    const labelFor = ownerProfileLabeler({
      active: localList,
      activeConnectionId: null,
      byConnection: new Map([['remote-1', remoteList]]),
      registry
    })

    const groups = buildGatewaySessionGroups(rows, registry, {}, labelFor)

    expect(byOwner(groups)).toEqual({
      [JSON.stringify(['local', 'default'])]: 'This computer · NEON',
      [JSON.stringify(['remote-1', 'default'])]: 'Homelab · Homelab main',
      [JSON.stringify([null, 'default'])]: 'NEON'
    })
    expect(groups.every(group => group.profile === 'default')).toBe(true)
    expect(scopeGatewaySessionGroups(groups.slice(0, 1), 'messaging:telegram')[0].label).toBe(groups[0].profileLabel)
  })

  it("never borrows a same-named profile's display name from another backend", () => {
    // The remote is active, so `active` is ITS list; the primary's list isn't cached.
    const labelFor = ownerProfileLabeler({
      active: remoteList,
      activeConnectionId: 'remote-1',
      byConnection: new Map([['remote-1', remoteList]]),
      registry
    })

    expect(byOwner(buildGatewaySessionGroups(rows, registry, {}, labelFor))).toEqual({
      [JSON.stringify(['local', 'default'])]: 'This computer · default',
      [JSON.stringify(['remote-1', 'default'])]: 'Homelab · Homelab main',
      [JSON.stringify([null, 'default'])]: 'default'
    })
  })
})
