import { describe, expect, it } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'
import { makeSessionInfo } from '@/test/session-info'

import { buildGatewaySessionGroups, scopeGatewaySessionGroups } from './gateway-group-model'

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
      [JSON.stringify(['local', 'default'])]: ['a', 'c', 'd'],
      [JSON.stringify(['remote-1', 'default'])]: ['b']
    })

    for (const group of groups) {
      expect(group.sessions.every(session => (session.connection_id || registry.primary) === group.connectionId)).toBe(
        true
      )
    }
  })

  it('attributes rows with no connection_id to the primary gateway instead of floating them', () => {
    const groups = buildGatewaySessionGroups(rows, registry, {})

    expect(groups.some(group => group.connectionId === null)).toBe(false)

    const primary = groups.find(group => group.connectionId === 'local')!

    expect(primary.sessions.map(session => session.id)).toContain('c')
  })

  it('follows the registry primary and never rewrites explicit ownership', () => {
    const remotePrimary = { ...registry, primary: 'remote-1' } as DesktopConnectionsRegistry
    const groups = buildGatewaySessionGroups(rows, remotePrimary, {})

    // c carried no connection_id → it follows primary='remote-1' and joins b.
    expect(members(groups)[JSON.stringify(['remote-1', 'default'])]).toEqual(['b', 'c'])
    // a/d carried an explicit 'local' and are never pulled into the remote group.
    expect(members(groups)[JSON.stringify(['local', 'default'])]).toEqual(['a', 'd'])
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
