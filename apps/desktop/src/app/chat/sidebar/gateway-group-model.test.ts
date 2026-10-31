import { describe, expect, it } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'
import { makeSessionInfo } from '@/test/session-info'

import {
  buildGatewaySessionGroups,
  groupingOwner,
  scopeGatewaySessionGroups,
  showsProfileTotals,
  withEmptyProfileGroups
} from './gateway-group-model'

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
  it('keys groups by owner, so one profile name on two gateways stays two groups', () => {
    const groups = buildGatewaySessionGroups(rows, registry, {})

    expect(members(groups)).toEqual({
      [JSON.stringify(['local', 'default'])]: ['a', 'c', 'd'],
      [JSON.stringify(['remote-1', 'default'])]: ['b']
    })
  })

  it('groups a bare row under the registry primary without re-tagging the row', () => {
    const primary = buildGatewaySessionGroups(rows, registry, {}).find(group => group.connectionId === 'local')!

    expect(primary.sessions.find(session => session.id === 'c')?.connection_id).toBeUndefined()
    expect(groupingOwner(undefined, registry)).toBe('local')
    expect(groupingOwner('  ', registry)).toBe('local')
    expect(groupingOwner('remote-1', registry)).toBe('remote-1')
    expect(groupingOwner(undefined, { ...registry, primary: 'remote-1' })).toBe('remote-1')
  })

  it('leaves bare rows unassigned (flat profile groups) when no registry exists', () => {
    const groups = buildGatewaySessionGroups(rows, null, {})
    const legacy = groups.find(group => group.connectionId === null)!

    expect(groupingOwner(undefined, null)).toBeNull()
    expect(members(groups)[JSON.stringify([null, 'default'])]).toEqual(['c'])
    expect(legacy.label).toBe(legacy.profile)
  })
})

describe('withEmptyProfileGroups', () => {
  it('adds an untagged profile list under the registry primary, never as a duplicate', () => {
    const groups = withEmptyProfileGroups(
      buildGatewaySessionGroups(rows, registry, {}),
      ['default', 'quiet'],
      null,
      registry,
      {}
    )

    expect(groups.map(group => group.id).sort()).toEqual(
      [
        JSON.stringify(['local', 'default']),
        JSON.stringify(['local', 'quiet']),
        JSON.stringify(['remote-1', 'default'])
      ].sort()
    )
  })

  it('keeps an untagged profile list unassigned without a registry', () => {
    const groups = withEmptyProfileGroups([], ['quiet'], null, null, {})

    expect(groups.map(group => group.id)).toEqual([JSON.stringify([null, 'quiet'])])
  })
})

describe('showsProfileTotals', () => {
  it('shows primary-pool totals on bare and primary groups only', () => {
    expect(showsProfileTotals(null, false, 'local')).toBe(true)
    expect(showsProfileTotals('local', false, 'local')).toBe(true)
    expect(showsProfileTotals('remote-1', false, 'local')).toBe(false)
    expect(showsProfileTotals('local', false, null)).toBe(false)
    expect(showsProfileTotals('local', true, 'local')).toBe(false)
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
