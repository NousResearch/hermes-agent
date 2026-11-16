/**
 * Gateway Group Chats in the roster derivation. A gateway room is not a
 * Desktop room record: a Desktop room with the same name must not lend it a
 * section or a saved order. A room an open view reports as working stays
 * Active past the recency window.
 */

import { describe, expect, it } from 'vitest'

import { canonicalRoomLiveKey } from './canonical-group-details'
import type { GatewayGroupRoom } from './canonical-group-registry'
import { deriveRosterRows } from './roster-pane-derivation'
import { groupRowsBySection } from './user-sections'

const binding = { connectionId: 'local', profile: 'default', roomId: 'room-1', roomName: 'Team' }
const HOUR = 3600_000

const gatewayRoom: GatewayGroupRoom = {
  binding,
  key: 'canonical:local:default:room-1',
  members: [],
  name: 'Team',
  room: { room_id: 'room-1', name: 'Team', members: [] },
  updatedAt: Date.now() - HOUR
}

function derive(overrides: Partial<Parameters<typeof deriveRosterRows>[0]> = {}) {
  return deriveRosterRows({
    roster: [],
    allMeta: {},
    gatewayFilter: 'all',
    query: '',
    activityFilter: 'all',
    rowKindFilter: 'all',
    groupRooms: {},
    gatewayRooms: [gatewayRoom],
    gatewayRoomLive: {},
    activeRosterKeys: new Set(),
    gatewayOptions: [],
    activityOf: () => 0,
    isPinned: () => false,
    ...overrides
  })
}

describe('gateway Group Chats in the roster', () => {
  it('a same-name Desktop room lends a gateway room neither its section nor its saved order', () => {
    const desktop = { Team: { members: [], rosterOrder: 0, sectionId: 'work' } } as never
    const gateway = derive({ groupRooms: desktop }).sortedGroupRows.find(row => row.gateway)!
    const blocks = groupRowsBySection([gateway], [{ id: 'work', name: 'Work' }], {}, desktop)

    expect(blocks.find(block => block.id === 'work')!.rows).toEqual([])
    expect(blocks.find(block => block.id === null)!.rows).toEqual([gateway])
  })

  it('stays Active while an open view reports it working, past the recency window', () => {
    const live = { [canonicalRoomLiveKey(binding, 'room-1')]: { working: true, failed: false, needsYou: false } }

    expect(derive({ activityFilter: 'active' }).sortedGroupRows).toEqual([])
    expect(derive({ activityFilter: 'active', gatewayRoomLive: live }).sortedGroupRows.map(row => row.name)).toEqual([
      'Team'
    ])
  })
})
