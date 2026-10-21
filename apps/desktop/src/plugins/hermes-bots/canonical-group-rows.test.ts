import { describe, expect, it } from 'vitest'

import { canonicalMentionHandle, canonicalRoomMembers, mergeCanonicalGroupRows } from './canonical-group-rows'
import type { CanonicalRoom } from './canonical-groups'

// A hosted room lives in the gateway, not in this Desktop's group records, so
// the left roster used to show only the rooms the Desktop had made itself: a
// group created through `groups.create` vanished from the pane. These are the
// merge rules that put it back, and the handle rule that makes @-tags address
// somebody.

const room = (overrides: Partial<CanonicalRoom> = {}): CanonicalRoom => ({
  members: [
    { display_name: 'Atlas Prime', handle: 'atlas', member_id: 'm1', profile: 'atlas' },
    { handle: 'scribe', member_id: 'm2', profile: 'scribe' }
  ],
  name: 'Barbershop project',
  room_id: 'room-1',
  ...overrides
})

describe('mergeCanonicalGroupRows', () => {
  it('adds a hosted room the roster has no record of', () => {
    const merged = mergeCanonicalGroupRows([], [{ key: 'canonical:local:default:room-1', room: room() }])

    expect(merged).toHaveLength(1)
    expect(merged[0].name).toBe('Barbershop project')
    expect(merged[0].canonicalKey).toBe('canonical:local:default:room-1')
    expect(merged[0].members.map(member => member.name)).toEqual(['atlas', 'scribe'])
  })

  it('keeps a known row’s own members and routes it to the room', () => {
    const own = [{ members: [{ name: 'atlas' }], name: 'Barbershop project' }]
    const merged = mergeCanonicalGroupRows(own, [{ key: 'canonical:local:default:room-1', room: room() }])

    expect(merged).toHaveLength(1)
    expect(merged[0].members).toEqual([{ name: 'atlas' }])
    expect(merged[0].canonicalKey).toBe('canonical:local:default:room-1')
  })

  it('leaves rooms the gateway no longer holds, and disbanded ones, off the list', () => {
    const merged = mergeCanonicalGroupRows(
      [{ members: [], name: 'Desk-only room' }],
      [
        { key: 'canonical:local:default:room-2', room: room({ disbanded_at: 1, name: 'Gone', room_id: 'room-2' }) },
        { key: 'canonical:local:default:room-3', room: room({ name: '  ', room_id: 'room-3' }) }
      ]
    )

    expect(merged.map(row => row.name)).toEqual(['Desk-only room'])
    expect(merged[0].canonicalKey).toBeUndefined()
  })
})

describe('canonicalRoomMembers', () => {
  it('seats each profile once and keeps the room’s frozen handle', () => {
    // A profile renamed after the room was made: the room still calls it the
    // OLD handle, and that is the id the gateway resolves.
    const members = canonicalRoomMembers(
      room({
        members: [
          { display_name: 'Atlas Prime', handle: 'atlas', member_id: 'm1', profile: 'atlas-2' },
          { handle: 'atlas', member_id: 'm9', profile: 'atlas-2' }
        ]
      })
    )

    expect(members).toHaveLength(1)
    expect(members[0].name).toBe('atlas-2')
    expect(members[0].handle).toBe('atlas')
  })
})

describe('canonicalMentionHandle', () => {
  it('is the frozen handle, never the friendly name', () => {
    const [member] = canonicalRoomMembers(room())

    expect(member.display_name).toBe('Atlas Prime')
    expect(canonicalMentionHandle(member)).toBe('atlas')
  })

  it('is undefined when the room recorded no handle', () => {
    expect(canonicalMentionHandle({ name: 'atlas' })).toBeUndefined()
  })
})
