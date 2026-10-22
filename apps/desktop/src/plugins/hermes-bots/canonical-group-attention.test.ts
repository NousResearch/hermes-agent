import { expect, it } from 'vitest'

import { canonicalRoomAttention } from './canonical-group-activity'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'

it('calls the room blocked only for a block, not for a Stop that is settling', () => {
  const blocked = (counts?: Record<string, number>) => canonicalRoomAttention({ blocked: true, counts }, CANONICAL_GROUP_LOCALES.en)
  expect([blocked({ stopping: 1 }), blocked({ stopping: 1, indeterminate: 1 }), blocked({ queued: 1 }), blocked()])
    .toEqual([[], ['Blocked'], ['Blocked'], ['Blocked']])
  // A driver that is down sends no status; the room's unavailable notice says so instead.
  expect(canonicalRoomAttention(undefined, CANONICAL_GROUP_LOCALES.en)).toEqual([])
})
