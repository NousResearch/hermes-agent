import { expect, it } from 'vitest'

import { canonicalRoomAttention } from './canonical-group-activity'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'

const approvals = (count: number) => ({ pending_actions: Array.from({ length: count }, () => ({ kind: 'approval' })) })

// {count} is substituted into one string, so its wording must read for one approval as well as several.
it('says how many approvals the room waits on in wording that fits one and several', () => {
  expect([1, 2].map(count => canonicalRoomAttention(approvals(count), CANONICAL_GROUP_LOCALES.de))).toEqual([
    ['Freigabe ausstehend: 1'],
    ['Freigabe ausstehend: 2']
  ])
})

it('calls the room blocked only for a block, not for a Stop that is settling', () => {
  const blocked = (counts?: Record<string, number>) => canonicalRoomAttention({ blocked: true, counts }, CANONICAL_GROUP_LOCALES.en)
  expect([blocked({ stopping: 1 }), blocked({ stopping: 1, indeterminate: 1 }), blocked({ queued: 1 }), blocked()])
    .toEqual([[], ['Blocked'], ['Blocked'], ['Blocked']])
  // A driver that is down sends no status; the room's unavailable notice says so instead.
  expect(canonicalRoomAttention(undefined, CANONICAL_GROUP_LOCALES.en)).toEqual([])
})
