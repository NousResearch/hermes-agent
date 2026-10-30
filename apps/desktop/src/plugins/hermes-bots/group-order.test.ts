import { describe, expect, it } from 'vitest'

import { reorderGroupRows, sortGroupRosterRows } from './group-order'

const rows = [
  { kind: 'group' as const, name: 'Older', activity: 1, pinned: false },
  { kind: 'bot' as const, name: 'Bot', activity: 2, pinned: false },
  { kind: 'group' as const, name: 'Newer', activity: 3, pinned: false },
  { kind: 'group' as const, name: 'Pinned', activity: 0, pinned: true }
]

describe('room display order', () => {
  it('keeps legacy recency until explicitly ordered, then only replaces room slots within pin bands', () => {
    expect(sortGroupRosterRows(rows, {}).map(row => row.name)).toEqual(['Pinned', 'Newer', 'Bot', 'Older'])
    const rooms = { Older: { rosterOrder: 0 }, Newer: { rosterOrder: 1 } }
    expect(sortGroupRosterRows(rows, rooms).map(row => row.name)).toEqual(['Pinned', 'Older', 'Bot', 'Newer'])
    expect(
      sortGroupRosterRows(
        rows.map(row => ({ ...row, activity: row.name === 'Newer' ? 999 : row.activity })),
        rooms
      )
        .filter(row => row.kind === 'group')
        .map(row => row.name)
    ).toEqual(['Pinned', 'Older', 'Newer'])
    expect(rows[0].name).toBe('Older')
  })

  it('moves only visible same-band rooms while retaining hidden slots and ignoring stale targets', () => {
    const ordered = sortGroupRosterRows(
      rows.filter(row => row.kind === 'group'),
      {}
    )

    expect(reorderGroupRows(ordered, 'Older', -1)).toEqual(['Pinned', 'Older', 'Newer'])
    expect(reorderGroupRows(ordered, 'Newer', -1)).toBeNull()
    expect(reorderGroupRows(ordered, 'deleted', 1)).toBeNull()
    const hidden = { kind: 'group' as const, name: 'Hidden', activity: 2, pinned: false }
    expect(reorderGroupRows([ordered[0], ordered[1], hidden, ordered[2]], 'Older', -1, ['Newer', 'Older'])).toEqual([
      'Pinned',
      'Older',
      'Hidden',
      'Newer'
    ])
  })
})
