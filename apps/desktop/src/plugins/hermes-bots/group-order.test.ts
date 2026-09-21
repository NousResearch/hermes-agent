import { describe, expect, it } from 'vitest'

import { insertBotOrderByDrop, reorderGroupRows, sortGroupRosterRows } from './group-order'

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

  it('bot drop insertion reassigns the whole band and guards the pin band', () => {
    const band = [
      { name: 'alpha', pinned: false },
      { name: 'bravo', pinned: false },
      { name: 'charlie', pinned: false }
    ]

    // Drop alpha AFTER charlie: whole band reassigned 0..2 in the new sequence.
    expect(insertBotOrderByDrop(band, 'alpha', 'charlie', true)).toEqual({
      bravo: 0,
      charlie: 1,
      alpha: 2
    })

    // Drop charlie BEFORE bravo.
    expect(insertBotOrderByDrop(band, 'charlie', 'bravo', false)).toEqual({
      alpha: 0,
      charlie: 1,
      bravo: 2
    })

    // No-ops: dropping on itself, unknown names.
    expect(insertBotOrderByDrop(band, 'alpha', 'alpha', true)).toBeNull()
    expect(insertBotOrderByDrop(band, 'ghost', 'bravo', true)).toBeNull()

    // Pin-band guard: a pinned target refuses an unpinned drag (and vice
    // versa) — pinning stays the outer band.
    const mixed = [
      { name: 'pin', pinned: true },
      { name: 'loose', pinned: false }
    ]
    expect(insertBotOrderByDrop(mixed, 'loose', 'pin', true)).toBeNull()
  })

  it('manual bot order drives bot slots without touching room slots', () => {
    const rows = [
      { kind: 'bot' as const, name: 'BotA', activity: 1, order: 1, pinned: false },
      { kind: 'bot' as const, name: 'BotB', activity: 2, order: 0, pinned: false },
      { kind: 'group' as const, name: 'Room', activity: 3, pinned: false }
    ]

    // BotB carries order 0, BotA order 1: the BOT slots render BotB above
    // BotA regardless of activity; the room keeps the activity slot above both.
    expect(sortGroupRosterRows(rows, {}).map(row => row.name)).toEqual(['Room', 'BotB', 'BotA'])
  })
})
