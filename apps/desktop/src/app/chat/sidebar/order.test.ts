import { describe, expect, it } from 'vitest'

import type { SidebarListRow } from '@/lib/session-date-groups'
import type { SessionInfo } from '@/types/hermes'

import {
  orderByIds,
  orderRowsWithinGroups,
  rankSessions,
  reconcileOrderIds,
  reorderableRowIds,
  resolveManualSessionOrderIds,
  sameIds
} from './order'

describe('resolveManualSessionOrderIds', () => {
  it('clears legacy auto-seeded order until the user manually reorders sessions', () => {
    expect(resolveManualSessionOrderIds(['newest', 'older'], ['older', 'newest'], false)).toEqual([])
  })

  it('keeps a manual order and surfaces newly seen sessions first', () => {
    expect(resolveManualSessionOrderIds(['newest', 'older', 'oldest'], ['oldest', 'older'], true)).toEqual([
      'newest',
      'oldest',
      'older'
    ])
  })

  it('clears manual order when none of the saved ids still exist', () => {
    expect(resolveManualSessionOrderIds(['newest'], ['gone'], true)).toEqual([])
  })
})

describe('orderByIds', () => {
  const id = (item: { id: string }) => item.id

  it('returns items untouched when no order is given', () => {
    const items = [{ id: 'a' }, { id: 'b' }]
    expect(orderByIds(items, id, [])).toBe(items)
  })

  it('reorders by the given ids and drops missing ones', () => {
    const items = [{ id: 'a' }, { id: 'b' }, { id: 'c' }]
    // 'b' isn't in the order and sits after 'a' (the first ordered item), so
    // it's an older row: it sinks below the hand-picked pair, not above it.
    expect(orderByIds(items, id, ['c', 'gone', 'a'])).toEqual([{ id: 'c' }, { id: 'a' }, { id: 'b' }])
  })

  it('surfaces items absent from the order first', () => {
    const items = [{ id: 'fresh' }, { id: 'a' }, { id: 'b' }]
    expect(orderByIds(items, id, ['b', 'a'])).toEqual([{ id: 'fresh' }, { id: 'b' }, { id: 'a' }])
  })

  it('never duplicates an item when the persisted order repeats its id', () => {
    const items = [{ id: 'a' }, { id: 'b' }]
    expect(orderByIds(items, id, ['a', 'a', 'b'])).toEqual([{ id: 'a' }, { id: 'b' }])
  })

  it('keeps a newly-loaded older page below the hand-picked order', () => {
    // Callers pass recency-sorted lists, so an unknown id BELOW the ordered
    // ones is an older page that just loaded — hoisting it to the top was
    // dumping 50 stale sessions above the user's arrangement.
    const items = [{ id: 'a' }, { id: 'b' }, { id: 'old1' }, { id: 'old2' }]
    expect(orderByIds(items, id, ['b', 'a'])).toEqual([{ id: 'b' }, { id: 'a' }, { id: 'old1' }, { id: 'old2' }])
  })

  it('splits unknown ids either side of the order by their own position', () => {
    const items = [{ id: 'new' }, { id: 'a' }, { id: 'b' }, { id: 'old' }]
    expect(orderByIds(items, id, ['b', 'a'])).toEqual([{ id: 'new' }, { id: 'b' }, { id: 'a' }, { id: 'old' }])
  })
})

describe('rankSessions', () => {
  const sessions = [{ id: 'newest' }, { id: 'middle' }, { id: 'oldest' }]

  it('leaves the lane alone when the sidebar is on its default sort', () => {
    expect(rankSessions(sessions)).toBe(sessions)
    expect(rankSessions(sessions, [])).toBe(sessions)
  })

  it('applies the active sort key to a lane the flat list never renders', () => {
    expect(rankSessions(sessions, ['oldest', 'newest', 'middle']).map(s => s.id)).toEqual([
      'oldest',
      'newest',
      'middle'
    ])
  })
})

describe('reconcileOrderIds', () => {
  it('returns empty for no current ids', () => {
    expect(reconcileOrderIds([], ['a'])).toEqual([])
  })

  it('returns current ids when there is no saved order', () => {
    expect(reconcileOrderIds(['a', 'b'], [])).toEqual(['a', 'b'])
  })

  it('puts newly-seen ids ahead of the retained saved order', () => {
    expect(reconcileOrderIds(['fresh', 'a', 'b'], ['b', 'a', 'gone'])).toEqual(['fresh', 'b', 'a'])
  })

  it('dedupes a corrupted saved order instead of perpetuating it', () => {
    expect(reconcileOrderIds(['a', 'b'], ['a', 'a', 'b'])).toEqual(['a', 'b'])
  })
})

describe('sameIds', () => {
  it('is true only for identical ordered lists', () => {
    expect(sameIds(['a', 'b'], ['a', 'b'])).toBe(true)
    expect(sameIds(['a', 'b'], ['b', 'a'])).toBe(false)
    expect(sameIds(['a'], ['a', 'b'])).toBe(false)
  })
})

const session = (id: string, branchStem?: string): SidebarListRow => ({
  entry: { branchStem, session: { id } as SessionInfo },
  kind: 'session'
})

const divider = (key: string): SidebarListRow =>
  ({ bucket: { key }, key, kind: 'divider' }) as unknown as SidebarListRow

const ids = (rows: SidebarListRow[]): string[] =>
  rows.map(row => (row.kind === 'divider' ? `--${row.key}--` : `${row.entry.branchStem ?? ''}${row.entry.session.id}`))

describe('orderRowsWithinGroups', () => {
  it('returns the same rows when no order is given', () => {
    const rows = [session('a'), session('b')]
    expect(orderRowsWithinGroups(rows, [])).toBe(rows)
  })

  it('keeps identity when the order changes nothing', () => {
    const rows = [session('a'), session('b')]
    expect(orderRowsWithinGroups(rows, ['a', 'b'])).toBe(rows)
  })

  it('reorders within a group without moving the dividers', () => {
    const rows = [session('a'), session('b'), divider('yesterday'), session('c'), session('d')]

    expect(ids(orderRowsWithinGroups(rows, ['b', 'a', 'd', 'c']))).toEqual(['b', 'a', '--yesterday--', 'd', 'c'])
  })

  it('never moves a row across a date divider', () => {
    // 'd' is ranked first overall, but it lives below the divider and must
    // stay there — chronology owns the groups, the drag only ranks inside one.
    const rows = [session('a'), session('b'), divider('lastWeek'), session('c'), session('d')]

    expect(ids(orderRowsWithinGroups(rows, ['d', 'c', 'b', 'a']))).toEqual(['b', 'a', '--lastWeek--', 'd', 'c'])
  })

  it('leaves rows the order does not name in their recency slot', () => {
    // Only 'a' and 'c' are ranked; 'b' never moves, and the ranked pair swap
    // into the two slots they already occupied.
    const rows = [session('a'), session('b'), session('c')]

    expect(ids(orderRowsWithinGroups(rows, ['c', 'a']))).toEqual(['c', 'b', 'a'])
  })

  it('carries branch children with their parent', () => {
    const rows = [session('parent'), session('child', '└─ '), session('other')]

    expect(ids(orderRowsWithinGroups(rows, ['other', 'parent']))).toEqual(['other', 'parent', '└─ child'])
  })
})

describe('reorderableRowIds', () => {
  it('lists root session ids in render order, skipping dividers and branches', () => {
    const rows = [session('a'), session('branch', '├─ '), divider('yesterday'), session('b')]

    expect(reorderableRowIds(rows)).toEqual(['a', 'b'])
  })
})
