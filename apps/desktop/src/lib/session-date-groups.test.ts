import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { makeSessionInfo } from '../test/session-info'

import type { SidebarSessionEntry } from './session-branch-tree'
import {
  filterCollapsedDateGroupRows,
  groupEntriesByRecency,
  type SidebarListRow,
  toSessionRows
} from './session-date-groups'

const session = (id: string, overrides: Partial<SessionInfo> = {}): SessionInfo =>
  makeSessionInfo({ id, message_count: 1, source: 'cli', title: id, ...overrides })

const entry = (value: SessionInfo, branchStem?: string): SidebarSessionEntry =>
  branchStem ? { branchStem, session: value } : { session: value }

// Thursday 18 June 2026, local noon; the pinned week starts on Monday.
const NOW = new Date(2026, 5, 18, 12, 0, 0).getTime()
const MONDAY = 1

const at = (year: number, month: number, day: number, hour = 10): number =>
  Math.floor(new Date(year, month, day, hour, 0, 0).getTime() / 1000)

const group = (entries: SidebarSessionEntry[]) => groupEntriesByRecency(entries, NOW, MONDAY)

const dividerKeys = (rows: ReturnType<typeof groupEntriesByRecency>): string[] =>
  rows.flatMap(row => (row.kind === 'divider' ? [row.key] : []))

describe('groupEntriesByRecency', () => {
  it('labels every populated group, including the first, with the required calendar taxonomy', () => {
    const rows = group([
      entry(session('today', { last_active: at(2026, 5, 18) })),
      entry(session('yesterday', { last_active: at(2026, 5, 17) })),
      entry(session('tuesday', { last_active: at(2026, 5, 16) })),
      entry(session('last-week', { last_active: at(2026, 5, 12) })),
      entry(session('third-week', { last_active: at(2026, 5, 3) })),
      entry(session('fourth-week', { last_active: at(2026, 4, 28) })),
      entry(session('month', { last_active: at(2026, 3, 20) }))
    ])

    expect(rows[0]).toMatchObject({ key: 'day:2026-06-18', kind: 'divider' })
    expect(dividerKeys(rows)).toEqual([
      'day:2026-06-18',
      'day:2026-06-17',
      'day:2026-06-16',
      'week:2026-06-08',
      'week:2026-06-01',
      'week:2026-05-25',
      'month:2026-04'
    ])
  })

  it('emits a Today divider even when every session belongs to today', () => {
    const rows = group([
      entry(session('a', { last_active: at(2026, 5, 18, 11) })),
      entry(session('b', { last_active: at(2026, 5, 18, 10) }))
    ])

    expect(dividerKeys(rows)).toEqual(['day:2026-06-18'])
  })

  it('keeps branch children in their parent calendar group', () => {
    const parent = entry(session('parent', { last_active: at(2026, 5, 18) }))
    const child = entry(session('child', { last_active: at(2024, 0, 1), parent_session_id: 'parent' }), '└─ ')
    const rows = group([parent, child])

    expect(rows).toEqual([
      expect.objectContaining({ key: 'day:2026-06-18', kind: 'divider' }),
      { entry: parent, kind: 'session' },
      { entry: child, kind: 'session' }
    ])
  })

  it('gives repeated non-monotonic segments stable identities while sharing their collapse key', () => {
    const rows = group([
      entry(session('today-a', { last_active: at(2026, 5, 18, 11) })),
      entry(session('old', { last_active: at(2026, 3, 20) })),
      entry(session('today-b', { last_active: at(2026, 5, 18, 9) }))
    ])

    const todayRows = rows.filter(
      (row): row is Extract<SidebarListRow, { kind: 'divider' }> =>
        row.kind === 'divider' && row.key === 'day:2026-06-18'
    )

    expect(dividerKeys(rows)).toEqual(['day:2026-06-18', 'month:2026-04', 'day:2026-06-18'])
    expect(new Set(todayRows.map(row => row.rowKey))).toHaveLength(2)
  })

  it('falls back to started_at when last_active is absent', () => {
    const rows = group([entry(session('fallback', { last_active: 0, started_at: at(2026, 5, 17) }))])

    expect(dividerKeys(rows)).toEqual(['day:2026-06-17'])
  })
})

describe('toSessionRows', () => {
  it('wraps entries as session rows with no date metadata', () => {
    const entries = [entry(session('a')), entry(session('b'), '└─ ')]

    expect(toSessionRows(entries)).toEqual([
      { entry: entries[0], kind: 'session' },
      { entry: entries[1], kind: 'session' }
    ])
  })
})

describe('filterCollapsedDateGroupRows', () => {
  it('keeps every divider while hiding sessions in every collapsed segment', () => {
    const todayA = entry(session('today-a'))
    const old = entry(session('old'))
    const todayB = entry(session('today-b'))

    const rows: SidebarListRow[] = [
      { key: 'today', kind: 'divider', label: 'Today', rowKey: 'today:a' },
      { entry: todayA, kind: 'session' },
      { key: 'old', kind: 'divider', label: 'Old' },
      { entry: old, kind: 'session' },
      { key: 'today', kind: 'divider', label: 'Today', rowKey: 'today:b' },
      { entry: todayB, kind: 'session' }
    ]

    expect(filterCollapsedDateGroupRows(rows, new Set(['today']))).toEqual([rows[0], rows[2], rows[3], rows[4]])
  })

  it('does not alter rows when no group is collapsed', () => {
    const rows: SidebarListRow[] = [
      { key: 'today', kind: 'divider', label: 'Today' },
      { entry: entry(session('today')), kind: 'session' }
    ]

    expect(filterCollapsedDateGroupRows(rows, new Set())).toEqual(rows)
  })
})
