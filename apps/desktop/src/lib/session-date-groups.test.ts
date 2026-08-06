import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import type { SidebarSessionEntry } from './session-branch-tree'
import {
  groupEntriesByRecency,
  sessionDateGroupKeys,
  toSessionRows,
  visibleSessionDateGroupRows
} from './session-date-groups'

const session = (id: string, overrides: Partial<SessionInfo> = {}): SessionInfo =>
  ({
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'cli',
    started_at: 0,
    title: id,
    tool_call_count: 0,
    ...overrides
  }) as SessionInfo

const entry = (value: SessionInfo, branchStem?: string): SidebarSessionEntry =>
  branchStem ? { branchStem, session: value } : { session: value }

// Friday 31 July 2026, local noon; French weeks start on Monday.
const NOW = new Date(2026, 6, 31, 12, 0, 0).getTime()
const MONDAY = 1

const at = (year: number, month: number, day: number, hour = 10): number =>
  Math.floor(new Date(year, month, day, hour, 0, 0).getTime() / 1000)

const group = (entries: SidebarSessionEntry[]) => groupEntriesByRecency(entries, NOW, MONDAY)

const dividerKeys = (rows: ReturnType<typeof groupEntriesByRecency>): string[] =>
  rows.flatMap(row => (row.kind === 'divider' ? [row.key] : []))

const sessionIds = (rows: ReturnType<typeof groupEntriesByRecency>): string[] =>
  rows.flatMap(row => (row.kind === 'session' ? [row.entry.session.id] : []))

describe('groupEntriesByRecency', () => {
  it('labels every group, including the first one, with the required taxonomy', () => {
    const rows = group([
      entry(session('today', { last_active: at(2026, 6, 31) })),
      entry(session('yesterday', { last_active: at(2026, 6, 30) })),
      entry(session('wednesday', { last_active: at(2026, 6, 29) })),
      entry(session('last-week', { last_active: at(2026, 6, 25) })),
      entry(session('third-week', { last_active: at(2026, 6, 15) })),
      entry(session('fourth-week', { last_active: at(2026, 5, 30) })),
      entry(session('month', { last_active: at(2026, 5, 20) }))
    ])

    expect(rows[0]).toMatchObject({ key: 'day:2026-07-31', kind: 'divider' })
    expect(dividerKeys(rows)).toEqual([
      'day:2026-07-31',
      'day:2026-07-30',
      'day:2026-07-29',
      'week:2026-07-20',
      'week:2026-07-13',
      'week:2026-06-29',
      'month:2026-06'
    ])
    expect(dividerKeys(rows)).not.toContain('__recent__')
  })

  it('emits exactly one divider for one populated group', () => {
    const rows = group([
      entry(session('a', { last_active: at(2026, 6, 31, 11) })),
      entry(session('b', { last_active: at(2026, 6, 31, 10) }))
    ])

    expect(dividerKeys(rows)).toEqual(['day:2026-07-31'])
    expect(sessionIds(rows)).toEqual(['a', 'b'])
  })

  it('emits one Today divider for current and clock-skewed future activity', () => {
    const rows = group([
      entry(session('today', { last_active: at(2026, 6, 31, 11) })),
      entry(session('future-a', { last_active: at(2026, 7, 1) })),
      entry(session('future-b', { last_active: at(2026, 7, 15) }))
    ])

    expect(dividerKeys(rows)).toEqual(['day:2026-07-31'])
    expect(sessionIds(rows)).toEqual(['today', 'future-a', 'future-b'])
  })

  it('omits empty calendar ranges', () => {
    const rows = group([
      entry(session('today', { last_active: at(2026, 6, 31) })),
      entry(session('old', { last_active: at(2025, 11, 3) }))
    ])

    expect(dividerKeys(rows)).toEqual(['day:2026-07-31', 'month:2025-12'])
  })

  it('keeps branch children in their parent calendar group', () => {
    const parent = entry(session('parent', { last_active: at(2026, 6, 31) }))
    const child = entry(session('child', { last_active: at(2024, 0, 1), parent_session_id: 'parent' }), '└─ ')
    const rows = group([parent, child])

    expect(rows).toEqual([
      expect.objectContaining({ key: 'day:2026-07-31', kind: 'divider' }),
      { entry: parent, groupKey: 'day:2026-07-31', kind: 'session' },
      { entry: child, groupKey: 'day:2026-07-31', kind: 'session' }
    ])
  })

  it('preserves non-monotonic input order with stable segments sharing one business key', () => {
    const rows = group([
      entry(session('today-a', { last_active: at(2026, 6, 31, 11) })),
      entry(session('old', { last_active: at(2026, 5, 20) })),
      entry(session('today-b', { last_active: at(2026, 6, 31, 9) }))
    ])

    const todayDividerRowKeys = rows.flatMap(row =>
      row.kind === 'divider' && row.key === 'day:2026-07-31' ? [row.rowKey] : []
    )

    expect(dividerKeys(rows)).toEqual(['day:2026-07-31', 'month:2026-06', 'day:2026-07-31'])
    expect(sessionIds(rows)).toEqual(['today-a', 'old', 'today-b'])
    expect(new Set(todayDividerRowKeys)).toHaveLength(2)
    expect(sessionDateGroupKeys(rows)).toEqual(['day:2026-07-31', 'month:2026-06'])
    expect(sessionIds(visibleSessionDateGroupRows(rows, new Set(['day:2026-07-31'])))).toEqual(['old'])
  })

  it('falls back to started_at when last_active is absent', () => {
    const rows = group([entry(session('fallback', { last_active: 0, started_at: at(2026, 6, 30) }))])

    expect(dividerKeys(rows)).toEqual(['day:2026-07-30'])
  })

  it('does not change a group when only title-like presentation data changes', () => {
    const original = group([entry(session('same', { last_active: at(2026, 6, 30), title: 'Before' }))])
    const renamed = group([entry(session('same', { last_active: at(2026, 6, 30), title: 'After' }))])

    expect(dividerKeys(original)).toEqual(dividerKeys(renamed))
  })

  it('changes a group only when the authoritative activity timestamp changes', () => {
    const before = group([entry(session('same', { last_active: at(2026, 6, 30) }))])
    const after = group([entry(session('same', { last_active: at(2026, 6, 31) }))])

    expect(dividerKeys(before)).toEqual(['day:2026-07-30'])
    expect(dividerKeys(after)).toEqual(['day:2026-07-31'])
  })
})

describe('visibleSessionDateGroupRows', () => {
  const rows = group([
    entry(session('today-a', { last_active: at(2026, 6, 31) })),
    entry(session('today-b', { last_active: at(2026, 6, 31, 9) })),
    entry(session('yesterday', { last_active: at(2026, 6, 30) }))
  ])

  it('removes collapsed conversation rows before virtual measurement but keeps dividers', () => {
    const visible = visibleSessionDateGroupRows(rows, new Set(['day:2026-07-31']))

    expect(dividerKeys(visible)).toEqual(['day:2026-07-31', 'day:2026-07-30'])
    expect(sessionIds(visible)).toEqual(['yesterday'])
    expect(visible).toHaveLength(3)
  })

  it('restores every row when all known groups are expanded', () => {
    expect(sessionDateGroupKeys(rows)).toEqual(['day:2026-07-31', 'day:2026-07-30'])
    expect(visibleSessionDateGroupRows(rows, new Set())).toEqual(rows)
  })
})

describe('toSessionRows', () => {
  it('keeps the existing ungrouped path free of temporal metadata', () => {
    const entries = [entry(session('a')), entry(session('b'), '└─ ')]

    expect(toSessionRows(entries)).toEqual([
      { entry: entries[0], kind: 'session' },
      { entry: entries[1], kind: 'session' }
    ])
  })
})
