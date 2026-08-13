import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import type { SidebarSessionEntry } from './session-branch-tree'
import { groupEntriesByRecency, toSessionRows } from './session-date-groups'

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

const entry = (s: SessionInfo, branchStem?: string): SidebarSessionEntry =>
  branchStem ? { branchStem, session: s } : { session: s }

// Fixed "now": Thursday 18 Jun 2026, local noon (15 Jun 2026 is a Monday).
// All tests pin a Monday week start so calendar boundaries are deterministic.
const NOW = new Date(2026, 5, 18, 12, 0, 0).getTime()
const MONDAY = 1

const at = (year: number, month: number, day: number, hour = 10, minute = 0): number =>
  Math.floor(new Date(year, month, day, hour, minute, 0).getTime() / 1000)

const group = (entries: SidebarSessionEntry[], nowMs = NOW) => groupEntriesByRecency(entries, nowMs, MONDAY)

const dividerKeys = (rows: ReturnType<typeof groupEntriesByRecency>): string[] =>
  rows.flatMap(row => (row.kind === 'divider' ? [row.key] : []))

describe('groupEntriesByRecency', () => {
  it('cuts the head after the most recent handful, then divides by coarse ranges', () => {
    // The morning run (30m/30m/4h/30m gaps, then a 14h silence) is the
    // unlabelled head; each older group gets one divider, coarsening with age.
    const rows = group([
      entry(session('a', { last_active: at(2026, 5, 18, 11) })),
      entry(session('b', { last_active: at(2026, 5, 18, 10, 30) })),
      entry(session('c', { last_active: at(2026, 5, 18, 10) })),
      entry(session('d', { last_active: at(2026, 5, 18, 6) })),
      entry(session('e', { last_active: at(2026, 5, 18, 5, 30) })),
      entry(session('f', { last_active: at(2026, 5, 17, 15) })), // yesterday
      entry(session('g', { last_active: at(2026, 5, 16, 15) })), // Tue this week
      entry(session('h', { last_active: at(2026, 5, 14) })), // Sun last week
      entry(session('i', { last_active: at(2026, 5, 3) })), // earlier in June
      entry(session('j', { last_active: at(2026, 4, 28) })), // May
      entry(session('k', { last_active: at(2025, 11, 3) })) // December 2025
    ])

    expect(rows.slice(0, 5).every(row => row.kind === 'session')).toBe(true)
    expect(rows[5]).toMatchObject({ key: 'yesterday', kind: 'divider' })
    expect(dividerKeys(rows)).toEqual(['yesterday', 'this-week', 'last-week', 'this-month', 'm-2026-4', 'my-2025-11'])
  })

  it('labels the rest of the current day "earlier today" past a real break', () => {
    // Five rapid-fire sessions, a ~4h pause, then more of the same day.
    const rows = group([
      entry(session('a', { last_active: at(2026, 5, 18, 11) })),
      entry(session('b', { last_active: at(2026, 5, 18, 10, 58) })),
      entry(session('c', { last_active: at(2026, 5, 18, 10, 56) })),
      entry(session('d', { last_active: at(2026, 5, 18, 10, 54) })),
      entry(session('e', { last_active: at(2026, 5, 18, 10, 52) })),
      entry(session('f', { last_active: at(2026, 5, 18, 7) })),
      entry(session('g', { last_active: at(2026, 5, 18, 6, 58) }))
    ])

    expect(rows.findIndex(row => row.kind === 'divider')).toBe(5)
    expect(dividerKeys(rows)).toEqual(['today'])
  })

  it('never slices a rapid-fire burst mid-run', () => {
    // Eleven sessions two minutes apart: no gap qualifies as a break, so the
    // whole burst stays in the head and the divider lands after it.
    const burst = Array.from({ length: 11 }, (_, i) =>
      entry(session(`s${i}`, { last_active: at(2026, 5, 18, 11) - i * 120 }))
    )

    const rows = group([...burst, entry(session('old', { last_active: at(2026, 5, 17, 15) }))])

    expect(rows.findIndex(row => row.kind === 'divider')).toBe(11)
    expect(dividerKeys(rows)).toEqual(['yesterday'])
  })

  it('chains the head run across midnight', () => {
    // Viewed at 00:58: tonight plus last evening is one run; yesterday's
    // afternoon (a different nominal day) opens the labelled groups.
    const smallHours = new Date(2026, 5, 19, 0, 58).getTime()

    const rows = group(
      [
        entry(session('a', { last_active: at(2026, 5, 19, 0, 30) })),
        entry(session('b', { last_active: at(2026, 5, 18, 23, 50) })),
        entry(session('c', { last_active: at(2026, 5, 18, 23, 20) })),
        entry(session('d', { last_active: at(2026, 5, 17, 20) }))
      ],
      smallHours
    )

    expect(rows.findIndex(row => row.kind === 'divider')).toBe(3)
    expect(dividerKeys(rows)).toEqual(['yesterday'])
  })

  it('dissolves a stale head into its own calendar group (fuzzy merge)', () => {
    // Newest session is 6 days old and the rows below it share its "last week"
    // bucket: cutting there would strand near-identical neighbours around a
    // divider, so no head is kept and the whole bucket leads unlabelled.
    const rows = group([
      entry(session('a', { last_active: at(2026, 5, 12) })),
      entry(session('b', { last_active: at(2026, 5, 11, 15) })),
      entry(session('c', { last_active: at(2026, 5, 11, 10) })),
      entry(session('d', { last_active: at(2026, 5, 3) })),
      entry(session('e', { last_active: at(2026, 4, 20) }))
    ])

    expect(rows.slice(0, 3).every(row => row.kind === 'session')).toBe(true)
    expect(dividerKeys(rows)).toEqual(['this-month', 'm-2026-4'])
  })

  it('keeps an isolated newest session as the head when its bucket differs', () => {
    const rows = group([
      entry(session('a', { last_active: at(2026, 5, 18, 9) })),
      entry(session('b', { last_active: at(2026, 5, 17, 15) })),
      entry(session('c', { last_active: at(2026, 5, 3) }))
    ])

    expect(rows.findIndex(row => row.kind === 'divider')).toBe(1)
    expect(dividerKeys(rows)).toEqual(['yesterday', 'this-month'])
  })

  it('emits no dividers when everything is one unbroken run', () => {
    const rows = group([
      entry(session('a', { last_active: at(2026, 5, 18, 11) })),
      entry(session('b', { last_active: at(2026, 5, 18, 10, 50) })),
      entry(session('c', { last_active: at(2026, 5, 18, 10, 40) }))
    ])

    expect(rows.every(row => row.kind === 'session')).toBe(true)
  })

  it('collapses a big gap straight to the next month/year (empty ranges omitted)', () => {
    const rows = group([
      entry(session('t', { last_active: at(2026, 5, 18, 11) })),
      entry(session('t2', { last_active: at(2026, 5, 18, 10, 30) })),
      entry(session('j1', { last_active: at(2026, 0, 5) })),
      entry(session('j2', { last_active: at(2026, 0, 3) })),
      entry(session('old', { last_active: at(2024, 2, 9) }))
    ])

    expect(dividerKeys(rows)).toEqual(['m-2026-0', 'my-2024-2'])
  })

  it('never labels the first rendered group, even when it is not recent', () => {
    // Newest session is weeks old and alone in its month: it opens the list
    // unlabelled; only the transitions below it are marked.
    const rows = group([
      entry(session('a', { last_active: at(2026, 4, 20) })),
      entry(session('b', { last_active: at(2026, 2, 3) })),
      entry(session('c', { last_active: at(2025, 11, 3) }))
    ])

    expect(rows[0]).toMatchObject({ kind: 'session' })
    expect(dividerKeys(rows)).toEqual(['m-2026-2', 'my-2025-11'])
  })

  it('keeps branch children in their parent cluster without opening a new bucket', () => {
    const parent = session('parent', { last_active: at(2026, 5, 18, 11) })
    const child = session('child', { last_active: at(2024, 0, 1), parent_session_id: 'parent' })

    const rows = group([entry(parent), entry(child, '└─ ')])

    expect(rows).toEqual([
      { entry: entry(parent), kind: 'session' },
      { entry: entry(child, '└─ '), kind: 'session' }
    ])
  })

  it('never emits a divider twice under a non-monotonic order', () => {
    const rows = group([
      entry(session('a', { last_active: at(2026, 5, 16) })), // head run
      entry(session('b', { last_active: at(2026, 4, 5) })), // May — divider
      entry(session('c', { last_active: at(2026, 5, 16) })) // head again — no repeat
    ])

    expect(dividerKeys(rows)).toEqual(['m-2026-4'])
  })

  it('falls back to started_at when last_active is missing', () => {
    const rows = group([
      entry(session('head', { last_active: at(2026, 5, 18, 11) })),
      entry(session('s', { last_active: 0, started_at: at(2026, 5, 10) }))
    ])

    expect(dividerKeys(rows)).toEqual(['last-week'])
  })
})

describe('toSessionRows', () => {
  it('wraps entries as session rows with no dividers', () => {
    const entries = [entry(session('a')), entry(session('b'), '└─ ')]

    expect(toSessionRows(entries)).toEqual([
      { entry: entries[0], kind: 'session' },
      { entry: entries[1], kind: 'session' }
    ])
  })
})
