import { describe, expect, it } from 'vitest'

import type { SessionDotState } from '@/store/session-dot-state'
import type { SessionInfo } from '@/types/hermes'

import { makeSessionInfo } from '../test/session-info'

import type { SidebarSessionEntry } from './session-branch-tree'
import {
  groupEntriesByRecency,
  groupEntriesByStatus,
  hideCollapsedGroupRows,
  idleLane,
  listGroupDefaultOpen,
  toSessionRows
} from './session-date-groups'

const session = (id: string, overrides: Partial<SessionInfo> = {}): SessionInfo =>
  makeSessionInfo({ id, message_count: 1, source: 'cli', title: id, ...overrides })

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
  it('limits complete Updated groups without splitting bursts or branch clusters', () => {
    const burst = Array.from({ length: 11 }, (_, i) =>
      entry(session(`burst-${i}`, { last_active: at(2026, 5, 18, 11) - i * 120 }))
    )

    const entries = [
      ...burst,
      entry(session('yesterday', { last_active: at(2026, 5, 17, 15) })),
      entry(session('branch', { last_active: at(2026, 4, 1) }), '└'),
      entry(session('older', { last_active: at(2026, 5, 16, 15) }))
    ]

    const full = group(entries)
    const dividers = full.flatMap((row, index) => (row.kind === 'divider' ? [index] : []))
    const limited = groupEntriesByRecency(entries, NOW, MONDAY, 2)

    expect(limited).toEqual(full.slice(0, dividers[1]))
    expect(limited.at(-1)).toMatchObject({ entry: { session: { id: 'branch' } } })
    expect(groupEntriesByRecency(burst, NOW, MONDAY, 2)).toEqual(group(burst))
    expect(groupEntriesByRecency([], NOW, MONDAY, 2)).toEqual([])
  })

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

describe('hideCollapsedGroupRows', () => {
  it('returns the same array when every group is open', () => {
    const rows = [
      { entry: entry(session('a')), kind: 'session' as const },
      { key: 'yesterday', kind: 'divider' as const, label: 'Yesterday' },
      { entry: entry(session('b')), kind: 'session' as const }
    ]

    expect(hideCollapsedGroupRows(rows, () => true)).toBe(rows)
  })

  it('keeps the divider and drops sessions under a closed group', () => {
    const rows = [
      { entry: entry(session('head')), kind: 'session' as const },
      { key: 'yesterday', kind: 'divider' as const, label: 'Yesterday' },
      { entry: entry(session('y1')), kind: 'session' as const },
      { entry: entry(session('y2'), '└─ '), kind: 'session' as const },
      { key: 'last-week', kind: 'divider' as const, label: 'Last week' },
      { entry: entry(session('lw')), kind: 'session' as const }
    ]

    const visible = hideCollapsedGroupRows(rows, key => key !== 'yesterday')

    expect(visible).toEqual([rows[0], rows[1], rows[4], rows[5]])
  })

  it('never hides the unlabelled head above the first divider', () => {
    const rows = [
      { entry: entry(session('a')), kind: 'session' as const },
      { entry: entry(session('b')), kind: 'session' as const },
      { key: 'yesterday', kind: 'divider' as const, label: 'Yesterday' },
      { entry: entry(session('c')), kind: 'session' as const }
    ]

    const visible = hideCollapsedGroupRows(rows, () => false)

    expect(visible.map(row => (row.kind === 'session' ? row.entry.session.id : row.key))).toEqual([
      'a',
      'b',
      'yesterday'
    ])
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

describe('groupEntriesByStatus', () => {
  const LABELS = {
    attention: 'Needs attention',
    idle: 'Idle today',
    idleDays: 'Idle 3–6 days',
    idleRecent: 'Idle 1–2 days',
    idleWeek: 'Idle 7+ days',
    working: 'Working'
  }

  // Lane membership by state is tested on sessions touched today; age has its
  // own cases below.
  const session = (id: string, overrides: Partial<SessionInfo> = {}): SessionInfo =>
    makeSessionInfo({ id, last_active: at(2026, 5, 18, 10), message_count: 1, source: 'cli', title: id, ...overrides })

  const byStatus = (entries: SidebarSessionEntry[], states: Record<string, SessionDotState>, nowMs = NOW) =>
    groupEntriesByStatus(entries, e => states[e.session.id] ?? 'idle', LABELS, nowMs)

  const daysAgo = (id: string, days: number, hour = 10) =>
    entry(session(id, { last_active: at(2026, 5, 18 - days, hour) }))

  const layout = (rows: ReturnType<typeof byStatus>) =>
    rows.map(row => (row.kind === 'divider' ? row.key : row.entry.session.id))

  it('splits on whose move it is: needs-input and unread need attention, drafts are idle', () => {
    const states: Record<string, SessionDotState> = {
      bg: 'background',
      draft: 'draft',
      idle: 'idle',
      input: 'needs-input',
      quiet: 'stalled',
      run: 'working',
      unread: 'unread'
    }

    const rows = byStatus(
      Object.keys(states).map(id => entry(session(id))),
      states
    )

    expect(layout(rows)).toEqual([
      'status:attention',
      'input',
      'unread',
      'status:working',
      'bg',
      'quiet',
      'run',
      'status:idle',
      'draft',
      'idle'
    ])
    expect(rows[0]).toMatchObject({ count: 2, label: 'Needs attention' })
  })

  it('leads each lane with the louder state but otherwise keeps the incoming order', () => {
    const states: Record<string, SessionDotState> = {
      'idle-a': 'idle',
      'draft-a': 'draft',
      'unread-a': 'unread',
      'input-a': 'needs-input',
      'unread-b': 'unread',
      'input-b': 'needs-input'
    }

    const rows = byStatus(
      Object.keys(states).map(id => entry(session(id))),
      states
    )

    expect(layout(rows)).toEqual([
      'status:attention',
      'input-a',
      'input-b',
      'unread-a',
      'unread-b',
      'status:idle',
      'draft-a',
      'idle-a'
    ])
  })

  it('keeps branch children with their parent and drops empty lanes', () => {
    const rows = byStatus([entry(session('parent')), entry(session('child'), '└'), entry(session('other'))], {
      child: 'working',
      parent: 'unread'
    })

    expect(layout(rows)).toEqual(['status:attention', 'parent', 'child', 'status:idle', 'other'])
    expect(byStatus([], {})).toEqual([])
  })

  it('splits idle sessions by days untouched: today, 1–2, 3–6, 7+', () => {
    const rows = byStatus(
      [
        daysAgo('d0', 0),
        daysAgo('d1', 1),
        daysAgo('d2', 2),
        daysAgo('d3', 3),
        daysAgo('d6', 6),
        daysAgo('d7', 7),
        daysAgo('d40', 40)
      ],
      {}
    )

    expect(layout(rows)).toEqual([
      'status:idle',
      'd0',
      'status:idleRecent',
      'd1',
      'd2',
      'status:idleDays',
      'd3',
      'd6',
      'status:idleWeek',
      'd7',
      'd40'
    ])
    expect(rows.find(row => row.kind === 'divider' && row.key === 'status:idleWeek')).toMatchObject({
      count: 2,
      label: 'Idle 7+ days'
    })
  })

  it('never ages a session that needs the user or is still running', () => {
    const rows = byStatus([daysAgo('old-input', 30), daysAgo('old-unread', 9), daysAgo('old-run', 12)], {
      'old-input': 'needs-input',
      'old-run': 'background',
      'old-unread': 'unread'
    })

    expect(layout(rows)).toEqual(['status:attention', 'old-input', 'old-unread', 'status:working', 'old-run'])
  })

  it('ages a branch cluster by its newest member and keeps it whole', () => {
    const rows = byStatus([daysAgo('old-parent', 10), { ...daysAgo('fresh-branch', 0), branchStem: '└' }], {})

    expect(layout(rows)).toEqual(['status:idle', 'old-parent', 'fresh-branch'])
    expect(rows[0]).toMatchObject({ count: 1 })
  })

  it('moves sessions to the next lane at the 4 AM rollover, not midnight', () => {
    const lateLastNight = daysAgo('late', 1, 23)
    const beforeRollover = new Date(2026, 5, 18, 3, 30).getTime()
    const afterRollover = new Date(2026, 5, 18, 4, 30).getTime()

    expect(layout(byStatus([lateLastNight], {}, beforeRollover))).toEqual(['status:idle', 'late'])
    expect(layout(byStatus([lateLastNight], {}, afterRollover))).toEqual(['status:idleRecent', 'late'])
  })
})

describe('idleLane', () => {
  it('treats a future timestamp as today and a missing one as the oldest lane', () => {
    expect(idleLane(NOW + 3_600_000, NOW)).toBe('idle')
    expect(idleLane(0, NOW)).toBe('idleWeek')
  })
})

describe('listGroupDefaultOpen', () => {
  it('folds only the 7+ day idle lane by default', () => {
    expect(listGroupDefaultOpen('status:idleWeek')).toBe(false)
    expect(
      ['status:attention', 'status:working', 'status:idle', 'status:idleDays', 'today'].every(listGroupDefaultOpen)
    ).toBe(true)
  })
})
