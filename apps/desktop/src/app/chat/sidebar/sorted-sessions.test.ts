import { describe, it, expect } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { sessionRecency } from './projects/workspace-groups'

/**
 * The recency sort in the sidebar (`index.tsx` sortedSessions) orders by
 * `sessionRecency` with no tie-breaker. Two rows with the same timestamp
 * (cron runs created in the same second, batch imports, rows whose
 * `last_active` is absent so `started_at` decides and collides) compare as
 * equal, and Array.prototype.sort is NOT guaranteed stable for equal
 * elements across engine versions — equal keys may swap between refreshes,
 * which repaints as the sidebar "flashing"/reordering rows.
 *
 * The fix: break ties on a stable per-row key (id) so equal-recency rows
 * keep one order forever.
 */
function sortLikeSidebar<T extends { id: string }>(rows: readonly T[]): T[] {
  return [...rows].sort(
    (a, b) => sessionRecency(b as never) - sessionRecency(a as never) || a.id.localeCompare(b.id)
  )
}

function row(id: string, ts: number): SessionInfo {
  return { id, started_at: ts } as unknown as SessionInfo
}

describe('sidebar recency sort tie-breaking', () => {
  it('keeps equal-recency rows in a deterministic order across repeated sorts', () => {
    const ts = 1_791_336_000 // same second for all three — cron-batch shape
    const input = [row('c', ts), row('a', ts), row('b', ts)]

    const first = sortLikeSidebar(input)
    const second = sortLikeSidebar([...input].reverse())

    // Without a tie-breaker these two sorts may differ (engine-dependent);
    // with one, both orders are identical and stable.
    expect(first.map(r => r.id)).toEqual(second.map(r => r.id))
    expect(first.map(r => r.id)).toEqual(['a', 'b', 'c'])
  })

  it('still orders by recency first — newer rows above older ones', () => {
    const sorted = sortLikeSidebar([row('old', 100), row('new', 200), row('mid', 150)])
    expect(sorted.map(r => r.id)).toEqual(['new', 'mid', 'old'])
  })
})
