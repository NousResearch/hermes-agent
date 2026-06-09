// Ordering helpers for sidebar recents.
//
// The sidebar's unpinned session lists are for finding recent work, so they must
// stay activity-ordered. Manual ordering belongs to the Pinned section only.

// Merge several session lists into one newest-first list, deduped by id (the
// freshest timestamp wins on collision), capped at `n`. Powers the "Recent N"
// quick-access section. Callers can pass `isExcluded` for rows that have their
// own dedicated surface (for example scheduled cron runs).
export function topRecentSessions<T>(
  lists: T[][],
  getId: (item: T) => string,
  getTime: (item: T) => number,
  n: number,
  isExcluded?: (item: T) => boolean
): T[] {
  if (n <= 0) {
    return []
  }

  const best = new Map<string, T>()

  for (const list of lists) {
    for (const item of list) {
      if (isExcluded?.(item)) {
        continue
      }

      const id = getId(item)
      const existing = best.get(id)

      if (!existing || getTime(item) > getTime(existing)) {
        best.set(id, item)
      }
    }
  }

  return Array.from(best.values()).sort((a, b) => getTime(b) - getTime(a)).slice(0, n)
}

// Ids of the most-recently-active open session per work lineage.
//
// A "Live" badge is a work-state marker, not a single global recency marker:
// every unfinished lineage gets exactly one head, and archived rows never count
// as live. Legacy rows without lineage metadata are treated as one-session work
// items keyed by their own id.
export function workHeadSessionIds<T>(
  lists: T[][],
  getId: (item: T) => string,
  getLineageRoot: (item: T) => null | string,
  getTime: (item: T) => number,
  isArchived: (item: T) => boolean
): Set<string> {
  const bestPerLineage = new Map<string, T>()

  for (const list of lists) {
    for (const item of list) {
      if (isArchived(item)) {
        continue
      }

      const id = getId(item)
      const lineage = getLineageRoot(item) ?? id
      const existing = bestPerLineage.get(lineage)

      if (!existing || getTime(item) > getTime(existing)) {
        bestPerLineage.set(lineage, item)
      }
    }
  }

  return new Set(Array.from(bestPerLineage.values(), getId))
}
