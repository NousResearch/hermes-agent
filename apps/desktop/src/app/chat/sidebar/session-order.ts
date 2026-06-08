// Ordering helpers for sidebar recents.
//
// The sidebar's unpinned session lists are for finding recent work, so they must
// stay activity-ordered. Manual ordering belongs to the Pinned section only.

// Merge several session lists into one newest-first list, deduped by id (the
// freshest timestamp wins on collision), capped at `n`. Powers the "Recent N"
// quick-access section, which spans local, external-source, and cron sessions —
// the same session can appear in more than one source list, so dedup matters.
export function topRecentSessions<T>(
  lists: T[][],
  getId: (item: T) => string,
  getTime: (item: T) => number,
  n: number
): T[] {
  if (n <= 0) {
    return []
  }

  const best = new Map<string, T>()

  for (const list of lists) {
    for (const item of list) {
      const id = getId(item)
      const existing = best.get(id)

      if (!existing || getTime(item) > getTime(existing)) {
        best.set(id, item)
      }
    }
  }

  return Array.from(best.values()).sort((a, b) => getTime(b) - getTime(a)).slice(0, n)
}

// Id of the single most-recently-active session across the given lists, or null
// when there are none. Used to badge the newest session 'Live' so it reads as
// the freshest at a glance.
export function freshestSessionId<T>(
  lists: T[][],
  getId: (item: T) => string,
  getTime: (item: T) => number
): null | string {
  const [top] = topRecentSessions(lists, getId, getTime, 1)

  return top ? getId(top) : null
}
