import type { SidebarListRow } from '@/lib/session-date-groups'

/**
 * Fold ids absent from the persisted order back in, on the side they belong.
 *
 * Callers hand us a recency-sorted `currentIds`, so an id's position relative
 * to the newest id the saved order still knows about says which side it goes:
 * ahead of it is genuinely newer (a session created since the last reconcile,
 * which must not sink), behind it is an older page that just loaded. Hoisting
 * every unknown id to the top treated those two cases the same, so one "load
 * more" dropped 50 of the oldest sessions above the hand-ordered list.
 */
function mergeFreshByPosition(currentIds: string[], keptIds: string[]): string[] {
  const kept = new Set(keptIds)
  const firstKept = currentIds.findIndex(id => kept.has(id))
  const newer: string[] = []
  const older: string[] = []

  currentIds.forEach((id, index) => {
    if (kept.has(id)) {
      return
    }

    if (firstKept >= 0 && index < firstKept) {
      newer.push(id)
    } else {
      older.push(id)
    }
  })

  return [...newer, ...keptIds, ...older]
}

/** Ids still present in the persisted order, with new ids folded in by position. */
export function reconcileFreshFirst(currentIds: string[], orderIds: string[]): string[] {
  const current = new Set(currentIds)

  // Dedupe both inputs: a corrupted persisted order (same id twice) must not
  // self-perpetuate through reconcile, and duplicate live ids (e.g. the same
  // repo surfacing under several projects) must not be written back into the
  // saved order — either one paints as duplicate headers (#73314).
  return mergeFreshByPosition([...new Set(currentIds)], [...new Set(orderIds.filter(id => current.has(id)))])
}

export function resolveManualSessionOrderIds(currentIds: string[], orderIds: string[], manual: boolean): string[] {
  if (!manual || !currentIds.length || !orderIds.length) {
    return []
  }

  const current = new Set(currentIds)
  const retained = orderIds.filter(id => current.has(id))

  if (!retained.length) {
    return []
  }

  return reconcileFreshFirst(currentIds, orderIds)
}

/**
 * Reorder `items` by `orderIds`. Items missing from the order are folded in by
 * position, same rule as {@link reconcileFreshFirst}: a brand-new session ahead
 * of everything the order knows stays on top, an older page that just loaded
 * sinks below the hand-picked list instead of jumping it.
 */
export function orderByIds<T>(items: T[], getId: (item: T) => string, orderIds: string[]): T[] {
  if (!orderIds.length) {
    return items
  }

  const byId = new Map(items.map(item => [getId(item), item]))
  const seen = new Set<string>()
  const ordered: T[] = []

  for (const id of orderIds) {
    const item = byId.get(id)

    // Guard against duplicates in the persisted order: pushing the same item
    // twice renders the row/header twice (e.g. two identical repo headers
    // under one project).
    if (item && !seen.has(id)) {
      ordered.push(item)
      seen.add(id)
    }
  }

  if (seen.size === items.length) {
    return ordered
  }

  const firstOrdered = items.findIndex(item => seen.has(getId(item)))
  const newer: T[] = []
  const older: T[] = []

  items.forEach((item, index) => {
    const itemId = getId(item)

    // `seen` doubles as the duplicate guard for live items: two rows carrying
    // the same id (e.g. one repo surfacing under several projects) must render
    // once, not once per occurrence (#73314).
    if (seen.has(itemId)) {
      return
    }

    seen.add(itemId)

    if (firstOrdered >= 0 && index < firstOrdered) {
      newer.push(item)
    } else {
      older.push(item)
    }
  })

  return [...newer, ...ordered, ...older]
}

/**
 * Apply the active sort key (as an id order) to a set of session rows, leaving
 * them in the order they came in when nothing is ranked. Grouped views call
 * this on their own lane so a sort key reaches rows the flat list never renders.
 */
export function rankSessions<T extends { id: string }>(sessions: T[], rankIds?: string[]): T[] {
  return rankIds?.length ? orderByIds(sessions, session => session.id, rankIds) : sessions
}

/** Reconcile a persisted order against the live id set. */
export function reconcileOrderIds(currentIds: string[], orderIds: string[]): string[] {
  if (!currentIds.length) {
    return []
  }

  if (!orderIds.length) {
    // Still dedupe: persisting duplicate live ids here is what seeded the
    // #73314 feedback loop in the first place.
    return [...new Set(currentIds)]
  }

  return reconcileFreshFirst(currentIds, orderIds)
}

/** True when two id lists are element-for-element identical. */
export function sameIds(left: string[], right: string[]): boolean {
  return left.length === right.length && left.every((item, index) => item === right[index])
}

/**
 * Apply a hand-picked order WITHIN each chronological group, never across them.
 *
 * A drag used to switch the whole list into a frozen manual mode with no date
 * dividers at all: one reorder and the sidebar stopped being chronological,
 * permanently and for every row. Ranking and grouping are separate concerns —
 * the calendar buckets stay exactly where recency put them, and the manual
 * order only decides the sequence of rows inside a bucket.
 *
 * Rows are permuted as clusters (a parent plus the branch children nested under
 * it) so a reorder can't strand a branch away from its parent, and clusters the
 * saved order doesn't know about keep the slot recency gave them — only the
 * ones it names are re-slotted, into the positions they already occupied.
 */
export function orderRowsWithinGroups(rows: SidebarListRow[], orderIds: string[]): SidebarListRow[] {
  if (!orderIds.length || !rows.length) {
    return rows
  }

  const rank = new Map(orderIds.map((id, index) => [id, index]))
  const out: SidebarListRow[] = []
  let cluster: SidebarListRow[][] = []
  let reordered = false

  // Re-slot the clusters the saved order names, leaving every other cluster
  // (and every divider) exactly where it is.
  const flushGroup = () => {
    const ranked = cluster
      .map((rows_, index) => ({ index, rank: rank.get(clusterId(rows_)) }))
      .filter((entry): entry is { index: number; rank: number } => entry.rank !== undefined)

    const slots = ranked.map(entry => entry.index)
    const sorted = [...ranked].sort((a, b) => a.rank - b.rank)
    const next = [...cluster]

    slots.forEach((slot, i) => {
      const source = cluster[sorted[i].index]

      reordered ||= source !== next[slot]
      next[slot] = source
    })

    for (const rows_ of next) {
      out.push(...rows_)
    }

    cluster = []
  }

  for (const row of rows) {
    if (row.kind === 'divider') {
      flushGroup()
      out.push(row)

      continue
    }

    // A branch child rides with the parent cluster above it.
    if (row.entry.branchStem && cluster.length) {
      cluster[cluster.length - 1].push(row)

      continue
    }

    cluster.push([row])
  }

  flushGroup()

  return reordered ? out : rows
}

/** A cluster's identity: its root row's session id. */
function clusterId(rows: SidebarListRow[]): string {
  const root = rows[0]

  return root.kind === 'session' ? root.entry.session.id : ''
}

/** The reorderable ids of a rendered row list: root sessions, in render order. */
export function reorderableRowIds(rows: SidebarListRow[]): string[] {
  return rows.flatMap(row => (row.kind === 'session' && !row.entry.branchStem ? [row.entry.session.id] : []))
}
