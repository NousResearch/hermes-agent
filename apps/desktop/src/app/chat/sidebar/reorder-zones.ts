import { atom } from 'nanostores'

/**
 * Sidebar reorder drop-zones — the seam that lets the shared pointer session
 * drag (session-drag.ts) reorder a flat sidebar list by DROP LOCATION, the
 * same way it links/splits by dropping over a chat surface.
 *
 * A list (currently just Pinned) registers itself while mounted: its container
 * element, a live getter for its ordered ids, and the commit callback. The drag
 * resolver snapshots the registered zones at engage, hit-tests the pointer
 * against a zone and its rows, and on release calls `onReorder` with the moved
 * order. One drag system, no dnd-kit handle to hunt for — dragging a row within
 * the bar reorders; dragging it onto the chat links (session-drag's other
 * targets are unchanged).
 */
export interface ReorderZone {
  el: HTMLElement
  /** Live ordered ids — read at engage so a mid-session refresh can't stale it. */
  getIds: () => string[]
  onReorder: (ids: string[]) => void
}

const zones = new Set<ReorderZone>()

/** Register a reorder zone for the drag resolver. Returns an unregister fn. */
export function registerReorderZone(zone: ReorderZone): () => void {
  zones.add(zone)

  return () => {
    zones.delete(zone)
  }
}

/** A resolver-time snapshot of one zone: its rect, its rows' vertical spans in
 *  order, and the commit callback. Rows are captured from the live DOM so the
 *  slot math never depends on the virtualizer or render timing. */
export interface ReorderZoneSnapshot {
  bottom: number
  ids: string[]
  left: number
  onReorder: (ids: string[]) => void
  right: number
  /** Row mid-Y per id, in DOM order — the insertion boundaries. */
  rows: { id: string; mid: number }[]
  top: number
}

/** Snapshot every registered zone. Called once at drag engage. */
export function snapshotReorderZones(): ReorderZoneSnapshot[] {
  const snaps: ReorderZoneSnapshot[] = []

  for (const zone of zones) {
    const zoneRect = zone.el.getBoundingClientRect()
    const ids = zone.getIds()
    const rows: { id: string; mid: number }[] = []

    for (const id of ids) {
      // Rows tag themselves with data-reorder-row so the snapshot can find the
      // element that owns each id without coupling to row markup. Match by
      // scanning tagged rows (not a CSS attribute selector) so an id with
      // selector-special characters — or a context without CSS.escape — can't
      // break the lookup.
      const rowEl = [...zone.el.querySelectorAll<HTMLElement>('[data-reorder-row]')].find(
        el => el.dataset.reorderRow === id
      )

      if (rowEl) {
        const r = rowEl.getBoundingClientRect()
        rows.push({ id, mid: r.top + r.height / 2 })
      }
    }

    snaps.push({
      bottom: zoneRect.bottom,
      ids,
      left: zoneRect.left,
      onReorder: zone.onReorder,
      right: zoneRect.right,
      rows,
      top: zoneRect.top
    })
  }

  return snaps
}

export interface ReorderTarget {
  /** Insert the dragged id before this id, or at the end when null. */
  before: null | string
  onReorder: (ids: string[]) => void
  /** The zone's ordered ids at snapshot time. */
  ids: string[]
}

/**
 * Resolve the reorder target for a pointer position, or null when it isn't over
 * a zone that contains the dragged id. Pure given the snapshots — unit-tested.
 *
 * The insertion boundary is the first row whose mid-Y is below the pointer; ties
 * and below-the-last-row resolve to end (`before: null`). A zone only accepts a
 * drag whose id it already owns, so dropping a non-pinned row onto Pinned is a
 * miss (falls back to the drag's other targets / deny), never a spurious move.
 */
export function resolveReorderTarget(
  snapshots: ReorderZoneSnapshot[],
  draggedId: string,
  x: number,
  y: number
): null | ReorderTarget {
  for (const snap of snapshots) {
    if (x < snap.left || x > snap.right || y < snap.top || y > snap.bottom) {
      continue
    }

    if (!snap.ids.includes(draggedId)) {
      // Over the zone, but this row doesn't belong to it — not a reorder.
      return null
    }

    const beforeRow = snap.rows.find(row => row.id !== draggedId && y < row.mid)

    return { before: beforeRow ? beforeRow.id : null, ids: snap.ids, onReorder: snap.onReorder }
  }

  return null
}

/**
 * Apply a resolved reorder to an id list: move `draggedId` to sit immediately
 * before `before` (or to the end when null). Returns the same reference when the
 * order is unchanged so callers can skip a no-op commit. Pure — unit-tested.
 */
export function reorderIds(ids: string[], draggedId: string, before: null | string): string[] {
  const from = ids.indexOf(draggedId)

  if (from < 0) {
    return ids
  }

  const without = ids.filter(id => id !== draggedId)
  const insertAt = before === null ? without.length : without.indexOf(before)
  const target = insertAt < 0 ? without.length : insertAt
  const next = [...without.slice(0, target), draggedId, ...without.slice(target)]

  // No-op guard: same order in, same reference out.
  return next.every((id, i) => id === ids[i]) ? ids : next
}

/** Live reorder hint for the insertion-line UI: which zone, insert before which
 *  id. A dedicated atom (not the heavy `$dropHint`) so only the pinned list
 *  re-renders on hint churn, never the chat surfaces. */
export const $sidebarReorderHint = atom<null | { before: null | string; draggedId: string }>(null)
