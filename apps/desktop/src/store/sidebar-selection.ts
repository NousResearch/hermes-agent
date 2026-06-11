import { atom } from 'nanostores'

/** Section identity for sidebar multi-select. A selection lives in exactly one
 * section at a time: every section shares the same bulk verbs (pin, archive,
 * delete, restore) but with section-specific meaning, so a cross-section
 * selection would have no honest answer for what its action bar applies. */
export type SidebarSectionKey = 'archived' | 'pinned' | 'results' | 'sessions' | `messaging:${string}`

export interface SidebarSelection {
  section: null | SidebarSectionKey
  /** Selected live session ids in click order; the LAST id is the range
   * anchor for the next shift-click. */
  ids: string[]
}

const EMPTY_SELECTION: SidebarSelection = { ids: [], section: null }

export const $sidebarSelection = atom<SidebarSelection>(EMPTY_SELECTION)

export function clearSidebarSelection() {
  const current = $sidebarSelection.get()

  if (current.section !== null || current.ids.length > 0) {
    $sidebarSelection.set(EMPTY_SELECTION)
  }
}

/** Toggle one row's membership. Selecting in a different section restarts the
 * selection there rather than silently merging two sections' rows. */
export function toggleSessionSelected(section: SidebarSectionKey, sessionId: string) {
  const current = $sidebarSelection.get()

  if (current.section !== section) {
    $sidebarSelection.set({ ids: [sessionId], section })

    return
  }

  const ids = current.ids.includes(sessionId)
    ? current.ids.filter(id => id !== sessionId)
    : [...current.ids, sessionId]

  $sidebarSelection.set(ids.length ? { ids, section } : EMPTY_SELECTION)
}

/** Shift-click: select the contiguous run between the current anchor (last
 * selected id) and `sessionId`, in the section's rendered order. A cold
 * shift-click (no selection yet) anchors from `seedAnchorId` — the section's
 * active/open row — so the run includes the row the user started from,
 * Finder-style, instead of beginning a one-row selection that reads as
 * "deselected my starting point". Falls back to a plain toggle when no anchor
 * is usable. The clicked id becomes the new anchor. */
export function rangeSelectSessions(
  section: SidebarSectionKey,
  sessionId: string,
  orderedIds: readonly string[],
  seedAnchorId?: null | string
) {
  const current = $sidebarSelection.get()
  let anchor = current.section === section ? current.ids[current.ids.length - 1] : undefined

  if (anchor === undefined && seedAnchorId && seedAnchorId !== sessionId) {
    anchor = seedAnchorId
  }

  const anchorIndex = anchor ? orderedIds.indexOf(anchor) : -1
  const targetIndex = orderedIds.indexOf(sessionId)

  if (anchorIndex < 0 || targetIndex < 0) {
    toggleSessionSelected(section, sessionId)

    return
  }

  const [from, to] = anchorIndex <= targetIndex ? [anchorIndex, targetIndex] : [targetIndex, anchorIndex]
  const range = orderedIds.slice(from, to + 1)
  const rangeSet = new Set(range)

  // Union with what's already selected; keep prior click order, append the
  // range in list order, and pin the clicked id to the end as the new anchor.
  const merged = [...current.ids.filter(id => !rangeSet.has(id)), ...range]

  $sidebarSelection.set({ ids: [...merged.filter(id => id !== sessionId), sessionId], section })
}

interface PrunableRow {
  id: string
  _lineage_root_id?: null | string
}

/** Reconcile the selection against its section's CURRENT rows. The sidebar
 * refreshes constantly in the background (every turn-end triggers one), so
 * this has to survive two kinds of churn without nuking an in-progress
 * selection:
 *
 *  - Auto-compression rotates a conversation's live id (root → tip) between
 *    clicks. A selected id that now matches a row's lineage root is REMAPPED
 *    to the new tip instead of dropped — otherwise one background compaction
 *    silently deselects the row while its checkbox is on screen.
 *  - A transient refresh can hand the section an EMPTY row list for a beat
 *    (slice swap mid-flight). Dropping everything then would reset the
 *    selection, so the next shift-click "starts over" with one row — the
 *    deselects-everything-but-the-last bug. An empty list keeps the selection
 *    untouched; real removals always come from a populated list.
 *
 * Ids that genuinely left a populated list (archived/deleted elsewhere, paged
 * out) are still dropped so the count stays honest. */
export function pruneSidebarSelection(section: SidebarSectionKey, rows: readonly PrunableRow[]) {
  const current = $sidebarSelection.get()

  if (current.section !== section || rows.length === 0) {
    return
  }

  const liveIds = new Set<string>()
  const tipByRoot = new Map<string, string>()

  for (const row of rows) {
    liveIds.add(row.id)

    if (row._lineage_root_id) {
      tipByRoot.set(row._lineage_root_id, row.id)
    }
  }

  const seen = new Set<string>()
  const ids: string[] = []

  for (const id of current.ids) {
    const live = liveIds.has(id) ? id : tipByRoot.get(id)

    if (live && !seen.has(live)) {
      seen.add(live)
      ids.push(live)
    }
  }

  if (ids.length === current.ids.length && ids.every((id, index) => id === current.ids[index])) {
    return
  }

  $sidebarSelection.set(ids.length ? { ids, section: current.section } : EMPTY_SELECTION)
}
