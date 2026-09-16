import { atom } from 'nanostores'

import type { SessionInfo } from '@/types/hermes'

// The Pinned section's rows, published by the sidebar for keyboard navigation.
// `resolvePinnedSessions()` decides the visible order (local pin ids first,
// then backend-flagged rows); this store only carries it to keybind handlers
// that live outside the sidebar tree (`use-keybinds.ts`).
export const $pinnedRows = atom<readonly SessionInfo[]>([])

/**
 * Step through the pinned rows the way the sidebar displays them.
 *
 * The active session matches by either identity — live tip or lineage root —
 * the same rule the sidebar applies when deciding which section a row and a
 * pin belong to. With no match (fresh draft, a pinned row that is not the
 * current one), next starts from the first row and previous from the last.
 * Returns null when there is nothing to select.
 */
export function stepPinnedSession(
  rows: readonly SessionInfo[],
  currentId: null | string,
  direction: 1 | -1
): null | string {
  if (rows.length === 0) {
    return null
  }

  const index = rows.findIndex(
    row => row.id === currentId || (row._lineage_root_id != null && row._lineage_root_id === currentId)
  )

  if (index === -1) {
    return direction === 1 ? rows[0].id : rows[rows.length - 1].id
  }

  const wrapped = ((index + direction) % rows.length + rows.length) % rows.length

  return rows[wrapped].id
}
