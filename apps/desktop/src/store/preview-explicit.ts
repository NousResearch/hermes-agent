/**
 * An explicit preview open (tool or user `openPreview`) must survive a later
 * `follow()` from a zone the user was already in. `follow()` copies that
 * zone's tab into `$rightRailActiveTabId`; when the opened pane lives in a
 * different group, that copy is the desync `read_preview` used to report.
 *
 * A focus change after the open is a new look, so follow may sync then.
 * Hover is not tracked here — the reader treats the pointer as a live look.
 */

import { findGroupOfPane } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import type { RightRailTabId } from '@/store/layout'

/** Pane id prefix for a preview tab. Must match `preview-tile.tsx`. */
export const PREVIEW_TILE_PREFIX = 'preview-tile'

type ExplicitOpen = {
  tabId: RightRailTabId
  /** `$activeTreeGroup` when the open happened. Unchanged means follow is
   *  still the pre-open zone, not a zone the user moved to afterwards. */
  activeGroupAtSelect: null | string
}

let explicit: ExplicitOpen | null = null

/** Stamp a tool/user open. Call before `selectRightRailTab` so a synchronous
 *  follow from that select already sees the guard. */
export function noteExplicitPreviewOpen(tabId: RightRailTabId): void {
  explicit = { activeGroupAtSelect: $activeTreeGroup.get(), tabId }
}

export function clearExplicitPreviewOpen(): void {
  explicit = null
}

function paneGroupId(tabId: string): null | string {
  const tree = $layoutTree.get()

  if (!tree) {
    return null
  }

  return findGroupOfPane(tree, `${PREVIEW_TILE_PREFIX}:${tabId}`)?.id ?? null
}

/**
 * True when copying `sourceGroupId`'s preview would overwrite an explicit
 * open that lives in a different group, and the user has not focused a
 * different zone since that open.
 */
export function explicitOpenBlocksZone(sourceGroupId: null | string, openTabIds: readonly string[]): boolean {
  if (!explicit || !openTabIds.includes(explicit.tabId)) {
    return false
  }

  if (sourceGroupId !== explicit.activeGroupAtSelect) {
    return false
  }

  const openedIn = paneGroupId(explicit.tabId)

  // Not placed yet: the stale zone must not win the race with reveal.
  if (!openedIn) {
    return true
  }

  return openedIn !== sourceGroupId
}
