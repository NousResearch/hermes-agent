import { findGroupOfPane } from '@/components/pane-shell/tree/model'
import { $layoutTree, focusedSessionTabAnchor } from '@/components/pane-shell/tree/store'
import { TILE_PANE_PREFIX } from '@/store/session-focus'
import {
  $sessionTiles,
  closeSessionTile,
  discardSessionTile,
  focusOpenSession,
  openSessionTile,
  type SessionTileWorkspaceScope
} from '@/store/session-states'

import { isUntouchedDraftTile } from './reuse-draft-tab'

/**
 * A plain click on a session while a session TAB is the focused tab loads the
 * session into that tab, like a link opened in the current browser tab. Routing
 * it into main instead replaced whichever tab happened to hold main — a tab the
 * user neither picked nor could identify, and one that moves whenever main is
 * closed and a neighbour is promoted into it (close-tab.ts).
 *
 * The new tab takes the focused tab's slot and is opened BEFORE the old one
 * closes, so a lone tab's zone never empties and collapses under the click. The
 * replaced chat goes to the ⌘⇧T stack (a busy one keeps streaming in the
 * background — closeSessionTile keeps its state); an untouched untitled draft is
 * discarded instead, since it never held a conversation.
 *
 * False when the focused tab is main or not a session tab: the caller keeps
 * loading the session into main.
 */
export function openInFocusedSessionTab(storedSessionId: string, workspaceScope: SessionTileWorkspaceScope): boolean {
  const paneId = focusedSessionTabAnchor()
  const tree = $layoutTree.get()
  const group = paneId && tree ? findGroupOfPane(tree, paneId) : null

  if (!paneId?.startsWith(TILE_PANE_PREFIX) || !group || group.active !== paneId) {
    return false
  }

  const replaced = $sessionTiles.get().find(tile => `${TILE_PANE_PREFIX}${tile.storedSessionId}` === paneId)

  if (!replaced) {
    return false
  }

  const before = group.panes[group.panes.indexOf(paneId) + 1] ?? null
  openSessionTile(storedSessionId, 'center', paneId, before, workspaceScope)

  if (isUntouchedDraftTile(replaced)) {
    discardSessionTile(replaced.storedSessionId)
  } else {
    closeSessionTile(replaced.storedSessionId)
  }

  focusOpenSession(storedSessionId, workspaceScope)

  return true
}
