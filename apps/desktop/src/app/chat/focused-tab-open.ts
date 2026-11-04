import { findGroupOfPane } from '@/components/pane-shell/tree/model'
import { $layoutTree, focusedSessionTabAnchor } from '@/components/pane-shell/tree/store'
import { TILE_PANE_PREFIX } from '@/store/session-focus'
import {
  $sessionTiles,
  closeSessionTile,
  discardSessionTile,
  focusOpenSession,
  openSessionTile,
  type SessionTile,
  type SessionTileWorkspaceScope,
  type TileDock
} from '@/store/session-states'

import { isUntouchedDraftTile } from './reuse-draft-tab'

interface FocusedSessionTabSlot {
  /** The pane after the focused tab in its group: a replacement stacks before it. */
  before: null | string
  paneId: string
  replaced: SessionTile
}

/** The focused tab when it is a session TAB (not main, not a page); null otherwise. */
function focusedSessionTabSlot(): FocusedSessionTabSlot | null {
  const paneId = focusedSessionTabAnchor()
  const tree = $layoutTree.get()
  const group = paneId && tree ? findGroupOfPane(tree, paneId) : null

  if (!paneId?.startsWith(TILE_PANE_PREFIX) || !group || group.active !== paneId) {
    return null
  }

  const replaced = $sessionTiles.get().find(tile => `${TILE_PANE_PREFIX}${tile.storedSessionId}` === paneId)

  return replaced ? { before: group.panes[group.panes.indexOf(paneId) + 1] ?? null, paneId, replaced } : null
}

/** The replaced chat goes to the ⌘⇧T stack (a busy one keeps streaming in the background —
 *  closeSessionTile keeps its state); an untouched untitled draft never held a conversation. */
function retireReplacedTab(replaced: SessionTile): void {
  if (isUntouchedDraftTile(replaced)) {
    discardSessionTile(replaced.storedSessionId)
  } else {
    closeSessionTile(replaced.storedSessionId)
  }
}

/**
 * A plain click on a session while a session TAB is the focused tab loads the
 * session into that tab, like a link opened in the current browser tab. Routing
 * it into main instead replaced whichever tab happened to hold main — a tab the
 * user neither picked nor could identify, and one that moves whenever main is
 * closed and a neighbour is promoted into it (close-tab.ts).
 *
 * The new tab takes the focused tab's slot and is opened BEFORE the old one
 * closes, so a lone tab's zone never empties and collapses under the click.
 *
 * False when the focused tab is main or not a session tab: the caller keeps
 * loading the session into main.
 */
export function openInFocusedSessionTab(storedSessionId: string, workspaceScope: SessionTileWorkspaceScope): boolean {
  const slot = focusedSessionTabSlot()

  if (!slot) {
    return false
  }

  openSessionTile(storedSessionId, 'center', slot.paneId, slot.before, workspaceScope)
  retireReplacedTab(slot.replaced)
  focusOpenSession(storedSessionId, workspaceScope)

  return true
}

type OpenNewSessionTile = (
  dir: TileDock,
  options: { anchor: string; before: null | string; listed: boolean }
) => Promise<void>

/**
 * "New session" (the sidebar row and ⌘N) with a session TAB focused opens the
 * fresh chat as a new tab right after it. It never replaces the conversation
 * the user is in (that tab stays open, untouched), and it never resets main,
 * which replaced whichever tab held main. An untouched untitled tab already IS
 * a new session, so it is kept as is instead of minting a twin.
 *
 * False when the focused tab is main or not a session tab: the caller starts the
 * fresh draft in main as before.
 */
export function startNewSessionBesideFocusedTab(openNewSessionTile: OpenNewSessionTile): boolean {
  const slot = focusedSessionTabSlot()

  if (!slot) {
    return false
  }

  if (!isUntouchedDraftTile(slot.replaced)) {
    void openNewSessionTile('center', { anchor: slot.paneId, before: slot.before, listed: false })
  }

  return true
}
