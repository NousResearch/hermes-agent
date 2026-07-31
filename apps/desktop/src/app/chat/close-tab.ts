import { mainChatOccupied } from '@/app/open-session'
import { closeActiveTerminal } from '@/app/right-sidebar/terminal/terminals'
import { $workspaceIsPage } from '@/app/routes'
import { closeFocusedSessionTab } from '@/components/pane-shell/tree/store'
import { isFocusWithin } from '@/lib/keybinds/combo'
import { $previewTabs, closeActiveRightRailTab } from '@/store/preview'
import { requestFreshSession } from '@/store/profile'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'
import { closeSessionTile, nextSessionTileForWorkspace } from '@/store/session-states'

/**
 * Close the MAIN tab. The workspace pane itself can't leave the tree, so
 * "closing" it means emptying it, and what fills the hole depends on what's
 * stacked beside it:
 *
 *  - session tabs stacked with it → the next one shifts INTO main (drop its
 *    tile, load it as the primary — the session stays alive, no busy prompt),
 *  - nothing stacked → main drops to a fresh "New session" draft.
 *
 * The second half is what makes the gesture honest when main is the ONLY tab:
 * ⌘W / ⌘-click / middle-click used to be a dead key there, since the only
 * available answer was "remove the pane", which this app never does.
 *
 * Returns false when there is nothing to close — a blank draft (already the
 * post-close state) or a full-page view (skills / artifacts, which isn't a
 * chat and owns no tab). ⌘W then stays a no-op; it never closes the window.
 *
 * `loadSessionIntoWorkspace` carries the app's route-based "load this session
 * into main"; omitting it disables the promotion half.
 */
export function closeWorkspaceTab(loadSessionIntoWorkspace?: (storedSessionId: string) => void): boolean {
  // Order matters — close the tile FIRST so the selection homes to the
  // workspace instead of re-fronting the tile.
  if (loadSessionIntoWorkspace) {
    const next = nextSessionTileForWorkspace()

    if (next) {
      closeSessionTile(next)
      loadSessionIntoWorkspace(next)

      return true
    }
  }

  if ($workspaceIsPage.get() || !mainChatOccupied($activeSessionId.get(), $selectedStoredSessionId.get())) {
    return false
  }

  requestFreshSession()

  return true
}

/**
 * ⌘W — close the tab of the context you're in, by precedence:
 *   1. a focused terminal → its active terminal tab,
 *   2. right-rail tabs (live preview and/or file peeks),
 *   3. the FOCUSED chat zone → its active tab (a session tile stacked into it).
 *   4. the workspace tab itself — see `closeWorkspaceTab`.
 * Returns false when nothing closes, so ⌘W is a no-op — it never closes the
 * window. Shared by the keyboard path (Win/Linux) and the macOS
 * menu-accelerator IPC.
 *
 * Steps 3-4 follow the same focused zone ⌘1…⌘9 indexes, so a second chat zone
 * with its own tab strip closes ITS tab instead of main's.
 */
export function closeActiveTab(loadSessionIntoWorkspace?: (storedSessionId: string) => void): boolean {
  if (isFocusWithin('[data-terminal]')) {
    closeActiveTerminal()

    return true
  }

  // Gate on tab *presence*, not on the selection: a stale `$rightRailActiveTabId`
  // would otherwise make ⌘W fall through to closeFocusedSessionTab() and look
  // broken with a tab still on screen. The store resolves which tab that is.
  if ($previewTabs.get().length > 0) {
    return closeActiveRightRailTab()
  }

  // A closeable tab in the focused chat zone (a session tile that's the active
  // tab) closes outright; the uncloseable workspace tab falls through.
  if (closeFocusedSessionTab()) {
    return true
  }

  return closeWorkspaceTab(loadSessionIntoWorkspace)
}
