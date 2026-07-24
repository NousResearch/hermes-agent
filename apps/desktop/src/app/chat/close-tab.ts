import { closeActiveTerminal } from '@/app/right-sidebar/terminal/terminals'
import { closeWorkspaceTab } from '@/components/pane-shell/tree/store'
import { isFocusWithin } from '@/lib/keybinds/combo'
import { $filePreviewTabs, $previewTarget, closeActiveRightRailTab } from '@/store/preview'
import { closeSessionTile, nextSessionTileForWorkspace } from '@/store/session-states'

/**
 * ⌘W — close the tab of the context you're in, by precedence:
 *   1. a focused terminal → its active terminal tab,
 *   2. right-rail tabs (live preview and/or file peeks),
 *   3. the MAIN zone → its active tab (a session tile stacked into the workspace).
 *   4. the MAIN (workspace) tab itself, when session tabs are stacked with it:
 *      the workspace can't close, so ⌘W shifts the NEXT session tab into main
 *      (loads it as the primary + drops its now-redundant tile).
 * Returns false when nothing closes, so ⌘W is a no-op — it never closes the
 * window (a bare workspace stays put). Shared by the keyboard path (Win/Linux)
 * and the macOS menu-accelerator IPC.
 *
 * `loadSessionIntoWorkspace` carries the app's route-based "load this session
 * into main" (the two call sites have router access); omitting it disables the
 * step-4 promotion (⌘W stays the pre-existing no-op on the main tab).
 */
export function closeActiveTab(loadSessionIntoWorkspace?: (storedSessionId: string) => void): boolean {
  if (isFocusWithin('[data-terminal]')) {
    closeActiveTerminal()

    return true
  }

  // Prefer tab *presence* over the derived active file target. After the live
  // preview is cleared, `$rightRailActiveTabId` can stay on `preview` while
  // file tabs remain (the rail UI falls back to tabs[0]). Gating only on
  // `$filePreviewTarget` made ⌘W fall through to closeWorkspaceTab() and look
  // broken with a file tab still on screen.
  if ($previewTarget.get() || $filePreviewTabs.get().length > 0) {
    return closeActiveRightRailTab()
  }

  // A closeable main-zone tab (a session tile that's the active tab) closes
  // outright; the uncloseable workspace tab returns false and falls through.
  if (closeWorkspaceTab()) {
    return true
  }

  // The main (workspace) tab is active and can't be closed — but if session
  // tabs are stacked with it, ⌘W shifts the next one into the main tab: drop
  // its tile (the session stays alive, no busy-close prompt) and load it into
  // main. Order matters — close the tile FIRST so the selection homes to the
  // workspace instead of re-fronting the tile.
  if (loadSessionIntoWorkspace) {
    const next = nextSessionTileForWorkspace()

    if (next) {
      closeSessionTile(next)
      loadSessionIntoWorkspace(next)

      return true
    }
  }

  return false
}
