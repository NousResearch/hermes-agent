import { closeActiveTerminal } from '@/app/right-sidebar/terminal/terminals'
import { closeWorkspaceTab } from '@/components/pane-shell/tree/store'
import { isFocusWithin } from '@/lib/keybinds/combo'
import { $filePreviewTabs, $previewTarget, closeActiveRightRailTab } from '@/store/preview'

/**
 * ⌘W — close the tab of the context you're in, by precedence:
 *   1. a focused terminal → its active terminal tab,
 *   2. right-rail tabs (live preview and/or file peeks),
 *   3. the MAIN zone → its active tab (a session tile stacked into the workspace).
 * Returns false when nothing closes, so ⌘W is a no-op — it never closes the
 * window (a bare workspace stays put). Shared by the keyboard path (Win/Linux)
 * and the macOS menu-accelerator IPC.
 */
export function closeActiveTab(): boolean {
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

  return closeWorkspaceTab()
}
