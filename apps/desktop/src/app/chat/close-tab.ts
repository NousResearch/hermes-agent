import { closeActiveTerminal } from '@/app/right-sidebar/terminal/terminals'
import { closeWorkspaceTab } from '@/components/pane-shell/tree/store'
import { isFocusWithin } from '@/lib/keybinds/combo'
import { $filePreviewTarget, $previewTarget, closeActiveRightRailTab } from '@/store/preview'

/**
 * ⌘W — close the tab of the context you're in, by precedence:
 *   1. a focused terminal → its active terminal tab,
 *   2. an open preview → its active preview tab (unchanged from pre-tiling),
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

  if ($filePreviewTarget.get() || $previewTarget.get()) {
    closeActiveRightRailTab()

    return true
  }

  return closeWorkspaceTab()
}
