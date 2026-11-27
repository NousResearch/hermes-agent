/**
 * Orders the macOS application menu relative to the first BrowserWindow.
 *
 * AppKit routes every key equivalent through the installed application
 * menu's delegate (`_populateFromDelegateWithEventProvider`), and Electron's
 * delegate segfaults when that callback fires before any window exists. The
 * updater's relaunch races the user's keystroke against startup, so the shell
 * died on both post-update relaunches reported in #115332.
 *
 * Two facts shape the fix:
 * - Electron installs its own default menu at `will-finish-launching` (before
 *   `ready`) unless `Menu.setApplicationMenu` has already been called, so the
 *   suppression must run at module scope, not inside `whenReady`.
 * - On macOS `Menu.setApplicationMenu(null)` never removes an already
 *   installed NSMenu; it only suppresses the default when it runs first.
 *
 * `installApplicationMenuAfterFirstWindow` is the ready-phase half: create the
 * first window, then install the menu (macOS only; the other platforms ship
 * without an application menu).
 */
export interface ApplicationMenuStartupDeps<TMenu> {
  isMac: boolean
  buildMenu: () => TMenu
  setApplicationMenu: (menu: TMenu | null) => void
  createWindow: () => void
}

export function installApplicationMenuAfterFirstWindow<TMenu>(deps: ApplicationMenuStartupDeps<TMenu>): void {
  deps.createWindow()

  if (deps.isMac) {
    deps.setApplicationMenu(deps.buildMenu())
  }
}
