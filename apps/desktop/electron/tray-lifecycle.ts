/**
 * Pure decision logic for the system tray / menu-bar icon lifecycle.
 * No Electron imports so it stays unit-testable without spinning up
 * a BrowserWindow or NSStatusItem.
 */

/**
 * Given the current tray + handoff state, should closing the main
 * window hide-to-tray (keep process alive) instead of destroying it?
 */
export function shouldHideToTray(opts: {
  hasTray: boolean
  isHandoff: boolean
}): boolean {
  if (!opts.hasTray) return false
  if (opts.isHandoff) return false
  return true
}

export type ToggleAction = 'hide' | 'minimize' | 'show' | 'create'

/**
 * Given the current window / visibility / platform state, what should
 * clicking the tray icon do?
 */
export function decideTrayClickAction(opts: {
  hasWindow: boolean
  isMac: boolean
  isWindowVisible?: boolean
}): ToggleAction {
  if (!opts.hasWindow) return 'create'
  if (opts.isWindowVisible) {
    // Hide on macOS (Cmd+H-style), minimize elsewhere
    return opts.isMac ? 'hide' : 'minimize'
  }
  return 'show'
}

/**
 * Should the app quit when the last window is closed?
 * Pure logic: no Electron dependencies.
 */
export function shouldQuitOnLastWindowClose(opts: {
  isHandoff: boolean
  hasTray: boolean
  isMac: boolean
}): 'quit' | 'stay-alive' {
  if (opts.isHandoff) return 'quit'
  if (opts.hasTray) return 'stay-alive'
  if (opts.isMac) return 'stay-alive'
  return 'quit'
}
