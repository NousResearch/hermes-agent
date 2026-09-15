/**
 * Pure window-focus sequencing for the Electron main process.
 *
 * On macOS, `BrowserWindow.focus()` only makes the window key *within* the
 * application. When another application is active it does not bring Hermes to
 * the foreground — the user clicks the Dock icon and the window never comes
 * forward. Activating the application (`app.focus()`, which maps to
 * `-[NSApplication activateIgnoringOtherApps:]`) is what raises the whole app.
 *
 * The application-activation callback is injected so this stays a pure,
 * host-independent function; callers pass it only on platforms that need it
 * (macOS) and MUST only reach this from user-initiated paths (Dock `activate`,
 * HUD toggle, deep-link reveal) — never from a background event, which would
 * steal the user's foreground.
 */
export type FocusableWindow = {
  isDestroyed: () => boolean
  isMinimized: () => boolean
  restore: () => void
  isVisible: () => boolean
  show: () => void
  focus: () => void
}

export type ActivateApplicationFn = () => void

export function focusTargetWindow(
  win: FocusableWindow | null | undefined,
  activateApplication?: ActivateApplicationFn
): void {
  if (!win || win.isDestroyed()) {
    return
  }

  if (win.isMinimized()) {
    win.restore()
  }

  if (!win.isVisible()) {
    win.show()
  }

  win.focus()

  // After the window is key, raise the whole application so a user-initiated
  // focus request actually brings the app forward (macOS).
  activateApplication?.()
}
