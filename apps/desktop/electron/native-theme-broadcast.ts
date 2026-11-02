// OS light/dark switch fan-out (#128622).
//
// The nativeTheme "updated" handler in main used to push only the refreshed
// titlebar overlay, so a system appearance switch left every renderer on its
// stale color scheme: Chromium does not reliably re-fire
// prefers-color-scheme for a live page on Windows, and nothing told the page
// to re-resolve. This module forwards the resolved mode to every live window
// and invalidates for a repaint. Pure and Electron-free (the window surface is
// injected) so it can be unit-tested.

/** Renderer channel carrying the resolved OS appearance after a system switch. */
export const NATIVE_THEME_UPDATED_CHANNEL = 'hermes:native-theme-updated'

export interface NativeThemeBroadcastContents {
  isDestroyed(): boolean
  send(channel: string, payload: unknown): void
  /** Repaint trigger; absent on older shells — the IPC alone still re-resolves. */
  invalidate?(): void
}

export interface NativeThemeBroadcastWindow {
  isDestroyed(): boolean
  webContents?: NativeThemeBroadcastContents | null
}

export interface NativeThemeBroadcastOptions {
  /**
   * True while a turn is streaming. The notify still goes out (cheap, and it
   * is what flips the scheme), but the forced invalidate is skipped: the
   * streaming transcript repaints continuously, so it would only risk tearing
   * a frame mid-token.
   */
  streaming?: boolean
}

/**
 * Send the resolved OS appearance to every live window and invalidate for a
 * repaint. Returns the number of windows notified. Never throws: a window
 * mid-teardown must not break fan-out to the rest.
 */
export function broadcastNativeThemeUpdated(
  windows: Iterable<NativeThemeBroadcastWindow>,
  isDark: boolean,
  options: NativeThemeBroadcastOptions = {}
): number {
  const { streaming = false } = options
  let notified = 0

  for (const win of windows) {
    try {
      if (!win || win.isDestroyed()) {
        continue
      }

      const contents = win.webContents

      if (!contents || contents.isDestroyed()) {
        continue
      }

      contents.send(NATIVE_THEME_UPDATED_CHANNEL, { dark: isDark })
      notified += 1

      if (!streaming) {
        try {
          contents.invalidate?.()
        } catch {
          // A window mid-teardown can throw; the notify above already landed.
        }
      }
    } catch {
      // Same: one dying window must not break fan-out to the rest.
    }
  }

  return notified
}
