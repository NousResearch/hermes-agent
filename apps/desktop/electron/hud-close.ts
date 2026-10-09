/**
 * Bounded graceful close for the HUD window.
 *
 * `win.close()` is a request, not an action: Electron asks the renderer to run
 * its beforeunload/unload handlers and only destroys the window once it
 * answers. That round-trip is what lets the HUD flush a half-typed draft into
 * the shared stash the app window re-reads on handoff — but a renderer that is
 * hung never answers, and an always-on-top window that never closes is exactly
 * the "HUD won't go away" report. Electron's own escalation is a 5 s
 * `unresponsive` event nobody acts on.
 *
 * So: ask nicely, then take the window away if the renderer has not let go by
 * the grace deadline. Pure and timer-injected so the deadline is testable.
 */

export interface HudCloseWindowLike {
  close(): void
  destroy(): void
  isDestroyed(): boolean
  once(event: 'closed', listener: () => void): unknown
}

interface HudCloseTimers {
  setTimer?: (fn: () => void, ms: number) => unknown
  clearTimer?: (handle: unknown) => void
}

/** How long a HUD renderer gets to answer the close before it is destroyed.
 *  A healthy beforeunload round-trip is tens of milliseconds; a transcript
 *  flush on a slow machine still fits comfortably. */
export const HUD_CLOSE_GRACE_MS = 1500

export function requestHudClose(
  win: HudCloseWindowLike,
  {
    setTimer = (fn, ms) => setTimeout(fn, ms),
    clearTimer = handle => clearTimeout(handle as ReturnType<typeof setTimeout>)
  }: HudCloseTimers = {},
  graceMs: number = HUD_CLOSE_GRACE_MS
): void {
  if (win.isDestroyed()) {
    return
  }

  const deadline = setTimer(() => {
    if (!win.isDestroyed()) {
      win.destroy()
    }
  }, graceMs)

  win.once('closed', () => clearTimer(deadline))
  win.close()
}
