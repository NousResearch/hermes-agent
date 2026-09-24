import type { BrowserWindow, WebContents } from 'electron'

/**
 * Live window state to attach to a connection reply (#102451).
 *
 * A republished connection must carry the calling window's CURRENT chrome state
 * (fullscreen, maximized, …), not whatever `startHermes()` baked into the cached
 * backend descriptor at mint time. The backend pool entry outlives renderer
 * reloads, reconnects and sleep/wake, so a mint-time `...getWindowState()`
 * spread goes stale the moment the user toggles fullscreen — and a stale
 * `isFullscreen: false` overwrites the live flag in the renderer, losing the
 * fullscreen titlebar inset until the next toggle. Reading the state at IPC
 * reply time keeps every republish consistent with the
 * `hermes:window-state-changed` live-push path.
 *
 * `undefined` when no usable window exists (sender gone, fallback destroyed) —
 * callers then leave the connection's own (possibly stale) fields untouched
 * rather than fabricating state.
 */

export interface LiveWindowStateDeps<TWindowState> {
  /** Electron's `BrowserWindow.fromWebContents` (injectable for tests). */
  fromWebContents: (contents: WebContents) => BrowserWindow | null
  /** The `getWindowState(win)` snapshot used elsewhere in main. */
  getWindowState: (win: BrowserWindow) => TWindowState
  /** Primary window — the fallback when the reply has no live sender. */
  fallback: BrowserWindow | null
}

export function liveWindowState<TWindowState>(
  sender: WebContents | undefined,
  deps: LiveWindowStateDeps<TWindowState>
): TWindowState | undefined {
  const win = (sender && deps.fromWebContents(sender)) || deps.fallback
  if (!win || win.isDestroyed()) {
    return undefined
  }
  return deps.getWindowState(win)
}

/** Overlay the (already computed) window state onto a connection reply. */
export function overlayWindowState<TConnection extends object, TWindowState extends object>(
  connection: TConnection,
  windowState: TWindowState | undefined
): TConnection & TWindowState {
  return (windowState ? { ...connection, ...windowState } : { ...connection }) as TConnection & TWindowState
}
