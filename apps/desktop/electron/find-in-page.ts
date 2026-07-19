/**
 * Pure helpers for the desktop find-in-page bridge (Ctrl/Cmd+F).
 *
 * The renderer drives an Electron `webContents.findInPage` over IPC so it can
 * reuse the native "find-in-page" experience (incremental search, match
 * highlight, Enter to step, Shift+Enter to step backwards, Escape to clear)
 * across chat transcripts and editor panels. Everything in this module is
 * pure with respect to its inputs so the routing + payload shaping can be
 * unit-tested without booting a BrowserWindow.
 *
 * Multi-window correctness: the IPC handlers in main.ts resolve the
 * requesting window via `BrowserWindow.fromWebContents(event.sender)` so a
 * Cmd+F pressed in a secondary session window searches THAT window, not the
 * primary. The `found-in-page` results are forwarded back to the same sender
 * — see {@link installFoundInPageForwarder}.
 */

/** Match options accepted by the renderer's `findInPage` bridge call. */
export interface FindInPageOptions {
  /** Step direction. Defaults to `true` (forward). */
  forward?: boolean
  /**
   * `true` to advance to the next/previous match using the previous query;
   * `false` to (re)search the current `query` from scratch. The renderer
   * passes `false` on a fresh query and `true` on Enter / Shift+Enter.
   */
  findNext?: boolean
}

/** Payload shape sent back to the renderer on every `found-in-page` event. */
export interface FoundInPagePayload {
  /** 1-indexed ordinal of the active match, or 0 when none. */
  activeMatchOrdinal: number
  /** Total matches in the document for the current query. */
  count: number
}

/**
 * Defensive projection of Electron's `found-in-page` event result. Electron
 * exposes more fields (finalUpdate, selectionArea, etc.) that we don't need;
 * keeping the projection explicit makes the wire shape auditable and keeps
 * tests independent of the runtime type.
 */
export function formatFoundInPage(result: {
  activeMatchOrdinal?: number
  matches?: number
}): FoundInPagePayload {
  return {
    activeMatchOrdinal: Number(result?.activeMatchOrdinal ?? 0),
    count: Number(result?.matches ?? 0)
  }
}

/**
 * Issue a `findInPage` against the given `webContents`. No-op when the
 * webContents is missing or destroyed — surfaces as a silent miss rather
 * than throwing across the IPC boundary, matching Electron's own semantics
 * for a destroyed renderer.
 */
export function performFind(
  webContents: Electron.WebContents | null | undefined,
  query: string,
  options: FindInPageOptions | null | undefined
): void {
  if (!webContents || webContents.isDestroyed()) {
    return
  }

  const opts = options && typeof options === 'object' ? options : {}

  webContents.findInPage(String(query ?? ''), {
    forward: opts.forward !== false,
    findNext: Boolean(opts.findNext)
  })
}

/**
 * Stop the current find and clear highlights. The default `action` matches
 * what the renderer sends on Escape / close.
 */
export function stopFind(
  webContents: Electron.WebContents | null | undefined,
  action: 'clearSelection' | 'keepSelection' | 'activateSelection' = 'clearSelection'
): void {
  if (!webContents || webContents.isDestroyed()) {
    return
  }

  webContents.stopFindInPage(action)
}

/**
 * Install a `found-in-page` listener on the given sender `webContents` and
 * forward each result back to the SAME renderer (via `webContents.send`).
 *
 * Returns an uninstall function. Call it from `webContents.on('destroyed', …)`
 * to avoid leaking the listener when the window goes away — Electron does
 * not auto-detach webContents listeners on close.
 *
 * The forwarder is intentionally bound to a single sender rather than the
 * primary window: a Cmd+F pressed in a secondary session window must
 * highlight matches in THAT window, and the match counter must reflect
 * THAT window's DOM, not the primary's.
 */
export function installFoundInPageForwarder(
  webContents: Electron.WebContents | null | undefined
): () => void {
  if (!webContents || webContents.isDestroyed()) {
    return () => {}
  }

  const handler = (_event: Electron.Event, result: Parameters<typeof formatFoundInPage>[0]) => {
    if (webContents.isDestroyed()) {
      return
    }

    webContents.send('hermes:found-in-page', formatFoundInPage(result))
  }

  webContents.on('found-in-page', handler)

  return () => {
    webContents.off('found-in-page', handler)
  }
}
