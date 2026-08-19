import x11 from 'x11'

/**
 * Reliable cursor position on X11, bypassing Electron's frozen cache.
 *
 * `screen.getCursorScreenPoint()` is a Chromium-side cache updated by X11
 * motion events the browser process receives. The moment the HUD window is
 * `setIgnoreMouseEvents(true)` — click-through — the window stops receiving
 * those events, the cache freezes on the last point, and the Linux cursor
 * feed (startHudCursorFeed) keeps computing the same point forever, so the
 * dedup guard `key === last` never fires and the renderer never hears that the
 * pointer came back to the bar. One-way door on X11.
 *
 * XQueryPointer asks the X server itself, which always knows where the pointer
 * is, regardless of any window's input shape. That is exactly what `xdotool
 * getmouselocation` does. This module wraps a minimal x11 client around that
 * single request, with a cache so the 60ms feed never piles up round-trips.
 *
 * The x11 package is a pure-JS X client (no native build). If it cannot
 * connect (no DISPLAY, Wayland-native without XWayland, ...), `create()` still
 * succeeds but `read()` returns null and the caller falls back to Electron's
 * API — same behaviour as today.
 */

interface Point {
  x: number
  y: number
}

export interface X11CursorReader {
  /** Latest known pointer position in screen pixels, or null if unavailable. */
  read(): Point | null
  /** Terminate the X connection. Safe to call multiple times. */
  close(): void
}

/**
 * Create an X11-backed cursor reader. Never throws; a failed connection
 * degrades to `read() → null`.
 */
export function createX11CursorReader(): X11CursorReader {
  // The x11 package ships no TypeScript types; `client` is the opaque X client
  // returned by createClient, used only through its runtime methods below.
  let client: any = null
  let root: number | null = null
  let cached: Point | null = null
  let pending = false
  let failed = false
  let closed = false

  x11.createClient((err, display) => {
    if (err || !display) {
      failed = true
      return
    }

    try {
      client = display.client
      root = display.screen?.[0]?.root ?? null
      if (!root) {
        failed = true
      }
    } catch {
      failed = true
    }
  })

  function poll(): void {
    if (failed || closed || pending || !client || root === null) {
      return
    }

    pending = true

    try {
      client.QueryPointer(root, (qerr: Error | null, reply: { rootX: number; rootY: number } | null) => {
        pending = false

        if (!qerr && reply) {
          cached = { x: reply.rootX, y: reply.rootY }
        }
      })
    } catch {
      pending = false
    }
  }

  return {
    read() {
      poll()
      return cached
    },

    close() {
      closed = true

      try {
        client?.terminate?.()
      } catch {
        // best effort
      }
    }
  }
}
