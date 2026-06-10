// Secondary "session windows" — one extra OS window per chat so a user can
// work with multiple chats side by side. The pure, Electron-free pieces live
// here so they can be unit-tested with node --test (mirroring how the rest of
// electron/*.cjs splits testable logic out of the main.cjs monolith).

const { pathToFileURL } = require('node:url')

// Build the renderer URL for a secondary window. The renderer uses a
// HashRouter, so the session route lives after the '#'. The `?win=secondary`
// flag MUST sit in the query string BEFORE the '#': anything after the '#' is
// treated as the route by HashRouter and would break routeSessionId(). The
// renderer reads the flag from window.location.search to suppress the install /
// onboarding overlays and the global session sidebar.
function buildSessionWindowUrl(sessionId, { devServer, rendererIndexPath } = {}) {
  const route = `#/${encodeURIComponent(sessionId)}`

  if (devServer) {
    const base = devServer.endsWith('/') ? devServer.slice(0, -1) : devServer

    return `${base}/?win=secondary${route}`
  }

  return `${pathToFileURL(rendererIndexPath).toString()}?win=secondary${route}`
}

// A small registry keyed by sessionId that guarantees one window per chat:
// opening a session that already has a live window focuses it instead of
// spawning a duplicate, and a window removes itself from the registry when it
// closes. The actual BrowserWindow construction is injected (the `factory`) so
// this module stays free of Electron and is unit-testable.
function createSessionWindowRegistry() {
  const windows = new Map()

  function openOrFocus(sessionId, factory) {
    const key = typeof sessionId === 'string' ? sessionId.trim() : ''

    if (!key) {
      return null
    }

    const existing = windows.get(key)

    if (existing && !existing.isDestroyed()) {
      // Focus-or-create: never duplicate a window for the same chat.
      if (typeof existing.isMinimized === 'function' && existing.isMinimized()) {
        existing.restore?.()
      }

      if (typeof existing.isVisible === 'function' && !existing.isVisible()) {
        existing.show?.()
      }

      existing.focus?.()

      return existing
    }

    const win = factory(key)

    if (!win) {
      return null
    }

    windows.set(key, win)

    // Self-cleanup on close so the registry never holds a destroyed window.
    win.on?.('closed', () => {
      if (windows.get(key) === win) {
        windows.delete(key)
      }
    })

    return win
  }

  return {
    openOrFocus,
    get: key => windows.get(key),
    has: key => windows.has(key),
    get size() {
      return windows.size
    }
  }
}

module.exports = { buildSessionWindowUrl, createSessionWindowRegistry }
