// Secondary "session windows" — one extra OS window per chat so a user can
// work with multiple chats side by side. The pure, Electron-free pieces live
// here so they can be unit-tested with node --test (mirroring how the rest of
// electron/*.cjs splits testable logic out of the main.cjs monolith).

const path = require('node:path')
const { fileURLToPath, pathToFileURL } = require('node:url')

// Secondary windows open at the minimum usable size — a compact side panel for
// subagent watch / cmd-click session pop-out, not a second full desktop.
const SESSION_WINDOW_MIN_WIDTH = 420
const SESSION_WINDOW_MIN_HEIGHT = 620

// Shared webPreferences for every window that renders the chat transcript — the
// primary window AND the secondary session windows. Keeping it in one place is
// the whole point: the two BrowserWindow definitions in main.cjs used to be
// hand-copied, and the secondary windows silently lost `backgroundThrottling:
// false`, so a streamed answer stalled until the window regained focus.
//
// `backgroundThrottling: false` is load-bearing: the transcript streams to the
// screen through a requestAnimationFrame-gated flush, which Chromium pauses for
// blurred/occluded windows. A streaming chat app must keep painting in the
// background, so every chat window opts out. The preload path is injected
// because it depends on the Electron entry's __dirname.
function chatWindowWebPreferences(preloadPath) {
  return {
    preload: preloadPath,
    contextIsolation: true,
    webviewTag: true,
    sandbox: true,
    nodeIntegration: false,
    devTools: true,
    backgroundThrottling: false
  }
}

// Build the renderer URL for a secondary window. The renderer uses a
// HashRouter, so the session route lives after the '#'. The `?win=secondary`
// flag MUST sit in the query string BEFORE the '#': anything after the '#' is
// treated as the route by HashRouter and would break routeSessionId(). The
// renderer reads the flag from window.location.search to suppress the install /
// onboarding overlays and the global session sidebar. `new=1` marks the compact
// scratch window; `watch=1` marks a spectator window (e.g. a running subagent's
// session): the renderer resumes it lazily so the gateway never builds an agent
// just to stream into it.
function buildSessionWindowUrl(sessionId, { devServer, rendererIndexPath, watch, newSession } = {}) {
  const query = `?win=secondary${newSession ? '&new=1' : ''}${watch ? '&watch=1' : ''}`
  const route = newSession ? '#/' : `#/${encodeURIComponent(sessionId)}`

  if (devServer) {
    const base = devServer.endsWith('/') ? devServer.slice(0, -1) : devServer

    return `${base}/${query}${route}`
  }

  return `${pathToFileURL(rendererIndexPath).toString()}${query}${route}`
}

function isAllowedChatNavigation(rawUrl, { devServer, rendererIndexPath } = {}) {
  let parsed
  try {
    parsed = new URL(String(rawUrl || ''))
  } catch {
    return false
  }

  if (devServer) {
    let dev
    try {
      dev = new URL(devServer)
    } catch {
      return false
    }

    return parsed.protocol === dev.protocol && parsed.host === dev.host
  }

  if (parsed.protocol !== 'file:' || !rendererIndexPath) {
    return false
  }

  try {
    return path.resolve(fileURLToPath(parsed)) === path.resolve(rendererIndexPath)
  } catch {
    return false
  }
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

module.exports = {
  buildSessionWindowUrl,
  chatWindowWebPreferences,
  createSessionWindowRegistry,
  isAllowedChatNavigation,
  SESSION_WINDOW_MIN_HEIGHT,
  SESSION_WINDOW_MIN_WIDTH
}
