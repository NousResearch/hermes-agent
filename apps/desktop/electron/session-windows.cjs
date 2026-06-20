// Secondary "session windows" — one extra OS window per chat so a user can
// work with multiple chats side by side. The pure, Electron-free pieces live
// here so they can be unit-tested with node --test (mirroring how the rest of
// electron/*.cjs splits testable logic out of the main.cjs monolith).

const { pathToFileURL } = require('node:url')

// Secondary windows open at the minimum usable size — a compact side panel for
// subagent watch / cmd-click session pop-out, not a second full desktop.
const SESSION_WINDOW_MIN_WIDTH = 420
const SESSION_WINDOW_MIN_HEIGHT = 620
const SESSION_RESTORE_SCHEMA_VERSION = 1
const MAX_RESTORABLE_SESSION_WINDOWS = 32

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

function normalizeSessionId(value) {
  return typeof value === 'string' ? value.trim() : ''
}

function finiteInteger(value) {
  return Number.isFinite(value) ? Math.round(value) : null
}

function normalizeRestoreBounds(bounds) {
  if (!bounds || typeof bounds !== 'object') {
    return null
  }

  const x = finiteInteger(bounds.x)
  const y = finiteInteger(bounds.y)
  const width = finiteInteger(bounds.width)
  const height = finiteInteger(bounds.height)

  if (x === null || y === null || width === null || height === null) {
    return null
  }

  return {
    x,
    y,
    width: Math.max(width, SESSION_WINDOW_MIN_WIDTH),
    height: Math.max(height, SESSION_WINDOW_MIN_HEIGHT)
  }
}

function normalizeRestoreEntry(entry) {
  if (!entry || typeof entry !== 'object') {
    return null
  }

  const sessionId = normalizeSessionId(entry.sessionId)
  if (!sessionId) {
    return null
  }

  const normalized = {
    sessionId,
    watch: entry.watch === true
  }

  const bounds = normalizeRestoreBounds(entry.bounds)
  if (bounds) {
    normalized.bounds = bounds
  }

  return normalized
}

function buildSessionRestoreSnapshot(entries, { createdAt = Date.now() } = {}) {
  const seen = new Set()
  const normalized = []

  for (const entry of Array.isArray(entries) ? entries : []) {
    const next = normalizeRestoreEntry(entry)
    if (!next || seen.has(next.sessionId)) {
      continue
    }

    seen.add(next.sessionId)
    normalized.push(next)

    if (normalized.length >= MAX_RESTORABLE_SESSION_WINDOWS) {
      break
    }
  }

  return {
    schemaVersion: SESSION_RESTORE_SCHEMA_VERSION,
    createdAt,
    entries: normalized
  }
}

function parseSessionRestoreSnapshot(raw) {
  if (!raw || typeof raw !== 'object' || raw.schemaVersion !== SESSION_RESTORE_SCHEMA_VERSION) {
    return buildSessionRestoreSnapshot([])
  }

  return buildSessionRestoreSnapshot(raw.entries, {
    createdAt: typeof raw.createdAt === 'number' && Number.isFinite(raw.createdAt) ? raw.createdAt : Date.now()
  })
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
  buildSessionRestoreSnapshot,
  buildSessionWindowUrl,
  chatWindowWebPreferences,
  createSessionWindowRegistry,
  parseSessionRestoreSnapshot,
  SESSION_RESTORE_SCHEMA_VERSION,
  SESSION_WINDOW_MIN_HEIGHT,
  SESSION_WINDOW_MIN_WIDTH
}
