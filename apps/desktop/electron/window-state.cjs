// Window-state persistence for the Hermes Desktop app.
//
// Saves and restores window position, size, and maximized state so the
// app reopens in the same geometry the user left it. Follows the same
// persistence pattern as native-theme.json and translucency.json — a
// small JSON file in Electron's userData directory (%APPDATA%/Hermes/)
// written synchronously on close and on significant geometry changes.
//
// Multi-monitor safety: if the saved position falls on a display that
// is no longer connected, the window gets its default position/size
// instead of being stranded off-screen.

const fs = require('node:fs')
const path = require('node:path')
const { screen } = require('electron')

const WINDOW_STATE_FILE = 'window-state.json'
const SAVE_DEBOUNCE_MS = 400

// --- State shape ---
//
// {
//   x: number | null,        // screen-space left
//   y: number | null,        // screen-space top
//   width: number,           // normal (non-maximized) width
//   height: number,          // normal (non-maximized) height
//   isMaximized: boolean     // whether the window was maximized on close
// }
//
// Width and height are always the NORMAL size (the size the window snaps
// back to when un-maximized). isMaximized means "start already maximized".

function stateFilePath(userDataPath) {
  return path.join(userDataPath, WINDOW_STATE_FILE)
}

function defaultState() {
  return { x: null, y: null, width: 1220, height: 800, isMaximized: false }
}

function loadState(userDataPath) {
  try {
    const raw = fs.readFileSync(stateFilePath(userDataPath), 'utf8')
    const parsed = JSON.parse(raw)
    // Validate that all expected keys exist and have the right types.
    // Partial / corrupted files fall back to defaults for the missing keys.
    const state = { ...defaultState(), ...parsed }
    state.width = Math.max(400, Math.min(state.width, 9999))
    state.height = Math.max(620, Math.min(state.height, 9999))
    return state
  } catch {
    return defaultState()
  }
}

// Check whether the saved (x, y) position falls within a connected display.
// Returns true if at least one display contains the point.
function isPositionOnAnyDisplay(x, y) {
  if (x == null || y == null) return false
  try {
    const displays = screen.getAllDisplays()
    return displays.some(d => {
      const { x: dx, y: dy, width: dw, height: dh } = d.bounds
      return x >= dx && x < dx + dw && y >= dy && y < dy + dh
    })
  } catch {
    return false
  }
}

// Apply a previously saved state to a BrowserWindow.
// Returns the state that was applied (callers can inspect `isMaximized`).
function applyWindowState(win, userDataPath) {
  const saved = loadState(userDataPath)

  if (saved.x != null && saved.y != null && isPositionOnAnyDisplay(saved.x, saved.y)) {
    win.setPosition(saved.x, saved.y)
  }
  // Always set the normal size — it's used when the window is restored
  // from maximized state and when the window is not maximized.
  win.setSize(saved.width, saved.height)

  // Defer maximize until after the window is shown; on some platforms
  // (Windows) maximizing before first paint can glitch the title bar.
  win.once('ready-to-show', () => {
    if (saved.isMaximized && !win.isDestroyed() && typeof win.maximize === 'function') {
      win.maximize()
    }
  })

  return saved
}

// Start tracking a window and persist its state on geometry changes.
// Returns a cleanup function that tears down the listeners.
//
// `winKey` distinguishes state files when the desktop creates multiple
// windows — the primary window uses `'main'`, secondary session windows
// use their session id. Each gets its own persisted state file.
function trackWindowState(win, userDataPath, winKey) {
  const statePath = winKey
    ? path.join(userDataPath, `window-state-${sanitizeKey(winKey)}.json`)
    : stateFilePath(userDataPath)

  let debounceTimer = null
  let savedNormal = null // { width, height } captured before maximize

  // Save the current geometry to disk.
  function persist() {
    if (!win || win.isDestroyed()) return

    const isMaximized = typeof win.isMaximized === 'function' && win.isMaximized()
    let width, height

    if (isMaximized && savedNormal) {
      // Use the pre-maximize normal size for restore.
      width = savedNormal.width
      height = savedNormal.height
    } else {
      const size = win.getSize()
      width = size[0]
      height = size[1]
    }

    let x = null
    let y = null
    if (!isMaximized) {
      const pos = win.getPosition()
      x = pos[0]
      y = pos[1]
    }

    const state = { x, y, width, height, isMaximized }

    try {
      fs.mkdirSync(path.dirname(statePath), { recursive: true })
      fs.writeFileSync(statePath, JSON.stringify(state, null, 2), 'utf8')
    } catch (error) {
      console.error(`[window-state] persist failed: ${error.message}`)
    }
  }

  function schedulePersist() {
    if (debounceTimer) clearTimeout(debounceTimer)
    debounceTimer = setTimeout(() => {
      debounceTimer = null
      persist()
    }, SAVE_DEBOUNCE_MS)
    // Unref so the timer doesn't keep the process alive on quit.
    if (debounceTimer && typeof debounceTimer.unref === 'function') {
      debounceTimer.unref()
    }
  }

  function onMaximize() {
    // Capture the normal size just before the window fills the screen.
    if (!win.isDestroyed()) {
      const size = win.getSize()
      savedNormal = { width: size[0], height: size[1] }
    }
    persist()
  }

  function onUnmaximize() {
    savedNormal = null
    persist()
  }

  function onResizeOrMove() {
    // When maximized, resizes/moves are OS-driven; we only care about the
    // normal geometry. Debounce to avoid hammering the disk during drag.
    if (!win.isDestroyed() && typeof win.isMaximized === 'function' && !win.isMaximized()) {
      schedulePersist()
    }
  }

  function onClose() {
    // Flush any pending debounced write.
    if (debounceTimer) {
      clearTimeout(debounceTimer)
      debounceTimer = null
    }
    persist()
  }

  win.on('maximize', onMaximize)
  win.on('unmaximize', onUnmaximize)
  win.on('resize', onResizeOrMove)
  win.on('move', onResizeOrMove)
  // Use 'close' (not 'closed') so the window is still alive when we save.
  win.on('close', onClose)

  return function cleanup() {
    if (debounceTimer) {
      clearTimeout(debounceTimer)
      debounceTimer = null
    }
    try {
      if (win && !win.isDestroyed()) {
        win.removeListener('maximize', onMaximize)
        win.removeListener('unmaximize', onUnmaximize)
        win.removeListener('resize', onResizeOrMove)
        win.removeListener('move', onResizeOrMove)
        win.removeListener('close', onClose)
      }
    } catch {
      // Best-effort cleanup.
    }
  }
}

function sanitizeKey(key) {
  return String(key).replace(/[^a-zA-Z0-9_-]/g, '_')
}

module.exports = {
  applyWindowState,
  defaultState,
  loadState,
  trackWindowState
}
