// App-level zoom: one zoom value owned by the main process and applied to every
// window, instead of Chromium's default per-host + per-route zoom. The pure,
// Electron-free pieces live here so they can be unit-tested with node --test
// (mirroring session-windows.cjs). main.cjs injects the real persistence (a JSON
// file under userData) and window enumeration (BrowserWindow.getAllWindows).

// Chromium clamps zoom to roughly ±9 levels; match that so a persisted or
// keyboard-driven value never drifts outside the usable range.
const ZOOM_LEVEL_MIN = -9
const ZOOM_LEVEL_MAX = 9

function clampZoomLevel(value) {
  if (!Number.isFinite(value)) return 0

  return Math.min(Math.max(value, ZOOM_LEVEL_MIN), ZOOM_LEVEL_MAX)
}

// Build the zoom controller. Dependencies are injected so this stays free of
// Electron and is unit-testable:
//   - readLevel():        return the persisted level (any -> coerced + clamped)
//   - writeLevel(level):  persist the level
//   - getWindows():       return the live windows to apply zoom to
//   - getZoom(win) / setZoom(win, level): read/write a window's zoom level
//
// The controller owns the single source of truth (`level`), seeded from
// readLevel() at construction so a cold launch applies the saved zoom before any
// window paints.
function createZoomController({ readLevel, writeLevel, getWindows, getZoom, setZoom }) {
  let level = clampZoomLevel(readLevel())

  function getLevel() {
    return level
  }

  // Re-assert the app zoom on one window. Chromium resets the level on every
  // hash-route navigation, so main.cjs calls this on did-navigate-in-page (as
  // well as did-finish-load) to keep every route at the same scale. The
  // get/set guard avoids a redundant setZoom that would fire a zoom-changed
  // event for no actual change.
  function apply(win) {
    if (!win) return
    if (getZoom(win) !== level) {
      setZoom(win, level)
    }
  }

  // Set the app zoom, persist it when it actually changes, and apply to every
  // open window so the main window and all session windows stay in lockstep.
  function set(nextLevel) {
    const next = clampZoomLevel(nextLevel)
    if (next !== level) {
      level = next
      writeLevel(next)
    }

    for (const win of getWindows()) {
      apply(win)
    }
  }

  return { getLevel, apply, set }
}

module.exports = {
  ZOOM_LEVEL_MIN,
  ZOOM_LEVEL_MAX,
  clampZoomLevel,
  createZoomController
}
