/**
 * Pure helpers for persisting and restoring the main window's size, position,
 * and maximized state across launches (see window-state.json in userData).
 *
 * Kept dependency-free and side-effect-free so the validation logic — which is
 * the part that actually matters (rejecting garbage, dropping off-screen
 * positions) — can be unit-tested with `node --test` without spinning up
 * Electron. main.cjs supplies the real file I/O and the live `screen` displays.
 */

// Defaults match the historical hardcoded BrowserWindow size, so a fresh
// install (no saved state) behaves exactly as before.
const DEFAULT_WIDTH = 1220
const DEFAULT_HEIGHT = 800
// Mirror the BrowserWindow minWidth/minHeight in main.cjs (400 since the
// hover-reveal sidebar work allowed much narrower windows) so a persisted
// size is never clamped above what the live window itself permits.
const MIN_WIDTH = 400
const MIN_HEIGHT = 620

// Require this much of the window to overlap a display work area before we
// trust a saved position — enough that the title bar stays grabbable. Guards
// against restoring onto a monitor that has since been unplugged.
const MIN_VISIBLE_PX = 48

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value)
}

/**
 * Parse a raw persisted object into a clean window-state, or null if it is
 * missing/garbage. width/height are required and clamped to the minimums;
 * x/y are optional and only kept when both are present and finite.
 */
function sanitizeWindowState(raw) {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  if (!isFiniteNumber(raw.width) || !isFiniteNumber(raw.height)) {
    return null
  }

  const state = {
    width: Math.max(MIN_WIDTH, Math.round(raw.width)),
    height: Math.max(MIN_HEIGHT, Math.round(raw.height)),
    isMaximized: raw.isMaximized === true
  }

  if (isFiniteNumber(raw.x) && isFiniteNumber(raw.y)) {
    state.x = Math.round(raw.x)
    state.y = Math.round(raw.y)
  }

  return state
}

/**
 * True when `bounds` overlaps at least one display's work area by more than
 * MIN_VISIBLE_PX in both axes. `displays` is Electron's screen.getAllDisplays()
 * shape: an array of objects each with a `workArea` {x, y, width, height}.
 */
function boundsVisibleOnDisplays(bounds, displays) {
  if (
    !bounds ||
    !isFiniteNumber(bounds.x) ||
    !isFiniteNumber(bounds.y) ||
    !isFiniteNumber(bounds.width) ||
    !isFiniteNumber(bounds.height)
  ) {
    return false
  }

  if (!Array.isArray(displays) || displays.length === 0) {
    return false
  }

  return displays.some(display => {
    const area = display && display.workArea
    if (!area) {
      return false
    }

    const overlapX = Math.min(bounds.x + bounds.width, area.x + area.width) - Math.max(bounds.x, area.x)
    const overlapY = Math.min(bounds.y + bounds.height, area.y + area.height) - Math.max(bounds.y, area.y)

    return overlapX >= MIN_VISIBLE_PX && overlapY >= MIN_VISIBLE_PX
  })
}

/**
 * Turn a sanitized saved state (or null) into BrowserWindow size/position
 * options. Always returns width/height; only returns x/y when the saved
 * position is still visible on a current display, so an off-screen window
 * falls back to Electron's default centering instead of opening out of reach.
 */
function computeWindowOptions(savedState, displays) {
  const options = {
    width: savedState && isFiniteNumber(savedState.width) ? savedState.width : DEFAULT_WIDTH,
    height: savedState && isFiniteNumber(savedState.height) ? savedState.height : DEFAULT_HEIGHT
  }

  if (
    savedState &&
    isFiniteNumber(savedState.x) &&
    isFiniteNumber(savedState.y) &&
    boundsVisibleOnDisplays(
      { x: savedState.x, y: savedState.y, width: options.width, height: options.height },
      displays
    )
  ) {
    options.x = savedState.x
    options.y = savedState.y
  }

  return options
}

module.exports = {
  DEFAULT_WIDTH,
  DEFAULT_HEIGHT,
  MIN_WIDTH,
  MIN_HEIGHT,
  MIN_VISIBLE_PX,
  sanitizeWindowState,
  boundsVisibleOnDisplays,
  computeWindowOptions
}
