// titlebar-overlay-width.cjs
//
// Decide how much right-edge width the renderer must keep clear for the native
// window controls (min/max/close). The renderer consumes this as
// --titlebar-tools-right: it anchors the right-hand system tool cluster and
// feeds the window title's max-width + drag-region math. Get it wrong and the
// tools sit under the buttons or float too far from them.
//
// Pure + param-injected so it's unit-testable without spinning up Electron.
// main.cjs wraps it with the live platform flags.

// Electron's Window Controls Overlay footprint on native Windows (3 buttons).
const NATIVE_OVERLAY_BUTTON_WIDTH = 144

// WSLg honors the same WCO, but its rendered control cluster sits further from
// the right edge than native Windows: measured against the live window, the
// minimize button's left edge is ~172px in, so reserving only 144 left the app
// tool cluster floating ~28px+ short of the buttons (a multi-icon gap). 168
// parks the cluster just left of the minimize button.
const WSL_OVERLAY_RESERVE_WIDTH = 168

/**
 * @param {object} opts
 * @param {boolean} opts.isWindows - process.platform === 'win32'
 * @param {boolean} opts.isWsl     - running under WSL/WSLg
 * @returns {number} pixels to reserve on the right edge (0 = none)
 */
function nativeOverlayWidth({ isWindows = false, isWsl = false } = {}) {
  // Native Windows: Electron paints the WCO at its standard footprint.
  if (isWindows) {
    return NATIVE_OVERLAY_BUTTON_WIDTH
  }

  // WSLg: same WCO feature, but the cluster sits further in — reserve to match.
  if (isWsl) {
    return WSL_OVERLAY_RESERVE_WIDTH
  }

  // macOS (traffic lights on the LEFT, reported via windowButtonPosition) and
  // plain Linux (no overlay — WCO disabled) need no right-edge reservation.
  return 0
}

module.exports = {
  NATIVE_OVERLAY_BUTTON_WIDTH,
  WSL_OVERLAY_RESERVE_WIDTH,
  nativeOverlayWidth
}
