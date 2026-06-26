// Pre-layout fallback for WCO right-edge reservation (--titlebar-tools-right).
// Live width comes from navigator.windowControlsOverlay in the renderer.

const OVERLAY_FALLBACK_WIDTH = 144

/** @param {{ isWindows?: boolean, isWsl?: boolean }} opts */
function nativeOverlayWidth({ isWindows = false, isWsl = false } = {}) {
  return isWindows || isWsl ? OVERLAY_FALLBACK_WIDTH : 0
}

module.exports = { OVERLAY_FALLBACK_WIDTH, nativeOverlayWidth }
