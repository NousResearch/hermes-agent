'use strict'

const OVERLAY_FALLBACK_WIDTH = 144

/**
 * Static pre-layout reservation (px) for the right-side native window-controls
 * overlay (min/max/close). Only a FALLBACK — once laid out the renderer reads
 * the exact width from navigator.windowControlsOverlay
 * (use-window-controls-overlay-width.ts) and uses this value only when the WCO
 * API is unavailable.
 *
 * macOS uses traffic lights positioned via trafficLightPosition, not a WCO
 * overlay, so it reserves nothing here. Every other desktop platform now paints
 * the Electron overlay (Windows, WSLg, and plain Linux KDE/GNOME), so they all
 * reserve the fallback width.
 *
 * @param {{ isWindows?: boolean, isWsl?: boolean, isMac?: boolean }} opts
 */
function nativeOverlayWidth({ isWindows = false, isWsl = false, isMac = false } = {}) {
  if (isMac) return 0
  return OVERLAY_FALLBACK_WIDTH
}

module.exports = { OVERLAY_FALLBACK_WIDTH, nativeOverlayWidth }
