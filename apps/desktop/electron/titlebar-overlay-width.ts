export const OVERLAY_FALLBACK_WIDTH = 144

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
 * reserve the fallback width — the split is simply mac vs. not.
 *
 * @param {{ isMac?: boolean }} opts
 */
export function nativeOverlayWidth({ isWindows = false, isWsl = false, isMac = false } = {}) {
  if (isMac) {
    return 0
  }

  return OVERLAY_FALLBACK_WIDTH
}

// macOS Tahoe ships as Darwin 25 (Sequoia is 24); the Darwin number is truthful,
// unlike the product version which macOS reports as 16 or 26 depending on the
// build SDK.
export const MACOS_TAHOE_DARWIN_MAJOR = 25

/**
 * Height (px) to pass to `titleBarOverlay` on macOS. Tahoe (Darwin 25+)
 * miscalculates the native traffic-light position when the overlay carries a
 * nonzero height (electron#49183), shoving the lights into the left titlebar
 * tools. Return 0 there so `setWindowButtonPosition` lands them at the configured
 * inset; the renderer paints its own drag strips, so nothing is lost. Pre-Tahoe
 * keeps the full titlebar height, byte-identical.
 *
 * @param {{ darwinMajor?: number, titlebarHeight?: number }} opts
 */
export function macTitleBarOverlayHeight({ darwinMajor = 0, titlebarHeight = 0 } = {}) {
  return darwinMajor >= MACOS_TAHOE_DARWIN_MAJOR ? 0 : titlebarHeight
}

/**
 * Height (px) to pass to `titleBarOverlay` on Windows/Linux (WCO). Page zoom
 * (webContents.setZoomLevel) scales the renderer's own DOM but never the
 * native min/max/close glyphs the OS paints into the overlay region, so at
 * higher zoom levels the app's own titlebar content grows while those native
 * controls stay pinned at the un-zoomed size (#81086). Scaling the overlay
 * height by the same factor keeps the native buttons in proportion with the
 * rest of the UI.
 *
 * @param {{ titlebarHeight?: number, zoomFactor?: number }} opts
 */
export function scaledTitleBarOverlayHeight({ titlebarHeight = 0, zoomFactor = 1 } = {}) {
  return Math.round(titlebarHeight * zoomFactor)
}
