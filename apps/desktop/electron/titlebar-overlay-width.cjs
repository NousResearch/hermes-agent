'use strict'

const OVERLAY_FALLBACK_WIDTH = 144
// Transparent WCO — renderer chrome shows through. rgba(0,0,0,0) can fall back
// to GetFrameColor() on some Electron builds; rgba(1,0,0,0) is the escape hatch.
const TITLEBAR_OVERLAY_COLOR = 'rgba(1, 0, 0, 0)'

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
function nativeOverlayWidth({ isMac = false } = {}) {
  if (isMac) return 0
  return OVERLAY_FALLBACK_WIDTH
}

// macOS Tahoe ships as Darwin 25 (Sequoia is 24); the Darwin number is truthful,
// unlike the product version which macOS reports as 16 or 26 depending on the
// build SDK.
const MACOS_TAHOE_DARWIN_MAJOR = 25

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
function macTitleBarOverlayHeight({ darwinMajor = 0, titlebarHeight = 0 } = {}) {
  return darwinMajor >= MACOS_TAHOE_DARWIN_MAJOR ? 0 : titlebarHeight
}

/**
 * Options to pass to BrowserWindow `titleBarOverlay`.
 *
 * WSLg needs the Electron overlay just like native Windows and plain Linux.
 * Disabling it while using `titleBarStyle: "hidden"` leaves a frameless window
 * with no visible min/max/close buttons on common WSLg builds.
 *
 * @param {{ isMac?: boolean, isWindows?: boolean, isWsl?: boolean, darwinMajor?: number, titlebarHeight?: number, symbolColor?: string }} opts
 */
function titleBarOverlayOptions({
  isMac = false,
  isWindows = false,
  isWsl = false,
  darwinMajor = 0,
  titlebarHeight = 0,
  symbolColor = '#242424'
} = {}) {
  void isWindows
  void isWsl

  if (isMac) {
    return { height: macTitleBarOverlayHeight({ darwinMajor, titlebarHeight }) }
  }

  return {
    color: TITLEBAR_OVERLAY_COLOR,
    height: titlebarHeight,
    symbolColor
  }
}

module.exports = {
  MACOS_TAHOE_DARWIN_MAJOR,
  OVERLAY_FALLBACK_WIDTH,
  TITLEBAR_OVERLAY_COLOR,
  macTitleBarOverlayHeight,
  nativeOverlayWidth,
  titleBarOverlayOptions
}
