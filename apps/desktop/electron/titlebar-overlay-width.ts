export const OVERLAY_FALLBACK_WIDTH = 144

interface TitleBarOverlayOptionsInput {
  platform?: 'linux' | 'mac' | 'windows' | 'wslg'
  darwinMajor?: number
  titlebarHeight?: number
  color?: string
  foreground?: string | null
  dark?: boolean
  /** Current webContents zoom factor (1 = 100%); scales the overlay height. */
  zoomFactor?: number
}

/**
 * Static pre-layout reservation (px) for the right-side native window-controls
 * overlay (min/max/close). Only a FALLBACK — once laid out the renderer reads
 * the exact width from navigator.windowControlsOverlay
 * (use-window-controls-overlay-width.ts) and uses this value only when the WCO
 * API is unavailable.
 *
 * macOS uses traffic lights positioned via trafficLightPosition, not a WCO
 * overlay, so it reserves nothing here. Every other desktop platform reserves
 * the same right-side footprint: Electron paints it on Windows/plain Linux,
 * while the renderer paints larger controls on WSLg.
 *
 * @param {{ isMac?: boolean }} opts
 */
export function nativeOverlayWidth({ isWindows = false, isWsl = false, isMac = false } = {}) {
  if (isMac) {
    return 0
  }

  return OVERLAY_FALLBACK_WIDTH
}

/**
 * Build Electron's Window Controls Overlay options for every desktop host.
 * With `titleBarStyle: hidden`, Windows and Linux show no window controls
 * unless an overlay object is provided. WSLg deliberately returns false so
 * the renderer can paint correctly scaled Windows-style controls instead.
 */
export function titleBarOverlayOptions({
  platform = 'linux',
  darwinMajor = 0,
  titlebarHeight = 0,
  color,
  foreground,
  dark = false,
  zoomFactor = 1
}: TitleBarOverlayOptionsInput = {}) {
  // Electron's Linux overlay keeps a narrow, unscaled three-button cluster
  // under WSLg. The renderer owns larger Windows-shaped controls there while
  // the host's RAIL local-move path continues to own edge dragging and Snap.
  if (platform === 'wslg') {
    return false
  }

  if (platform === 'mac') {
    return { height: macTitleBarOverlayHeight({ darwinMajor, titlebarHeight }) }
  }

  return {
    color,
    height: scaledOverlayHeight(titlebarHeight, zoomFactor),
    symbolColor: foreground || (dark ? '#f7f7f7' : '#242424')
  }
}

/**
 * Scale the native window-controls overlay height with the webContents zoom
 * factor. Chromium scales the page (including the renderer's titlebar) but
 * not the native WCO buttons, so at any zoom other than 100% the fixed-height
 * overlay mismatches the titlebar it sits inside (#81086). Rounds to whole
 * pixels (the overlay API is integer-based) and never returns less than 1.
 */
export function scaledOverlayHeight(titlebarHeight: number, zoomFactor: number): number {
  if (!Number.isFinite(titlebarHeight) || titlebarHeight <= 0) {
    return titlebarHeight
  }

  const factor = Number.isFinite(zoomFactor) && zoomFactor > 0 ? zoomFactor : 1

  return Math.max(1, Math.round(titlebarHeight * factor))
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
