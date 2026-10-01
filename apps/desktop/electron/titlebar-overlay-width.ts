export const OVERLAY_FALLBACK_WIDTH = 144

interface TitleBarOverlayOptionsInput {
  platform?: 'linux' | 'mac' | 'windows' | 'wslg'
  darwinMajor?: number
  titlebarHeight?: number
  color?: string
  foreground?: string | null
  dark?: boolean
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
 * With `titleBarStyle: hidden`, Linux shows no window controls unless an
 * overlay object is provided. Windows and WSLg return false so the renderer
 * paints caption buttons whose hover stays inside the navbar.
 */
export function titleBarOverlayOptions({
  platform = 'linux',
  darwinMajor = 0,
  titlebarHeight = 0,
  color,
  foreground,
  dark = false
}: TitleBarOverlayOptionsInput = {}) {
  // Electron's Linux overlay keeps a narrow, unscaled three-button cluster
  // under WSLg. The renderer owns the controls there while the host's RAIL
  // local-move path continues to own edge dragging and Snap.
  // Windows uses the same renderer buttons so the hover stays inside the navbar
  // instead of filling the native caption height.
  if (platform === 'wslg' || platform === 'windows') {
    return false
  }

  if (platform === 'mac') {
    return { height: macTitleBarOverlayHeight({ darwinMajor, titlebarHeight }) }
  }

  return {
    color,
    height: titlebarHeight,
    symbolColor: foreground || (dark ? '#f7f7f7' : '#242424')
  }
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
