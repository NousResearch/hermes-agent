export interface PetOverlayDesktopBounds {
  height: number
  width: number
  x: number
  y: number
}

/**
 * `get-windows` reads Win32 `GetWindowRect`, which reports physical pixels in
 * Electron's per-monitor-DPI-aware process. BrowserWindow and `screen` bounds
 * use DIP coordinates, so leaving these unconverted makes a ledge drift farther
 * below its real window as its physical y coordinate increases.
 */
export function overlayWindowBoundsToDip(
  bounds: PetOverlayDesktopBounds,
  platform: NodeJS.Platform,
  screenToDipRect: (bounds: PetOverlayDesktopBounds) => PetOverlayDesktopBounds
): PetOverlayDesktopBounds {
  return platform === 'win32' ? screenToDipRect(bounds) : bounds
}
