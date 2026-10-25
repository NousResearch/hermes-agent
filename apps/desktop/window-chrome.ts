/** Shared CSS titlebar geometry; native coordinates are device-independent pixels. */
export const MACOS_TITLEBAR_HEIGHT = 44
export const MACOS_WINDOW_BUTTON_X = 16
export const MACOS_TRAFFIC_LIGHTS_HEIGHT = 14
export const MACOS_TRAFFIC_LIGHTS_WIDTH = 58

export const desktopTitlebarHeight = (isMac: boolean): number => isMac ? MACOS_TITLEBAR_HEIGHT : 34

export function macWindowButtonPosition(zoomFactor = 1) {
  return {
    x: MACOS_WINDOW_BUTTON_X * zoomFactor,
    y: (MACOS_TITLEBAR_HEIGHT * zoomFactor - MACOS_TRAFFIC_LIGHTS_HEIGHT) / 2
  }
}

interface NativeTitlebarWindow {
  isFullScreen(): boolean
  getWindowButtonPosition(): { x: number; y: number }
  webContents: { getZoomFactor(): number }
}

/** Native window coordinates must be converted before renderer layout uses them. */
export function getWindowButtonPosition(win: NativeTitlebarWindow | null, isMac: boolean) {
  if (!isMac || win?.isFullScreen()) {
    return null
  }

  const position = win?.getWindowButtonPosition() ?? macWindowButtonPosition()
  const zoom = win?.webContents.getZoomFactor() ?? 1

  return { x: position.x / zoom, y: position.y / zoom }
}
