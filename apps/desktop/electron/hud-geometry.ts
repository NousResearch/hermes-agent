export const HUD_WIDTH = 620
export const HUD_HEIGHT = 320
export const HUD_BOTTOM_MARGIN = 72

export interface HudWorkArea {
  height: number
  width: number
  x: number
  y: number
}

/** Return the display-aware default bounds used to recover the HUD layout. */
export function defaultHudBounds(area?: HudWorkArea): {
  height: number
  width: number
  x?: number
  y?: number
} {
  if (!area) {
    return { width: HUD_WIDTH, height: HUD_HEIGHT, x: undefined, y: undefined }
  }

  const width = Math.min(HUD_WIDTH, area.width)
  const height = Math.min(HUD_HEIGHT, area.height)

  return {
    width,
    height,
    x: Math.round(area.x + (area.width - width) / 2),
    y: Math.round(Math.max(area.y, area.y + area.height - height - HUD_BOTTOM_MARGIN))
  }
}

export interface HudBoundsWindow {
  isDestroyed(): boolean
  isResizable(): boolean
  setBounds(bounds: { height: number; width: number; x?: number; y?: number }): void
  setResizable(resizable: boolean): void
}

/** Apply recovery bounds and convert native-window failures into a result. */
export function applyHudResetBounds(
  win: HudBoundsWindow,
  bounds: { height: number; width: number; x?: number; y?: number }
): boolean {
  try {
    const wasResizable = win.isResizable()

    if (!wasResizable) {
      win.setResizable(true)
    }

    try {
      win.setBounds(bounds)
    } finally {
      if (!wasResizable && !win.isDestroyed()) {
        win.setResizable(false)
      }
    }

    return true
  } catch {
    return false
  }
}
