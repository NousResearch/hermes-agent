export const HUD_WIDTH = 620
export const HUD_HEIGHT = 320
export const HUD_BOTTOM_MARGIN = 72

export interface HudWorkArea {
  height: number
  width: number
  x: number
  y: number
}

/**
 * Where a HUD with no remembered geometry parks.
 *
 * `bottom-center` is the standing default (a game-chat frame belongs centered
 * above the taskbar). `bottom-right` is the tray's entry point: clicking the
 * icon near the clock should raise the bar in the corner the click came from,
 * clear of the taskbar and of whatever the user has parked at the center.
 * Only the FIRST open chooses — once the user moves or resizes it,
 * hud-state.json wins either way.
 */
export type HudBoundsAnchor = 'bottom-center' | 'bottom-right'

/** Display-aware default bounds used to spawn and recover the HUD layout. */
export function defaultHudBounds(
  area?: HudWorkArea,
  { anchor = 'bottom-center' }: { anchor?: HudBoundsAnchor } = {}
): {
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
    x: Math.round(
      anchor === 'bottom-right'
        ? // Clamped at the left edge: a display narrower than the bar plus its
          // margins would otherwise place x off-screen, and a window spawned
          // outside the visible area is a window the user cannot reach.
          Math.max(area.x, area.x + area.width - width - HUD_BOTTOM_MARGIN)
        : area.x + (area.width - width) / 2
    ),
    y: Math.round(Math.max(area.y, area.y + area.height - height - HUD_BOTTOM_MARGIN))
  }
}

export interface HudBoundsWindow {
  isDestroyed(): boolean
  isResizable(): boolean
  setBounds(bounds: { height: number; width: number; x?: number; y?: number }): void
  setResizable(resizable: boolean): void
}

export interface HudResizeBounds {
  height: number
  width: number
  x: number
  y: number
}

/** Validate renderer-provided resize geometry before it reaches native APIs. */
export function normalizeHudResizeBounds(value: unknown): HudResizeBounds | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const candidate = value as Partial<Record<keyof HudResizeBounds, unknown>>
  const x = Number(candidate.x)
  const y = Number(candidate.y)
  const width = Number(candidate.width)
  const height = Number(candidate.height)

  if (![x, y, width, height].every(Number.isFinite)) {
    return null
  }

  return {
    x: Math.round(x),
    y: Math.round(y),
    width: Math.max(380, Math.round(width)),
    height: Math.max(160, Math.round(height))
  }
}

/**
 * Apply recovery bounds. The HUD is created non-resizable (a transparent
 * frameless window must not expose a system resize hot-zone), which on
 * Windows/Linux also blocks programmatic setBounds sizing — flip it on
 * briefly, same as the corner-resize IPC.
 *
 * On native Wayland the compositor ignores the position half of setBounds
 * (clients cannot place themselves). Size still applies, which is what
 * unsticks the tall/narrow persisted geometry from #87055.
 */
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
