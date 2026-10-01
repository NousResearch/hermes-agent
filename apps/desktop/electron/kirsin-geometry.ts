// Default + persisted geometry for the Kirsin Agent Window.
//
// The Kirsin window is "a HUD pinned to the `kirsin` profile, with a
// Kirsin-branded shell" — a persistent always-on-top floating chat. Its
// geometry helpers mirror hud-geometry.ts: a display-aware default bounds for
// first spawn / off-screen recovery, a validator for renderer-provided resize
// geometry, and a reset-bounds applier that briefly flips `resizable` on (the
// window is created non-resizable, which on Windows/Linux also blocks
// programmatic setBounds sizing — same reason as the HUD).

export const KIRSIN_WIDTH = 576
export const KIRSIN_HEIGHT = 560
// The compact states the renderer morphs to (see kirsin-shell.tsx). The
// window is created non-resizable, so the set-bounds channel is driven ONLY by
// the shell's morphs (pill collapse + orb morph) — never by a user edge-resize.
// The floors therefore just need to admit the two collapsed shapes: the 96px
// pill (height) and the 196px orb (width + height — the 150px painted circle
// plus its glow margin, see the "Orb morph" block in styles.css).
export const KIRSIN_MIN_HEIGHT = 96
export const KIRSIN_MIN_WIDTH = 196

export interface KirsinWorkArea {
  height: number
  width: number
  x: number
  y: number
}

/** Display-aware default bounds used to spawn and recover the Kirsin window. */
export function defaultKirsinBounds(area?: KirsinWorkArea): {
  height: number
  width: number
  x?: number
  y?: number
} {
  if (!area) {
    return { width: KIRSIN_WIDTH, height: KIRSIN_HEIGHT, x: undefined, y: undefined }
  }

  const width = Math.min(KIRSIN_WIDTH, area.width)
  const height = Math.min(KIRSIN_HEIGHT, area.height)

  // A chat panel parks in the top-right: it never covers the app's
  // left-anchored sidebar/composer, and it sits clear of the taskbar.
  return {
    width,
    height,
    x: Math.round(area.x + area.width - width - 24),
    y: Math.round(area.y + 24)
  }
}

export interface KirsinBoundsWindow {
  isDestroyed(): boolean
  isResizable(): boolean
  setBounds(bounds: { height: number; width: number; x?: number; y?: number }): void
  setResizable(resizable: boolean): void
}

export interface KirsinResizeBounds {
  height: number
  width: number
  x: number
  y: number
}

/** Validate renderer-provided resize geometry before it reaches native APIs. */
export function normalizeKirsinResizeBounds(value: unknown): KirsinResizeBounds | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const candidate = value as Partial<Record<keyof KirsinResizeBounds, unknown>>
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
    width: Math.max(KIRSIN_MIN_WIDTH, Math.round(width)),
    height: Math.max(KIRSIN_MIN_HEIGHT, Math.round(height))
  }
}

/**
 * Apply recovery bounds. The Kirsin window is created non-resizable (a
 * transparent frameless window must not expose a system resize hot-zone), which
 * on Windows/Linux also blocks programmatic setBounds sizing — flip it on
 * briefly, exactly as the HUD does.
 */
export function applyKirsinResetBounds(
  win: KirsinBoundsWindow,
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
