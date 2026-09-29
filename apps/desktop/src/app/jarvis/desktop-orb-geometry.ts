/**
 * Where the floating desktop orb may be, and how big. Pure, so the rules for
 * resizing around the centre, staying reachable and snapping to screen edges
 * are testable without a window.
 */

export interface Rect {
  height: number
  width: number
  x: number
  y: number
}

/** The orb window at scale 1: the sphere plus its caption and control capsule. */
export const ORB_BASE_SIZE = { height: 360, width: 320 }

export const ORB_SCALE_MIN = 0.5
export const ORB_SCALE_MAX = 1.6
export const ORB_SCALE_STEP = 0.1

/** The sizes the S / M / L shortcuts jump to. */
export const ORB_SCALE_PRESETS = [0.7, 1, 1.3] as const

export function clampOrbScale(scale: number): number {
  if (!Number.isFinite(scale)) {
    return 1
  }

  return Math.round(Math.min(ORB_SCALE_MAX, Math.max(ORB_SCALE_MIN, scale)) * 100) / 100
}

export function orbWindowSize(scale: number) {
  const s = clampOrbScale(scale)

  return { height: Math.round(ORB_BASE_SIZE.height * s), width: Math.round(ORB_BASE_SIZE.width * s) }
}

/** Resize keeping the orb's centre where it is, so growing never jumps the sphere sideways. */
export function resizeAroundCenter(bounds: Rect, size: { height: number; width: number }): Rect {
  return {
    ...size,
    x: Math.round(bounds.x + bounds.width / 2 - size.width / 2),
    y: Math.round(bounds.y + bounds.height / 2 - size.height / 2)
  }
}

/**
 * Keep enough of the orb on screen to grab it back: at least `visible` of the
 * window's width and height stays inside the work area.
 */
export function clampToArea(bounds: Rect, area: Rect, visible = 0.6): Rect {
  const keepW = bounds.width * visible
  const keepH = bounds.height * visible

  return {
    ...bounds,
    x: Math.round(Math.min(Math.max(bounds.x, area.x - (bounds.width - keepW)), area.x + area.width - keepW)),
    y: Math.round(Math.min(Math.max(bounds.y, area.y - (bounds.height - keepH)), area.y + area.height - keepH))
  }
}

/**
 * Magnetic edges: when the window comes within `threshold` px of a work-area
 * edge it settles `margin` px inside it. Reports whether anything snapped, so
 * the caller can give the click of feedback exactly then.
 */
export function snapToEdges(
  bounds: Rect,
  area: Rect,
  { margin = 12, threshold = 28 }: { margin?: number; threshold?: number } = {}
): { bounds: Rect; snapped: boolean } {
  let { x, y } = bounds
  let snapped = false

  const right = area.x + area.width - bounds.width
  const bottom = area.y + area.height - bounds.height

  if (Math.abs(x - area.x) <= threshold) {
    x = area.x + margin
    snapped = true
  } else if (Math.abs(x - right) <= threshold) {
    x = right - margin
    snapped = true
  }

  if (Math.abs(y - area.y) <= threshold) {
    y = area.y + margin
    snapped = true
  } else if (Math.abs(y - bottom) <= threshold) {
    y = bottom - margin
    snapped = true
  }

  return { bounds: { ...bounds, x, y }, snapped }
}

/** Scale implied by dragging the resize grip: distance from the centre, relative to where the drag began. */
export function scaleFromDrag(startScale: number, startDistance: number, distance: number): number {
  return startDistance > 0 ? clampOrbScale((startScale * distance) / startDistance) : clampOrbScale(startScale)
}
