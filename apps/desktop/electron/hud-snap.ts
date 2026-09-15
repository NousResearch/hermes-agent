/**
 * HUD snap-to-pointer math — where to park the window so a chosen anchor on
 * the bar (usually the composer center) sits under the OS cursor.
 *
 * Kept pure so the two unit mismatches (screen vs window origin, DIP vs CSS)
 * cannot drift from the Linux cursor feed in hud-cursor.ts.
 */

interface Point {
  x: number
  y: number
}

interface Rect {
  x: number
  y: number
  width: number
  height: number
}

/**
 * Window top-left in screen space (DIP) that places `anchor` — a point in the
 * window's CSS pixels — under `cursor` (screen DIP).
 */
export function windowOriginForCursorAnchor(cursor: Point, anchor: Point, zoomFactor: number): Point {
  const scale = Number.isFinite(zoomFactor) && zoomFactor > 0 ? zoomFactor : 1

  return {
    x: Math.round(cursor.x - anchor.x * scale),
    y: Math.round(cursor.y - anchor.y * scale)
  }
}

/**
 * Keep as much of the window on-screen as possible while preserving the anchor
 * under the cursor when there is room; when there is not, clamp the origin so
 * at least a sliver of the window stays in the work area.
 */
export function clampHudOrigin(origin: Point, windowSize: Pick<Rect, 'width' | 'height'>, workArea: Rect): Point {
  const minVisible = 40
  const maxX = workArea.x + workArea.width - minVisible
  const minX = workArea.x + minVisible - windowSize.width
  const maxY = workArea.y + workArea.height - minVisible
  const minY = workArea.y + minVisible - windowSize.height

  return {
    x: Math.min(Math.max(origin.x, minX), maxX),
    y: Math.min(Math.max(origin.y, minY), maxY)
  }
}

export function snapHudBounds(
  cursor: Point,
  anchor: Point,
  windowSize: Pick<Rect, 'width' | 'height'>,
  zoomFactor: number,
  workArea: Rect
): Point {
  return clampHudOrigin(windowOriginForCursorAnchor(cursor, anchor, zoomFactor), windowSize, workArea)
}

/** Magnetic flush at drag-end / snap-end — not during live tracking. */
export const HUD_DOCK_THRESHOLD = 24

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isFinitePoint(value: unknown): value is Point {
  return typeof value === 'object' && value !== null && isFiniteNumber((value as Point).x) && isFiniteNumber((value as Point).y)
}

function isFiniteSize(value: unknown): value is Pick<Rect, 'width' | 'height'> {
  return (
    typeof value === 'object' &&
    value !== null &&
    isFiniteNumber((value as Rect).width) &&
    isFiniteNumber((value as Rect).height)
  )
}

/**
 * If the window sits within `threshold` DIP of a work-area edge, flush that
 * axis to the edge. Ties prefer the vertical edge (top/bottom). Far from every
 * edge, or given non-finite geometry, the origin is unchanged (fail-open).
 */
export function dockHudToNearestEdge(
  origin: Point,
  windowSize: Pick<Rect, 'width' | 'height'>,
  workArea: Rect,
  threshold = HUD_DOCK_THRESHOLD
): Point {
  if (!isFinitePoint(origin) || !isFiniteSize(windowSize) || !isFinitePoint(workArea) || !isFiniteSize(workArea) || !isFiniteNumber(threshold)) {
    return origin
  }

  const top = origin.y - workArea.y
  const bottom = workArea.y + workArea.height - origin.y - windowSize.height
  const left = origin.x - workArea.x
  const right = workArea.x + workArea.width - origin.x - windowSize.width

  const vertical = top <= bottom ? top : bottom
  const horizontal = left <= right ? left : right

  if (vertical > threshold && horizontal > threshold) {
    return origin
  }

  const docked = { x: origin.x, y: origin.y }

  // Equal distances: flush the vertical axis (users park on the top/bottom).
  if (vertical <= threshold && vertical <= horizontal) {
    docked.y = top <= bottom ? workArea.y : workArea.y + workArea.height - windowSize.height
  } else {
    docked.x = left <= right ? workArea.x : workArea.x + workArea.width - windowSize.width
  }

  return clampHudOrigin(docked, windowSize, workArea)
}
