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
