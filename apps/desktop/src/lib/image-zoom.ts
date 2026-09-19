/**
 * Zoom + pan math for the image lightbox.
 *
 * Pure and DOM-free so the behaviour can be tested directly: the component
 * owns gestures and rendering, this owns what the numbers must be.
 *
 * The lightbox opens fit-to-screen, so `MIN_ZOOM` is 1 — below that the image
 * would shrink inside an already-fitted box for nothing. Zooming out means
 * walking back toward the fit, not past it.
 */

/**
 * Does this source redraw at any scale, or is it a fixed grid of pixels?
 *
 * It decides HOW zoom is applied. A `transform: scale()` magnifies the bitmap
 * the browser already rasterized at layout size, so a vector blurs exactly like
 * a photo would. Growing the layout box instead makes the engine re-render the
 * vector at the new size — the whole reason to keep a diagram in SVG.
 */
export function isVectorSource(src: string | undefined): boolean {
  if (!src) return false
  const value = src.trim().toLowerCase()
  if (value.startsWith('data:')) return value.startsWith('data:image/svg+xml')
  // Strip any query/fragment before testing the extension.
  return value.split(/[?#]/)[0].endsWith('.svg')
}

/**
 * Is this event coming from somewhere the user is typing?
 *
 * A zoom shortcut must never swallow a character the user meant to type. `+`
 * is a plain character, so an image surface that binds it globally steals it
 * from the composer, from search fields, from every text input on screen.
 */
export function isTypingTarget(target: EventTarget | null): boolean {
  if (!target || !(target instanceof HTMLElement)) return false
  if (target.isContentEditable) return true
  const tag = target.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
}

export const MIN_ZOOM = 1
export const MAX_ZOOM = 8

/** One button press / keyboard step. */
export const ZOOM_STEP = 1.25

export interface Point {
  x: number
  y: number
}

export interface Size {
  height: number
  width: number
}

export function clampZoom(scale: number): number {
  if (!Number.isFinite(scale)) return MIN_ZOOM
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, scale))
}

export function zoomBy(scale: number, factor: number): number {
  return clampZoom(scale * factor)
}

/**
 * Wheel delta to a zoom factor.
 *
 * Exponential in the delta so a trackpad's many small events and a mouse's few
 * large ones travel at the same rate per unit scrolled, and so repeated zooming
 * feels linear to the eye (each notch is a constant *ratio*, not a constant
 * addend).
 */
export function zoomFromWheel(scale: number, deltaY: number): number {
  if (!Number.isFinite(deltaY) || deltaY === 0) return clampZoom(scale)
  return clampZoom(scale * Math.exp(-deltaY / 320))
}

/**
 * How far the image may travel from center on one axis.
 *
 * Only the overflow is reachable: at fit (scale 1) there is none, so panning is
 * pinned to center and the image can never be dragged off-screen.
 */
export function maxPanOffset(baseSize: number, scale: number): number {
  if (!Number.isFinite(baseSize) || baseSize <= 0) return 0
  return Math.max(0, (baseSize * clampZoom(scale) - baseSize) / 2)
}

export function clampPan(offset: Point, base: Size, scale: number): Point {
  const maxX = maxPanOffset(base.width, scale)
  const maxY = maxPanOffset(base.height, scale)
  return {
    x: Math.min(maxX, Math.max(-maxX, offset.x)),
    y: Math.min(maxY, Math.max(-maxY, offset.y))
  }
}

/**
 * Keep the point under the cursor pinned while the scale changes.
 *
 * Without this, zooming always pulls toward the image's center and the detail
 * being inspected slides away — the difference between reading a diagram and
 * chasing it. `cursor` is relative to the container's center.
 */
export function panForZoomAtPoint(
  cursor: Point,
  offset: Point,
  prevScale: number,
  nextScale: number
): Point {
  if (prevScale <= 0) return offset
  const ratio = nextScale / prevScale
  return {
    x: cursor.x - (cursor.x - offset.x) * ratio,
    y: cursor.y - (cursor.y - offset.y) * ratio
  }
}

/** Zoom toward a focal point and land inside the pan bounds in one step. */
export function zoomAtPoint(
  cursor: Point,
  offset: Point,
  prevScale: number,
  nextScale: number,
  base: Size
): { offset: Point; scale: number } {
  const scale = clampZoom(nextScale)
  return {
    offset: clampPan(panForZoomAtPoint(cursor, offset, prevScale, scale), base, scale),
    scale
  }
}
