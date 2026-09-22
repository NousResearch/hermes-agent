/**
 * Floating-pane geometry — pure, so the clamping/anchoring rules are testable
 * without a DOM.
 *
 * A floating pane is NOT a track in the layout tree: it never takes width from
 * a zone. It's a fixed-position card the tree renders above itself. Its whole
 * contract is "stay a sane rect inside the viewport", which is this module.
 */

/** Corner a floating pane spawns from (and re-anchors to on reset). */
export type FloatingAnchor = 'bottom-left' | 'bottom-right' | 'top-left' | 'top-right'

export interface FloatingRect {
  x: number
  y: number
  width: number
  height: number
}

export interface FloatingViewport {
  width: number
  height: number
  /** Chrome reserved at the top (the 34px titlebar) — panes never cover it. */
  top: number
}

/** Keep this much of the card on screen when clamping, so it stays grabbable. */
const MIN_VISIBLE = 48

/** Gap between a spawned pane and the viewport edge it anchors to. */
export const FLOATING_MARGIN = 12

/** Smallest a floating card can be resized to before it stops being usable. */
export const MIN_FLOATING_WIDTH = 160
export const MIN_FLOATING_HEIGHT = 96

/** The one non-tiling placement — see renderer/floating-panes.tsx. */
export const FLOATING_PLACEMENT = 'floating'

export const clamp = (n: number, lo: number, hi: number): number => Math.min(Math.max(n, lo), hi)

/**
 * Clamp a rect into the viewport. Horizontal keeps `MIN_VISIBLE` px on screen
 * from either edge (a card can hang off the right, never vanish); vertical is
 * hard-bounded by the reserved chrome so the drag handle is always reachable.
 *
 * Both axes clamp `lo` before `hi`, so a card wider/taller than the viewport
 * pins to the top-left rather than inverting. The position edge rules live in
 * `clampPosition` — every mutation (drag, reflow, resize) must share them.
 */
export function clampFloatingRect(rect: FloatingRect, viewport: FloatingViewport): FloatingRect {
  return { ...rect, ...clampPosition(rect.x, rect.y, rect.width, rect.height, viewport) }
}

/** Spawn position for an anchor — the corner, inset by `FLOATING_MARGIN`. */
export function anchoredRect(
  anchor: FloatingAnchor,
  size: { width: number; height: number },
  viewport: FloatingViewport
): FloatingRect {
  const right = anchor === 'bottom-right' || anchor === 'top-right'
  const bottom = anchor === 'bottom-left' || anchor === 'bottom-right'

  const rect = {
    ...size,
    x: right ? viewport.width - size.width - FLOATING_MARGIN : FLOATING_MARGIN,
    y: bottom ? viewport.height - size.height - FLOATING_MARGIN : viewport.top + FLOATING_MARGIN
  }

  return clampFloatingRect(rect, viewport)
}

/**
 * Clamp a resized card: at least `MIN_FLOATING_WIDTH/HEIGHT` on each axis, at
 * most the viewport extent. The position is clamped with the NEW size so a
 * corner resize that grows past the viewport edge stops at the edge instead
 * of dragging the whole card off screen (a right-anchored pane's x is fixed
 * by the card's own coordinate frame — see the resize handler).
 */
export function clampFloatingRectSize(
  rect: FloatingRect,
  viewport: FloatingViewport
): FloatingRect {
  const width = clamp(rect.width, MIN_FLOATING_WIDTH, viewport.width)
  const height = clamp(rect.height, MIN_FLOATING_HEIGHT, viewport.height)

  return { ...rect, width, height, ...clampPosition(rect.x, rect.y, width, height, viewport) }
}

/**
 * The x/y clamp shared by every rect mutation. Extracted so the resize clamp
 * can re-run it with a changed size without duplicating the edge rules.
 */
export function clampPosition(
  x: number,
  y: number,
  width: number,
  height: number,
  viewport: FloatingViewport
): { x: number; y: number } {
  const maxX = Math.max(MIN_VISIBLE - width, viewport.width - MIN_VISIBLE)
  const maxY = Math.max(viewport.top, viewport.height - MIN_VISIBLE)

  return {
    x: clamp(x, Math.min(MIN_VISIBLE - width, maxX), maxX),
    y: clamp(y, viewport.top, maxY)
  }
}

/**
 * Re-clamp on viewport resize. A pane anchored to a right/bottom edge TRACKS
 * that edge (shrinking the window keeps it in the corner) instead of being
 * dragged inward only when it would fall off — matching how the pet and every
 * OS HUD behave.
 */
export function reflowRect(
  rect: FloatingRect,
  anchor: FloatingAnchor,
  previous: FloatingViewport,
  next: FloatingViewport
): FloatingRect {
  const right = anchor === 'bottom-right' || anchor === 'top-right'
  const bottom = anchor === 'bottom-left' || anchor === 'bottom-right'

  return clampFloatingRect(
    {
      ...rect,
      x: right ? rect.x + (next.width - previous.width) : rect.x,
      y: bottom ? rect.y + (next.height - previous.height) : rect.y
    },
    next
  )
}

/** Parse an authored CSS px length (`'216px'`, `216`) with a fallback. */
export function floatingPx(value: number | string | undefined, fallback: number): number {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : fallback
  }

  const parsed = Number.parseFloat(value ?? '')

  return Number.isFinite(parsed) ? parsed : fallback
}
