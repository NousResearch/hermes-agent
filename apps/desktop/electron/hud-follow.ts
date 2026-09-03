/**
 * HUD follow-the-pointer — the lazy-follow math.
 *
 * "Follow" is NOT "glued to the cursor". A bar welded to the pointer covers
 * whatever the user is pointing at and runs away the moment they reach for
 * it. Lazy follow keeps the bar a fixed offset from the cursor, eases toward
 * that spot, and holds still whenever the cursor is near enough to touch it —
 * so it trails the user around the screen and stays reachable at rest.
 *
 * Pure so the three unit traps (screen vs window origin, DIP vs CSS, work-area
 * clamping) are tested once and cannot drift from the snap chord's math in
 * hud-snap.ts, which this builds on.
 */

import { clampHudOrigin, windowOriginForCursorAnchor } from './hud-snap'

export interface Point {
  x: number
  y: number
}

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

export interface HudFollowInput {
  /** OS cursor, screen DIP. */
  cursor: Point
  /** Current window bounds, screen DIP. */
  bounds: Rect
  /** Height of the visible bar at the top of the window (headroom + composer),
   *  DIP. The rest of the window is transparent band and must not count for
   *  reach or for fitting on screen. */
  barHeight: number
  /** Work area of the display the cursor is on. */
  workArea: Rect
  /** Page zoom of the HUD renderer (CSS px → DIP scale). */
  zoomFactor: number
  /** Offset from the pointer to the bar's near corner, DIP — the bar parks
   *  just to the lower-right of the pointer, like a context menu. */
  gap: number
  /** The cursor is "within reach" while it is inside the window rect grown by
   *  this many DIP on every side; the bar holds still there. */
  reach: number
  /** Cursor wobble smaller than this (DIP) from the settled target is ignored. */
  deadZone: number
  /** Fraction of the remaining distance covered per tick, 0 < ease ≤ 1. */
  ease: number
  /** The target the bar last settled on, or null on the first tick. */
  lastTarget: Point | null
  /** The fractional origin the bar was last steered to, or null to start
   *  from the window's real (integer) bounds. Easing from integers re-rounds
   *  every tick and reads as stepping; easing from the fraction is smooth. */
  position?: Point | null
}

export interface HudFollowStep {
  /** Window origin to apply this tick, or null to leave the window alone. */
  origin: Point | null
  /** The settled target (fed back as `lastTarget` next tick). */
  target: Point | null
  /** Why the window stayed put, when it did. */
  hold: 'in-reach' | 'settled' | null
  /** The fractional origin to feed back as `position` next tick. */
  position: Point | null
}

export const HUD_FOLLOW_GAP = 18
export const HUD_FOLLOW_REACH = 40
/** Headroom (pets) + composer bar, DIP. Main cannot measure the renderer's
 *  bar, so follow assumes this; a few px off only shifts the flip point. */
export const HUD_FOLLOW_BAR_HEIGHT = 130
export const HUD_FOLLOW_DEAD_ZONE = 12
/** Per-tick fraction at 60 Hz — the same ~0.3 s settle a 0.28 step had at 30 Hz. */
export const HUD_FOLLOW_EASE = 0.16
/** Cursor poll cadence while following. 60 Hz matches the compositor, so the
 *  bar moves once per frame instead of every other frame; each tick is one
 *  `getCursorScreenPoint()` plus at most one `setBounds`. */
export const HUD_FOLLOW_TICK_MS = 16

function within(point: Point, rect: Rect, margin: number): boolean {
  return (
    point.x >= rect.x - margin &&
    point.x < rect.x + rect.width + margin &&
    point.y >= rect.y - margin &&
    point.y < rect.y + rect.height + margin
  )
}

/**
 * Where the bar should rest for a given pointer: its top-left corner `gap`
 * to the lower-right of the pointer, the way a context menu opens. Near the
 * right edge the bar flips to the pointer's left; near the bottom the BAR
 * (not the whole window — the band below it is transparent) flips above.
 * Either way the pointer stays just outside the bar, inside the reach band,
 * so a resting pointer can slide onto the bar without it moving away.
 */
export function hudFollowTarget(
  cursor: Point,
  size: Pick<Rect, 'width' | 'height'>,
  barHeight: number,
  zoomFactor: number,
  workArea: Rect,
  gap: number
): Point {
  const scale = Number.isFinite(zoomFactor) && zoomFactor > 0 ? zoomFactor : 1
  const off = gap * scale
  const bar = Math.min(size.height, Math.max(1, barHeight * scale))

  let x = cursor.x + off
  let y = cursor.y + off

  if (x + size.width > workArea.x + workArea.width) {
    x = cursor.x - off - size.width
  }

  if (y + bar > workArea.y + workArea.height) {
    y = cursor.y - off - bar
  }

  const origin = windowOriginForCursorAnchor({ x, y }, { x: 0, y: 0 }, 1)

  return clampHudOrigin(origin, size, workArea)
}

export function hudFollowStep(input: HudFollowInput): HudFollowStep {
  const { cursor, bounds, barHeight, workArea, zoomFactor, gap, reach, deadZone, ease, lastTarget } = input
  const from = input.position ?? { x: bounds.x, y: bounds.y }
  const scale = Number.isFinite(zoomFactor) && zoomFactor > 0 ? zoomFactor : 1
  const barRect = { x: bounds.x, y: bounds.y, width: bounds.width, height: Math.min(bounds.height, barHeight * scale) }

  // Reachable: the pointer is ON the bar (dragging, resizing, clicking), or
  // it is within reach of a bar that has already come to rest — resting
  // beside it, or coming for it. Nothing moves then. A bar still travelling
  // keeps travelling even when the pointer is near, otherwise it would stop
  // short of its spot; and only a RESTING bar holds for an approaching
  // pointer, otherwise the approach itself would retarget it and it would
  // back away as the pointer came. The band below the bar is transparent
  // and does not count — a pointer moving through it is moving away.
  const onBar = within(cursor, barRect, 0)
  const nearBar = within(cursor, barRect, reach)
  const resting = lastTarget !== null && Math.hypot(from.x - lastTarget.x, from.y - lastTarget.y) < 1

  if (onBar || (nearBar && resting)) {
    return { origin: null, target: lastTarget, hold: 'in-reach', position: null }
  }

  let target = hudFollowTarget(cursor, bounds, barHeight, zoomFactor, workArea, gap)

  // Hand jitter must not turn into a bar that never stops twitching.
  if (lastTarget && Math.hypot(target.x - lastTarget.x, target.y - lastTarget.y) < deadZone) {
    target = lastTarget
  }

  const dx = target.x - from.x
  const dy = target.y - from.y
  const remaining = Math.hypot(dx, dy)

  if (remaining < 1) {
    return { origin: null, target, hold: 'settled', position: null }
  }

  const k = Number.isFinite(ease) && ease > 0 && ease <= 1 ? ease : 1

  // Close enough: land exactly, so the bar cannot spend a second creeping the
  // last pixel and the geometry persistence sees one final position.
  const position = remaining <= 2 ? target : { x: from.x + dx * k, y: from.y + dy * k }
  const origin = { x: Math.round(position.x), y: Math.round(position.y) }

  return { origin, target, hold: null, position: remaining <= 2 ? null : position }
}
