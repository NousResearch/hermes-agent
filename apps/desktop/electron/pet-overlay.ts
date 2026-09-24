/**
 * Pure geometry helpers for the popped-out pet overlay — deciding where its
 * window may open and where it must be re-homed when the display topology
 * changes. Side-effect-free so the on-screen validation is unit-testable
 * without booting Electron; main.ts owns the live `screen` displays, the
 * anchor window, and the overlay window itself.
 *
 * The bug this exists for: the renderer remembers the overlay's absolute
 * screen position (localStorage hermes.desktop.pet-overlay-bounds.v1) and
 * reuses it verbatim on the next pop-out / app restart. If that spot was on an
 * external monitor that has since been unplugged, the overlay is created
 * off-screen — and because it is a transparent, frameless, non-activating
 * always-on-top panel hidden from Mission Control, an off-screen overlay is
 * completely unfindable: the pet "vanishes." A window that only *mostly* left
 * the screen is just as lost, so the overlay is kept wholly inside a work area
 * rather than trusting a sliver of overlap (#85092).
 */

import { matchingWorkArea } from './window-state'

// Below this, a pet window is too small to be useful — mirrors the floor
// enforced in main.ts's spawnPetOverlayWindow.
const MIN_SIZE = 80

const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi))

/**
 * Keep the WHOLE rect inside `workArea`: size is capped to the work area, then
 * the origin is clamped so no edge crosses it.
 */
export function clampRectToWorkArea(rect, workArea) {
  const width = clamp(Math.round(rect.width), MIN_SIZE, Math.round(workArea.width))
  const height = clamp(Math.round(rect.height), MIN_SIZE, Math.round(workArea.height))

  return {
    x: clamp(Math.round(rect.x), workArea.x, workArea.x + workArea.width - width),
    y: clamp(Math.round(rect.y), workArea.y, workArea.y + workArea.height - height),
    width,
    height
  }
}

/**
 * Resolve where the pet-overlay window may sit. A `requested` screen rect that
 * overlaps any connected display is clamped wholly into the work area it
 * overlaps most (the display Electron's `screen.getDisplayMatching` picks).
 *
 * A rect on no display at all (saved on a since-unplugged monitor) is
 * re-centered on the display holding `anchor` (the main window's content
 * bounds — where the user actually is), falling back to the primary display
 * when the anchor is missing.
 *
 * Returns null for missing/garbage input (main falls back to its defaults);
 * returns `requested` unchanged when there is nothing to validate against.
 */
export function resolvePetOverlayBounds(requested, displays, anchor) {
  if (!requested) {
    return null
  }

  const { x, y, width, height } = requested

  if (![x, y, width, height].every(Number.isFinite)) {
    return null
  }

  const list = Array.isArray(displays) ? displays : []

  if (!list.length) {
    return requested
  }

  const overlapping = matchingWorkArea({ x, y, width, height }, list, 1)

  if (overlapping) {
    return clampRectToWorkArea(requested, overlapping)
  }

  const area = workAreaForAnchor(list, anchor)

  if (!area) {
    return requested
  }

  const { width: w, height: h } = clampRectToWorkArea(requested, area)

  return {
    x: Math.round(area.x + (area.width - w) / 2),
    y: Math.round(area.y + (area.height - h) / 2),
    width: w,
    height: h
  }
}

// The work area of the display whose bounds contain the anchor's center (the
// display the main window sits on), or the first display when the anchor is
// missing/unknown. Null when there are no usable displays at all.
function workAreaForAnchor(displays, anchor) {
  if (
    anchor &&
    Number.isFinite(anchor.x) &&
    Number.isFinite(anchor.y) &&
    Number.isFinite(anchor.width) &&
    Number.isFinite(anchor.height)
  ) {
    const cx = anchor.x + anchor.width / 2
    const cy = anchor.y + anchor.height / 2

    const containing = displays.find(({ workArea: a }) => {
      if (!a) {
        return false
      }

      return cx >= a.x && cx < a.x + a.width && cy >= a.y && cy < a.y + a.height
    })

    if (containing?.workArea) {
      return containing.workArea
    }
  }

  return displays.find(({ workArea: a }) => a)?.workArea ?? null
}
