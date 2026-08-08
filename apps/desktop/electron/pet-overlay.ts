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
 * completely unfindable: the pet "vanishes." The main window's restore path
 * already solves this class of problem (see window-state.ts `onScreen`); the
 * overlay must apply the same rule, plus re-home itself while open.
 */

import { MIN_VISIBLE, onScreen } from './window-state'

// Below this, a pet window is too small to be useful — mirrors the floor
// enforced in main.ts's spawnPetOverlayWindow.
const MIN_SIZE = 80

/**
 * Resolve where the pet-overlay window should open. Trusts `requested` screen
 * bounds when they still intersect some connected display (the same
 * ≥ MIN_VISIBLE overlap rule the main window's restore path uses), so a spot
 * remembered on a since-unplugged monitor can never strand the pet off-screen.
 *
 * When the requested spot is off-screen the window is re-centered on the
 * display holding `anchor` (the main window's content bounds — where the user
 * actually is), falling back to the primary display when the anchor is
 * missing. Size is capped to the target display's work area so a spot saved on
 * a since-disconnected bigger monitor can't exceed any screen the user now
 * has.
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

  if (onScreen({ x, y, width, height }, list)) {
    return requested
  }

  const area = workAreaForAnchor(list, anchor)

  if (!area) {
    return requested
  }

  const cappedWidth = Math.max(MIN_SIZE, Math.min(Math.round(width), Math.round(area.width)))
  const cappedHeight = Math.max(MIN_SIZE, Math.min(Math.round(height), Math.round(area.height)))

  return {
    x: Math.round(area.x + (area.width - cappedWidth) / 2),
    y: Math.round(area.y + (area.height - cappedHeight) / 2),
    width: cappedWidth,
    height: cappedHeight
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
