/**
 * The "where can it stand" layer of the floating companion's wander: it measures the
 * live DOM for walkable surfaces and answers pure questions about them. Split
 * from the decision logic (`roam-behavior.ts`) and the RAF/DOM loop
 * (`use-companion-roam.ts`) so the loop reads as physics, not geometry, and the pure
 * helpers (`overlapsX`, `resolveLedge`, `groundTop`) stay unit testable.
 */

import { TITLEBAR_HEIGHT } from '@/app/shell/titlebar'

/**
 * A horizontal surface the companion can stand and walk on. `y` is the surface line
 * (where the companion's feet rest); `left`/`right` bound the companion's top-left x so the
 * whole sprite stays on the ledge.
 */
export interface Ledge {
  y: number
  left: number
  right: number
}

// Elements the companion can perch on top of, measured fresh each beat. The bottom
// floor is always a ledge; these add app furniture the companion can climb onto (the
// composer, the profile rail). Add a `data-slot` here to grow the playground.
const PERCH_SELECTORS = ['[data-slot="composer-surface"]', '[data-slot="profile-rail"]']

// A full-width bar pinned to the window bottom (the status bar). When present,
// the companion walks along its TOP edge instead of the window edge, so it stands on
// the bar rather than covering it.
const FLOOR_BAR_SELECTOR = '[data-slot="statusbar"]'

// Sprites carry a few px of transparent padding below the feet; sink the companion by
// this much so the visible feet meet the surface instead of hovering above it.
const FEET_DROP_PX = 4
// Snap distance: how close the feet must be to count as "on this ledge".
export const GROUND_EPS = 2

const vw = (): number => window.innerWidth || 800
const vh = (): number => window.innerHeight || 600

/** The y a companion of height `companionH` rests at when standing on `ledge`. */
export const groundTop = (ledge: Ledge, companionH: number): number => ledge.y - companionH + FEET_DROP_PX

/**
 * Do the companion's walkable x-ranges on two ledges overlap enough to step across?
 * (Pure — the wander uses it to find hop-reachable neighbours.)
 */
export const overlapsX = (from: Ledge, to: Ledge): boolean =>
  Math.min(from.right, to.right) > Math.max(from.left, to.left) + 2

/**
 * The highest surface at or below the companion's feet under its current x — i.e. what
 * it's standing on, or what it would fall onto. Pure; falls back to the floor
 * (always `ledges[0]`) if the companion is somehow below everything.
 */
export function resolveLedge(ledges: Ledge[], x: number, y: number, companionH: number): Ledge {
  const bottom = y + companionH
  let best: Ledge | null = null

  for (const ledge of ledges) {
    if (x < ledge.left - 2 || x > ledge.right + 2) {
      continue
    }

    if (ledge.y >= bottom - GROUND_EPS && (!best || ledge.y < best.y)) {
      best = ledge
    }
  }

  return best ?? ledges[0]!
}

/** The bottom ground line: the top of the status bar if it's pinned full-width
 *  across the window bottom, otherwise the window edge. */
function floorY(width: number, height: number, companionH: number): number {
  const bar = document.querySelector(FLOOR_BAR_SELECTOR)

  if (bar) {
    const rect = bar.getBoundingClientRect()

    if (rect.width >= width * 0.5 && height - rect.bottom < 4 && rect.top - companionH >= 0) {
      return rect.top
    }
  }

  return height
}

/** Snapshot the walkable surfaces right now: the bottom floor plus any on-screen
 *  perch element with room above it for the companion to stand. */
export function snapshotLedges(companionW: number, companionH: number): Ledge[] {
  const width = vw()
  const height = vh()
  const ledges: Ledge[] = [{ left: 0, right: Math.max(0, width - companionW), y: floorY(width, height, companionH) }]

  for (const selector of PERCH_SELECTORS) {
    const el = document.querySelector(selector)

    if (!el) {
      continue
    }

    const rect = el.getBoundingClientRect()
    const left = Math.max(0, rect.left)
    const right = Math.min(width - companionW, rect.right - companionW)

    // Skip surfaces that are too narrow for the companion, have no headroom above, or
    // sit off-screen / flush with the floor (no daylight between them).
    if (right <= left + 2 || rect.top - companionH < 0 || rect.top > height - 8 || height - rect.top < 12) {
      continue
    }

    ledges.push({ left, right, y: rect.top })
  }

  return ledges
}

/**
 * While a full-screen route overlay is up it's the only walkable surface: a
 * single ledge at the overlay card's bottom inner edge. The card uses
 * `OverlayView`'s equal inset on every side — `titlebar-height + padding` — so
 * we derive it from that rather than measuring.
 */
export function overlayLedge(companionW: number): Ledge {
  const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16
  const inset = TITLEBAR_HEIGHT + (vw() >= 640 ? 0.875 : 0.625) * rem

  return { left: inset, right: Math.max(0, vw() - inset - companionW), y: vh() - inset }
}
