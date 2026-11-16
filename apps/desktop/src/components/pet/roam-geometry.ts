/**
 * The "where can it stand" layer of the floating pet's wander: it measures the
 * live DOM for walkable surfaces and answers pure questions about them. Split
 * from the decision logic (`roam-behavior.ts`) and the RAF/DOM loop
 * (`use-pet-roam.ts`) so the loop reads as physics, not geometry, and the pure
 * helpers (`overlapsX`, `resolveLedge`, `groundTop`) stay unit testable.
 */

import { TITLEBAR_HEIGHT } from '@/app/shell/titlebar'

/**
 * A horizontal surface the pet can stand and walk on. `y` is the surface line
 * (where the pet's feet rest); `left`/`right` bound the pet's top-left x so the
 * whole sprite stays on the ledge.
 */
export interface Ledge {
  y: number
  left: number
  right: number
}

/**
 * A rectangle the pet's body must never overlap: an obstacle, not ground. The
 * walkable surface there is the rect's TOP edge.
 */
export interface Blocker {
  left: number
  right: number
  /** The surface line the pet stands on instead (the rect's top edge). */
  top: number
}

// Elements the pet can perch on top of, measured fresh each beat. The bottom
// floor is always a ledge; these add app furniture the pet can climb onto (the
// composer, the profile rail). Add a `data-slot` here to grow the playground.
const PERCH_SELECTORS = ['[data-slot="composer-surface"]', '[data-slot="profile-rail"]']

// Surfaces the pet must never stand IN FRONT OF. The composer (the input box)
// rests on the window floor and is about as tall as the pet, so a pet strolling
// the floor across the composer's span covered the input row. A blocker is an
// obstacle rather than ground: its TOP edge becomes the walkable surface there
// (it is also in PERCH_SELECTORS) and the floor is trimmed to the runs beside it.
const BLOCKER_SELECTORS = ['[data-slot="composer-surface"]']


// A full-width bar pinned to the window bottom (the status bar). When present,
// the pet walks along its TOP edge instead of the window edge, so it stands on
// the bar rather than covering it.
const FLOOR_BAR_SELECTOR = '[data-slot="statusbar"]'

// Sprites carry a few px of transparent padding below the feet; sink the pet by
// this much so the visible feet meet the surface instead of hovering above it.
const FEET_DROP_PX = 4
// Snap distance: how close the feet must be to count as "on this ledge".
export const GROUND_EPS = 2

const vw = (): number => window.innerWidth || 800
const vh = (): number => window.innerHeight || 600

/** The y a pet of height `petH` rests at when standing on `ledge`. */
export const groundTop = (ledge: Ledge, petH: number): number => ledge.y - petH + FEET_DROP_PX

/**
 * Do the pet's walkable x-ranges on two ledges overlap enough to step across?
 * (Pure — the wander uses it to find hop-reachable neighbours.)
 */
export const overlapsX = (from: Ledge, to: Ledge): boolean =>
  Math.min(from.right, to.right) > Math.max(from.left, to.left) + 2

/** Horizontal gap between two ledges' walkable spans (0 when they overlap). */
export const gapX = (from: Ledge, to: Ledge): number =>
  Math.max(0, Math.max(from.left, to.left) - Math.min(from.right, to.right))

/**
 * Can the pet get from one ledge to the other? Overlapping spans, or a gap no
 * wider than one body width — the step between the floor and the top of an
 * obstacle standing on it (the input box), which is exactly one pet wide.
 */
export const hopReachable = (from: Ledge, to: Ledge, petW: number): boolean =>
  overlapsX(from, to) || gapX(from, to) <= Math.max(8, petW)

/**
 * Where a hop between two ledges launches and lands. Overlapping spans share one
 * point (the pet springs straight up/down); a step across adjacent spans walks
 * to the near edge and lands on the far ledge's near edge, so the pet arcs over
 * the obstacle's corner instead of clipping through it.
 */
export function hopPlan(from: Ledge, to: Ledge, rng: () => number = Math.random): { launch: number; land: number } {
  const lo = Math.max(from.left, to.left)
  const hi = Math.min(from.right, to.right)

  if (hi > lo + 2) {
    const x = lo + rng() * (hi - lo)

    return { land: x, launch: x }
  }

  const rightwards = to.left > from.right
  const edge = rightwards ? to.left : to.right

  return { land: Math.min(Math.max(edge, to.left), to.right), launch: rightwards ? from.right : from.left }
}

/**
 * The highest surface at or below the pet's feet under its current x — i.e. what
 * it's standing on, or what it would fall onto. Pure; falls back to the floor
 * (always `ledges[0]`) if the pet is somehow below everything.
 */
export function resolveLedge(ledges: Ledge[], x: number, y: number, petH: number): Ledge {
  const bottom = y + petH
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
function floorY(width: number, height: number, petH: number): number {
  const bar = document.querySelector(FLOOR_BAR_SELECTOR)

  if (bar) {
    const rect = bar.getBoundingClientRect()

    if (rect.width >= width * 0.5 && height - rect.bottom < 4 && rect.top - petH >= 0) {
      return rect.top
    }
  }

  return height
}

/**
 * The obstacles in front of which the pet may not stand, as measured right now.
 * Only rects that rise clear of the ground line with room above for the pet
 * count — a composer already docked at/above the pet's head is not a blocker.
 */
export function snapshotBlockers(floor: number, petH: number): Blocker[] {
  const width = vw()
  const blockers: Blocker[] = []

  for (const selector of BLOCKER_SELECTORS) {
    const el = document.querySelector(selector)

    if (!el) {
      continue
    }

    const rect = el.getBoundingClientRect()

    if (rect.width < 8 || rect.height < 8 || rect.top - petH < 0 || rect.top >= floor - 4) {
      continue
    }

    if (rect.left > width || rect.right < 0) {
      continue
    }

    blockers.push({ left: Math.max(0, rect.left), right: Math.min(width, rect.right), top: rect.top })
  }

  return blockers
}

/**
 * Free pet-top-left x runs on a surface line once every blocker's footprint is
 * cut out. The pet's right edge may touch a blocker's left edge and its left
 * edge the blocker's right edge — its body never overlaps one.
 */
export function splitRuns(
  left: number,
  right: number,
  blockers: readonly Blocker[],
  petW: number
): Array<[number, number]> {
  let runs: Array<[number, number]> = [[left, right]]

  for (const blocker of blockers) {
    const cutLo = blocker.left - petW
    const cutHi = blocker.right
    const next: Array<[number, number]> = []

    for (const [start, end] of runs) {
      if (end <= cutLo || start >= cutHi) {
        next.push([start, end])

        continue
      }

      if (start < cutLo) {
        next.push([start, cutLo])
      }

      if (end > cutHi) {
        next.push([cutHi, end])
      }
    }

    runs = next
  }

  return runs
}

/** The ledge the pet may stand on ON TOP of `blocker` (body wholly over it). */
export function blockerTopLedge(blocker: Blocker, petW: number, width: number): Ledge | null {
  const left = Math.max(0, blocker.left)
  const right = Math.min(width - petW, blocker.right - petW)

  return right > left + 2 ? { left, right, y: blocker.top } : null
}

/** Snapshot the walkable surfaces right now: the bottom floor plus any on-screen
 *  perch element with room above it for the pet. */
export function snapshotLedges(petW: number, petH: number): Ledge[] {
  const width = vw()
  const height = vh()
  const floor = floorY(width, height, petH)
  const blockers = snapshotBlockers(floor, petH)
  const ledges: Ledge[] = []

  // The floor, minus every blocker's footprint: the pet walks up beside the
  // input box but never in front of it (the box's top edge is the surface
  // there, below). Runs too narrow to fit the pet are dropped — pacing a sliver
  // reads as a glitch, not as wandering.
  for (const [left, right] of splitRuns(0, Math.max(0, width - petW), blockers, petW)) {
    if (right - left >= petW) {
      ledges.push({ left, right, y: floor })
    }
  }

  for (const selector of PERCH_SELECTORS) {
    const el = document.querySelector(selector)

    if (!el) {
      continue
    }

    const rect = el.getBoundingClientRect()
    const left = Math.max(0, rect.left)
    const right = Math.min(width - petW, rect.right - petW)

    // Skip surfaces that are too narrow for the pet, have no headroom above, or
    // sit off-screen / flush with the floor (no daylight between them).
    if (right <= left + 2 || rect.top - petH < 0 || rect.top > height - 8 || height - rect.top < 12) {
      continue
    }

    ledges.push({ left, right, y: rect.top })
  }

  // resolveLedge falls back to ledges[0] as "the floor"; never leave it empty.
  return ledges.length ? ledges : [{ left: 0, right: Math.max(0, width - petW), y: floor }]
}

/**
 * The ledge the pet should be on when its body is currently in front of a
 * blocker — a position saved before this rule existed, a window resize, or the
 * composer growing upward under it. Null when it is clear (or the top has no
 * room for it).
 */
export function blockerLift(petW: number, petH: number, x: number, y: number): Ledge | null {
  const width = vw()

  for (const blocker of snapshotBlockers(floorY(width, vh(), petH), petH)) {
    if (x + petW <= blocker.left + 1 || x >= blocker.right - 1 || y + petH <= blocker.top + 1) {
      continue
    }

    const ledge = blockerTopLedge(blocker, petW, width)

    if (ledge) {
      return ledge
    }
  }

  return null
}


/**
 * While a full-screen route overlay is up it's the only walkable surface: a
 * single ledge at the overlay card's bottom inner edge. The card uses
 * `OverlayView`'s equal inset on every side — `titlebar-height + padding` — so
 * we derive it from that rather than measuring.
 */
export function overlayLedge(petW: number): Ledge {
  const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16
  const inset = TITLEBAR_HEIGHT + (vw() >= 640 ? 0.875 : 0.625) * rem

  return { left: inset, right: Math.max(0, vw() - inset - petW), y: vh() - inset }
}
