import { type RefObject, useEffect } from 'react'

import { $petMotion, $petRoamDir, type PetState } from '@/store/pet'

interface Point {
  x: number
  y: number
}

/**
 * A horizontal surface the pet can stand and walk on. `y` is the surface line
 * (where the pet's feet rest); `left`/`right` bound the pet's top-left x so the
 * whole sprite stays on the ledge.
 */
interface Ledge {
  y: number
  left: number
  right: number
}

// Elements the pet can perch on top of, measured fresh each beat. The bottom
// floor is always a ledge; these add app furniture the pet can climb onto (the
// composer, the profile rail). Add a `data-slot` here to grow the playground.
const PERCH_SELECTORS = ['[data-slot="composer-surface"]', '[data-slot="profile-rail"]']

// A full-width bar pinned to the window bottom (the status bar). When present,
// the pet walks along its TOP edge instead of the window edge, so it stands on
// the bar rather than covering it.
const FLOOR_BAR_SELECTOR = '[data-slot="statusbar"]'

const WALK_SPEED_PX_S = 58
// Downward acceleration for falls between ledges — fast enough to read as a drop.
const GRAVITY_PX_S2 = 5200
// Time to spring up onto a higher ledge.
const JUMP_DUR_MS = 460
const PAUSE_MIN_MS = 1800
const PAUSE_MAX_MS = 5200
// Tiny settle after a drag release before the pet re-plans (and usually falls),
// so dropping it in mid-air snaps down promptly instead of hanging for a beat.
const DROP_SETTLE_MS = 90
// Chance a beat hops to another ledge instead of strolling the current one.
const HOP_CHANCE = 0.45
// Strolls should cover ground, not shuffle: travel at least this fraction of the
// ledge (or this many px, whichever is larger), up to the room available.
const STROLL_MIN_FRACTION = 0.45
const STROLL_MIN_PX = 110
// Snap distances: "on this ledge" / arrived at a walk target.
const GROUND_EPS = 2
const ARRIVE_EPS = 1.5
// Cap dt so a backgrounded/throttled tab can't teleport the pet on resume.
const MAX_DT_S = 0.05

type Phase = 'pause' | 'walk' | 'fall' | 'jump'

const rand = (min: number, max: number): number => min + Math.random() * (max - min)
const easeOutCubic = (t: number): number => 1 - (1 - t) ** 3
const signDir = (n: number): -1 | 0 | 1 => (n > 0 ? 1 : n < 0 ? -1 : 0)

function vw(): number {
  return window.innerWidth || 800
}

function vh(): number {
  return window.innerHeight || 600
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

/** Snapshot the walkable surfaces right now: the bottom floor plus any on-screen
 *  perch element with room above it for the pet to stand. */
function snapshotLedges(petW: number, petH: number): Ledge[] {
  const width = vw()
  const height = vh()
  const ledges: Ledge[] = [{ left: 0, right: Math.max(0, width - petW), y: floorY(width, height, petH) }]

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

  return ledges
}

interface PetRoamOptions {
  /** Run the wander loop (roam opt-in + pet active + in-window + agent at rest). */
  enabled: boolean
  containerRef: RefObject<HTMLDivElement | null>
  /** True while the user is dragging — the loop yields so it never fights a drag. */
  isInteracting: () => boolean
  petW: number
  petH: number
  /** Persist the resting position back to React state when the loop settles. */
  commit: (point: Point) => void
}

/**
 * Make the floating pet wander the app like a platformer character: it walks
 * along surfaces (the window floor, the top of the composer, …), hops up onto
 * higher ledges, and drops off them — instead of drifting diagonally through
 * empty space. Surfaces are re-measured from the live DOM at the start of every
 * beat (`snapshotLedges`), so the pet tracks the composer growing, the sidebar
 * opening, the window resizing, and even falls back to the floor when its perch
 * disappears.
 *
 * Movement mutates `el.style.left/top` directly each frame — like the drag
 * handler — so a steady wander triggers no React re-renders, and because it
 * re-asserts the DOM position every frame, an incidental parent re-render that
 * snaps `style` back self-heals within a frame. State is only committed (via
 * `commit`) when the pet settles, keeping React's `position` in sync once the
 * loop stops driving it.
 *
 * Two signals publish the wander so the canvas/sprite react without a prop
 * change: `$petMotion` (`run` while walking, `jump` while hopping/falling) flips
 * the shared `$petState`, and `$petRoamDir` (-1/0/1) lets the floating pet pick
 * the directional run row + mirror for the travel direction.
 */
export function usePetRoam({ enabled, containerRef, isInteracting, petW, petH, commit }: PetRoamOptions): void {
  useEffect(() => {
    if (!enabled) {
      $petMotion.set(null)
      $petRoamDir.set(0)

      return
    }

    const el = containerRef.current

    if (!el) {
      return
    }

    const groundTop = (ledge: Ledge): number => ledge.y - petH

    // A stroll destination on `ledge` that actually goes somewhere: lean toward
    // the side with more room (so the pet crosses the app rather than shuffling
    // in place) and guarantee a decent minimum travel distance.
    const pickStrollTarget = (ledge: Ledge): number => {
      const span = ledge.right - ledge.left

      if (span <= 4) {
        return ledge.left
      }

      const roomLeft = cur.x - ledge.left
      const roomRight = ledge.right - cur.x
      const goRight = roomRight >= roomLeft ? Math.random() < 0.85 : Math.random() < 0.15
      const room = Math.max(0, goRight ? roomRight : roomLeft)
      const minDist = Math.min(room, Math.max(span * STROLL_MIN_FRACTION, STROLL_MIN_PX))
      const dist = minDist + Math.random() * Math.max(0, room - minDist)

      return goRight ? cur.x + dist : cur.x - dist
    }

    // Seed from the live DOM rect so we resume from wherever the pet actually is
    // (after a drag, reclamp, or activity pause) rather than a stale closure.
    const rect = el.getBoundingClientRect()
    const cur: Point = { x: rect.left, y: rect.top }

    let phase: Phase = 'pause'
    let pauseUntil = performance.now() + rand(400, 1200)
    let last = performance.now()
    let raf = 0

    let walkTargetX = cur.x
    let curLedge: Ledge | null = null
    let targetLedge: Ledge | null = null
    // When set, the current walk is the approach run before a hop to this ledge.
    let pendingHop: Ledge | null = null
    // Fall / jump integrators.
    let fallVel = 0
    let jumpFromY = 0
    let jumpElapsed = 0

    const applyDom = () => {
      el.style.left = `${cur.x}px`
      el.style.top = `${cur.y}px`
    }

    // One chokepoint for the wander signals: the pose (drives `$petState`) and
    // the travel direction (drives the floating pet's directional row + mirror).
    const signal = (pose: PetState | null, dir: -1 | 0 | 1) => {
      $petMotion.set(pose)
      $petRoamDir.set(dir)
    }

    const beginPause = (now: number) => {
      phase = 'pause'
      pauseUntil = now + rand(PAUSE_MIN_MS, PAUSE_MAX_MS)
      signal(null, 0)
      commit({ ...cur })
    }

    // Land flush on a ledge, then settle into the next idle beat.
    const settleOn = (ledge: Ledge, now: number) => {
      cur.y = groundTop(ledge)
      curLedge = ledge
      applyDom()
      beginPause(now)
    }

    const beginVertical = (ledge: Ledge) => {
      targetLedge = ledge

      if (groundTop(ledge) < cur.y - 1) {
        // Up onto a higher ledge: a quick spring.
        phase = 'jump'
        jumpFromY = cur.y
        jumpElapsed = 0
      } else {
        // Down off a ledge: let gravity take it.
        phase = 'fall'
        fallVel = 0
      }

      signal('jump', 0)
    }

    // Does the pet, standing at cur.x, have a stretch of `to` it can step across
    // to from `from` (their walkable x-ranges overlap)?
    const overlapsX = (from: Ledge, to: Ledge): boolean =>
      Math.min(from.right, to.right) > Math.max(from.left, to.left) + 2

    // Find the highest surface at or below the pet's feet under its current x —
    // i.e. what it's standing on, or what it would fall onto.
    const resolveLedge = (ledges: Ledge[]): Ledge => {
      const bottom = cur.y + petH
      let best: Ledge | null = null

      for (const ledge of ledges) {
        if (cur.x < ledge.left - 2 || cur.x > ledge.right + 2) {
          continue
        }

        if (ledge.y >= bottom - GROUND_EPS && (!best || ledge.y < best.y)) {
          best = ledge
        }
      }

      // Floor always spans the clamped x-range, so this only falls back if the
      // pet is somehow below everything — drop it to the floor.
      return best ?? ledges[0]!
    }

    const planNext = () => {
      const ledges = snapshotLedges(petW, petH)
      curLedge = resolveLedge(ledges)
      const grounded = Math.abs(cur.y - groundTop(curLedge)) <= GROUND_EPS

      if (!grounded) {
        // Dragged into the air (or a perch vanished): fall to the surface below.
        beginVertical(curLedge)

        return
      }

      const reachable = ledges.filter(ledge => ledge !== curLedge && overlapsX(curLedge!, ledge))

      if (reachable.length > 0 && Math.random() < HOP_CHANCE) {
        const next = reachable[Math.floor(Math.random() * reachable.length)]!
        const lo = Math.max(curLedge.left, next.left)
        const hi = Math.min(curLedge.right, next.right)
        pendingHop = next
        walkTargetX = lo + Math.random() * (hi - lo)
      } else {
        pendingHop = null
        walkTargetX = pickStrollTarget(curLedge)
      }

      phase = 'walk'
      signal('run', signDir(walkTargetX - cur.x))
    }

    const step = (now: number) => {
      const dt = Math.min(MAX_DT_S, (now - last) / 1000)
      last = now

      // Yield to a drag: track the pet so we resume from the drop point, and
      // reset the idle beat so it doesn't bolt the instant it's let go.
      if (isInteracting()) {
        const live = el.getBoundingClientRect()
        cur.x = live.left
        cur.y = live.top
        phase = 'pause'
        pendingHop = null
        // Short settle so the pet falls right after you drop it, not seconds later.
        pauseUntil = now + DROP_SETTLE_MS
        signal(null, 0)
        raf = requestAnimationFrame(step)

        return
      }

      switch (phase) {
        case 'pause': {
          if (now >= pauseUntil) {
            planNext()
          }

          break
        }

        case 'walk': {
          const remaining = walkTargetX - cur.x
          const stepDist = WALK_SPEED_PX_S * dt

          if (Math.abs(remaining) <= Math.max(ARRIVE_EPS, stepDist)) {
            cur.x = walkTargetX
            applyDom()

            if (pendingHop) {
              const next = pendingHop
              pendingHop = null
              beginVertical(next)
            } else {
              beginPause(now)
            }
          } else {
            cur.x += Math.sign(remaining) * stepDist
            applyDom()
          }

          break
        }

        case 'fall': {
          if (!targetLedge) {
            beginPause(now)

            break
          }

          fallVel += GRAVITY_PX_S2 * dt
          cur.y += fallVel * dt

          if (cur.y >= groundTop(targetLedge)) {
            settleOn(targetLedge, now)
          } else {
            applyDom()
          }

          break
        }

        case 'jump': {
          if (!targetLedge) {
            beginPause(now)

            break
          }

          jumpElapsed += dt * 1000
          const t = Math.min(1, jumpElapsed / JUMP_DUR_MS)
          cur.y = jumpFromY + (groundTop(targetLedge) - jumpFromY) * easeOutCubic(t)

          if (t >= 1) {
            settleOn(targetLedge, now)
          } else {
            applyDom()
          }

          break
        }
      }

      raf = requestAnimationFrame(step)
    }

    raf = requestAnimationFrame(step)

    return () => {
      cancelAnimationFrame(raf)
      signal(null, 0)
      // Hand the final position back to React so its `style` matches the DOM once
      // the loop stops re-asserting it.
      commit({ ...cur })
    }
  }, [enabled, petW, petH, containerRef, isInteracting, commit])
}
