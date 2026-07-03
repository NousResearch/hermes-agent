import { type RefObject, useEffect } from 'react'

import { $petMotion, $petRoamDir, type PetState } from '@/store/pet'

import { chooseMove, dwellMs, PAUSE_DWELL, pickStrollTarget } from './roam-behavior'
import {
  GROUND_EPS,
  groundTop,
  type Ledge,
  overlapsX,
  overlayLedge,
  resolveLedge,
  snapshotLedges
} from './roam-geometry'

interface Point {
  x: number
  y: number
}

// Foot-sync: advance this many body-widths per animation loop so the walk reads
// as steps, not a glide. Actual px/s is derived from the sprite's loop duration
// and on-screen size (see `walkSpeedPxS`).
const STRIDE_PER_LOOP = 0.8
// Downward acceleration for falls between ledges — fast enough to read as a drop.
const GRAVITY_PX_S2 = 5200
// Time to spring up onto a higher ledge.
const JUMP_DUR_MS = 460
// Tiny settle after a drag release before the pet re-plans (and usually falls),
// so dropping it in mid-air snaps down promptly instead of hanging for a beat.
const DROP_SETTLE_MS = 90
// Arrived at a walk target.
const ARRIVE_EPS = 1.5
// Cap dt so a backgrounded/throttled tab can't teleport the pet on resume.
const MAX_DT_S = 0.05

type Phase = 'pause' | 'walk' | 'fall' | 'jump'

const rand = (min: number, max: number): number => min + Math.random() * (max - min)
const easeOutCubic = (t: number): number => 1 - (1 - t) ** 3
const signDir = (n: number): -1 | 0 | 1 => (n > 0 ? 1 : n < 0 ? -1 : 0)

interface PetRoamOptions {
  /** Run the wander loop (roam opt-in + pet active + in-window + agent at rest). */
  enabled: boolean
  containerRef: RefObject<HTMLDivElement | null>
  /** True while the user is dragging — the loop yields so it never fights a drag. */
  isInteracting: () => boolean
  petW: number
  petH: number
  /** Sprite animation loop duration (ms) — paces the walk to the leg cadence. */
  loopMs: number
  /** A full-screen route overlay (settings/profiles/…) is up: patrol its base. */
  overlayOpen: boolean
  /** Persist the resting position back to React state when the loop settles. */
  commit: (point: Point) => void
}

/**
 * Drive the floating pet's wander as a platformer state machine: it loafs, then
 * walks along surfaces (the window floor, the top of the composer, …), hops up
 * onto higher ledges, and drops off them — instead of drifting through empty
 * space. The walkable surfaces come from `roam-geometry` (re-measured from the
 * live DOM every beat, so the pet tracks the composer growing, the sidebar
 * opening, the window resizing) and the beat-to-beat choices from
 * `roam-behavior`; this hook owns only the physics and the DOM writes.
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
export function usePetRoam({
  enabled,
  containerRef,
  isInteracting,
  petW,
  petH,
  loopMs,
  overlayOpen,
  commit
}: PetRoamOptions): void {
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

    // Pace the stride to the sprite: one body-width per animation loop.
    const walkSpeedPxS = (petW * STRIDE_PER_LOOP) / (loopMs / 1000)
    const restY = (ledge: Ledge): number => groundTop(ledge, petH)

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
      pauseUntil = now + dwellMs(PAUSE_DWELL)
      signal(null, 0)
      commit({ ...cur })
    }

    // Land flush on a ledge, then settle into the next idle beat.
    const settleOn = (ledge: Ledge, now: number) => {
      cur.y = restY(ledge)
      curLedge = ledge
      applyDom()
      beginPause(now)
    }

    const beginVertical = (ledge: Ledge) => {
      targetLedge = ledge

      if (restY(ledge) < cur.y - 1) {
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

    const planNext = (now: number) => {
      // An open overlay swaps the surface set to just its bottom edge, so the pet
      // patrols along it; closing it restores the normal surfaces (and the pet
      // drops to whatever's below).
      const ledges = overlayOpen ? [overlayLedge(petW)] : snapshotLedges(petW, petH)
      curLedge = resolveLedge(ledges, cur.x, cur.y, petH)

      if (Math.abs(cur.y - restY(curLedge)) > GROUND_EPS) {
        // Dragged into the air, or the surface moved out from under it: fall.
        beginVertical(curLedge)

        return
      }

      const reachable = ledges.filter(ledge => ledge !== curLedge && overlapsX(curLedge!, ledge))
      const move = chooseMove(reachable.length > 0)

      if (move === 'rest') {
        // Stay put and loaf another beat — movement is the exception.
        beginPause(now)

        return
      }

      if (move === 'hop') {
        const next = reachable[Math.floor(Math.random() * reachable.length)]!
        const lo = Math.max(curLedge.left, next.left)
        const hi = Math.min(curLedge.right, next.right)
        pendingHop = next
        walkTargetX = lo + Math.random() * (hi - lo)
      } else {
        pendingHop = null
        walkTargetX = pickStrollTarget(curLedge, cur.x)
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
            planNext(now)
          }

          break
        }

        case 'walk': {
          const remaining = walkTargetX - cur.x
          const stepDist = walkSpeedPxS * dt

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

          if (cur.y >= restY(targetLedge)) {
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
          cur.y = jumpFromY + (restY(targetLedge) - jumpFromY) * easeOutCubic(t)

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
  }, [enabled, petW, petH, loopMs, overlayOpen, containerRef, isInteracting, commit])
}
