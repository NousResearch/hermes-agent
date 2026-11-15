/**
 * The user handling the floating pet — the pure decisions behind poke / carry /
 * drop, plus the drop's fall. `floating-pet.tsx` owns the pointer events; the
 * surfaces come from `roam-geometry`, so a dropped pet lands on the same floor
 * and ledges the wander walks on.
 */

import { prefersReducedMotion } from '@/hooks/use-media-query'

import { FEET_DROP_PX, GRAVITY_PX_S2, GROUND_EPS, groundTop, type Ledge, MAX_DT_S, resolveLedge } from './roam-geometry'

/** Pointer travel (px) before a press becomes a carry; less is a poke, so click jitter never drags. */
export const DRAG_THRESHOLD_PX = 4

export const isDragMove = (dx: number, dy: number): boolean => Math.hypot(dx, dy) > DRAG_THRESHOLD_PX

/**
 * Resting top of the surface below a pet released at (x, y), or null when it's
 * already standing there (or is below every surface — it never falls upward).
 */
export function dropTarget(ledges: Ledge[], x: number, y: number, petH: number): null | number {
  // Measure from the visible feet: a pet resting on a ledge sits FEET_DROP_PX
  // into it, which would otherwise read as "below this ledge" and fall through.
  const rest = groundTop(resolveLedge(ledges, x, y - FEET_DROP_PX, petH), petH)

  return rest - y > GROUND_EPS ? rest : null
}

/**
 * Drop `el` (top-left anchored) from `fromY` to `toY` under gravity, then call
 * `onLand`. Reduced motion snaps there on the first frame — still async, so
 * `onLand` never runs before the caller has stored the returned cancel.
 */
export function animateDrop(el: HTMLElement, fromY: number, toY: number, onLand: () => void): () => void {
  const snap = prefersReducedMotion()
  let y = fromY
  let vel = 0
  let last = performance.now()
  let raf = 0

  const step = (now: number) => {
    const dt = Math.min(MAX_DT_S, Math.max(0, (now - last) / 1000))
    last = now
    vel += GRAVITY_PX_S2 * dt
    y = snap ? toY : Math.min(toY, y + vel * dt)
    el.style.top = `${y}px`

    if (y >= toY) {
      onLand()
    } else {
      raf = window.requestAnimationFrame(step)
    }
  }

  raf = window.requestAnimationFrame(step)

  return () => window.cancelAnimationFrame(raf)
}
