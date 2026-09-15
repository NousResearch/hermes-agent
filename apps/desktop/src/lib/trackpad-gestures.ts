// Trackpad / pointer gesture primitives shared across canvas + DOM surfaces.
//
// macOS quirk (Chromium/Electron): both pinch-zoom and "smart zoom" arrive as
// `wheel` events with `ctrlKey` synthetically set — there is no dedicated DOM
// event for either. They're disambiguated by their deltas:
//   - pinch-to-zoom: ctrlKey + a non-zero delta
//   - smart zoom:    ctrlKey + zero deltas   (the two-finger double-tap)
// Plain two-finger scroll has ctrlKey === false. Centralising this here keeps
// every zoom/pan surface from re-deriving the same OS trivia (and getting it
// wrong, which makes smart-zoom read as a zoom-in).

export interface WheelLike {
  ctrlKey: boolean
  deltaX: number
  deltaY: number
}

export interface HorizontalSwipeResult {
  claimed: boolean
  direction: -1 | 1 | null
}

/** macOS "smart zoom" (two-finger double-tap): a ctrl-wheel with no delta. */
export function isSmartZoomWheel(e: WheelLike): boolean {
  return e.ctrlKey && e.deltaX === 0 && e.deltaY === 0
}

/** Pinch-to-zoom (or ctrl + mouse wheel): a ctrl-wheel carrying a delta. */
export function isPinchZoomWheel(e: WheelLike): boolean {
  return e.ctrlKey && (e.deltaX !== 0 || e.deltaY !== 0)
}

export const HORIZONTAL_SWIPE_THRESHOLD = 36
export const HORIZONTAL_SWIPE_IDLE_MS = 180
export const HORIZONTAL_SWIPE_REARM_MS = 220
export const HORIZONTAL_SWIPE_REARM_DELTA = 8
export const HORIZONTAL_SWIPE_REARM_FACTOR = 1.8

/**
 * Turns Chromium's horizontal trackpad wheel stream into one action per
 * physical swipe. A fresh acceleration impulse can rearm the detector inside
 * macOS's long momentum tail, so repeated swipes do not require pointer motion.
 */
export function createHorizontalSwipeDetector(
  threshold: number = HORIZONTAL_SWIPE_THRESHOLD,
  idleMs: number = HORIZONTAL_SWIPE_IDLE_MS
): (event: WheelLike, now?: number) => HorizontalSwipeResult {
  let accumulated = 0
  let lastEventAt: number | null = null
  let triggered = false
  let triggeredAt: number | null = null
  let lastMagnitude = 0

  return (event: WheelLike, now: number = Date.now()): HorizontalSwipeResult => {
    const horizontal = !event.ctrlKey && Math.abs(event.deltaX) > Math.abs(event.deltaY)

    if (!horizontal) {
      return { claimed: false, direction: null }
    }

    if (lastEventAt === null || now - lastEventAt > idleMs) {
      accumulated = 0
      triggered = false
      triggeredAt = null
      lastMagnitude = 0
    }

    lastEventAt = now

    const magnitude = Math.abs(event.deltaX)

    if (triggered) {
      const freshImpulse =
        triggeredAt !== null &&
        now - triggeredAt >= HORIZONTAL_SWIPE_REARM_MS &&
        magnitude >= HORIZONTAL_SWIPE_REARM_DELTA &&
        magnitude > lastMagnitude * HORIZONTAL_SWIPE_REARM_FACTOR

      lastMagnitude = magnitude

      if (!freshImpulse) {
        return { claimed: true, direction: null }
      }

      accumulated = event.deltaX
      triggered = false
      triggeredAt = null
    } else {
      accumulated += event.deltaX
      lastMagnitude = magnitude
    }

    if (Math.abs(accumulated) < threshold) {
      return { claimed: true, direction: null }
    }

    triggered = true
    triggeredAt = now

    return { claimed: true, direction: accumulated > 0 ? 1 : -1 }
  }
}

export const DOUBLE_TAP_MS = 300

/**
 * Stateful double-tap detector for surfaces where a real `dblclick` may never
 * fire (e.g. a trackpad with tap-to-click off). Call it once per discrete tap;
 * it returns true when two taps land within `thresholdMs` of each other, then
 * resets so a third tap starts a fresh pair.
 */
export function createDoubleTapDetector(thresholdMs: number = DOUBLE_TAP_MS): (now?: number) => boolean {
  let last = 0

  return (now: number = Date.now()): boolean => {
    if (now - last < thresholdMs) {
      last = 0

      return true
    }

    last = now

    return false
  }
}
