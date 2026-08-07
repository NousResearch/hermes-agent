// Two-finger horizontal swipe detection for the sidebar.
//
// On macOS, a two-finger horizontal swipe on a trackpad produces native
// `wheel` events with a non‑zero `deltaX` and typically zero `deltaY`.
// This hook discriminates deliberate swipes from ordinary vertical scrolling
// and calls the profile‑cycle callback.
//
// Electron exposes no dedicated swipe‑gesture DOM event — the only reliable
// signal is the wheel‑event delta, which is what every browser-based swipe
// surface on macOS uses (see also: lib/trackpad-gestures).

import { useCallback, useEffect, useRef, type RefObject } from 'react'

import { cycleProfile } from '@/store/profile'

/**
 * Minimum accumulated horizontal delta (in pixels or equivalent) to treat as a
 * deliberate swipe.
 */
const SWIPE_THRESHOLD_PX = 20

/**
 * Cooldown (ms) between consecutive profile cycles.
 */
const COOLDOWN_MS = 400

/** Assumed line-height in pixels when deltaMode is DOM_DELTA_LINE. */
const LINE_HEIGHT_PX = 20

/**
 * Resolve a wheel delta to equivalent pixels regardless of deltaMode.
 */
function resolveDeltaX(event: WheelEvent): number {
  switch (event.deltaMode) {
    case WheelEvent.DOM_DELTA_LINE:
      return event.deltaX * LINE_HEIGHT_PX
    case WheelEvent.DOM_DELTA_PAGE:
      return event.deltaX * window.innerWidth
    default:
      return event.deltaX
  }
}

/**
 * Subscribe to a container element and fire `cycleProfile` on two‑finger
 * horizontal swipes.
 */
export function useSidebarSwipe(containerRef: RefObject<HTMLElement | null>): void {
  const lastCycleAt = useRef(0)

  const handleWheel = useCallback((event: WheelEvent) => {
    const dx = resolveDeltaX(event)
    const dy = event.deltaY

    // Ignore predominantly vertical movement
    if (dy !== 0 && Math.abs(dx / dy) < 1.2) {
      return
    }

    // Need enough horizontal movement
    const absDx = Math.abs(dx)
    if (absDx < SWIPE_THRESHOLD_PX) {
      return
    }

    // Cooldown
    const now = Date.now()
    if (now - lastCycleAt.current < COOLDOWN_MS) {
      event.preventDefault()
      return
    }

    lastCycleAt.current = now
    event.preventDefault()
    event.stopPropagation()
    cycleProfile(dx > 0 ? 1 : -1)
  }, [])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    // Use capture phase to catch events before child scrollers consume them
    el.addEventListener('wheel', handleWheel, { passive: false, capture: true })

    return () => {
      el.removeEventListener('wheel', handleWheel)
    }
  }, [containerRef, handleWheel])
}
