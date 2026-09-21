import { useCallback, useEffect, useRef } from 'react'

import { prewarmProfileBackend } from '@/store/profile'

// Dwell before firing: long enough that sweeping the pointer across the rail
// or a mixed-profile session list doesn't spawn a backend for every element
// passed through, short enough to beat the click by hundreds of ms.
export const PREWARM_DWELL_MS = 220

// A pointer can cross several rows while a virtualized rail is mounting or
// reflowing. Only the last intent represents where the user actually stopped;
// letting each row keep an independent timer turns that one gesture into a
// burst of speculative backend launches.
let latestPrewarmIntent = 0

/**
 * pointerenter/pointerleave/pointermove handlers that pre-warm `profile`'s
 * pool backend after a short hover dwell (see prewarmProfileBackend).
 *
 * `pointerenter` alone is not intent: virtualized session lists and running-arc
 * re-renders fire it when a *stationary* cursor sits over the sidebar and a
 * different profile's row slides under the pointer (#100548). Arm on enter,
 * start the dwell only after a real pointermove on that visit, cancel on leave.
 * Consumers merge these with their own pointer handlers.
 * All consumers share one intent so a pointer sweep launches at most one backend.
 */
export function usePrewarmIntent(prewarm: () => void) {
  const timer = useRef<null | number>(null)
  const armed = useRef(false)
  const intentRef = useRef(0)
  const prewarmRef = useRef(prewarm)
  prewarmRef.current = prewarm

  const cancelPrewarm = useCallback(() => {
    armed.current = false

    if (timer.current != null) {
      clearTimeout(timer.current)
      timer.current = null
    }

    if (intentRef.current === latestPrewarmIntent) {
      latestPrewarmIntent += 1
    }
  }, [])

  useEffect(() => cancelPrewarm, [cancelPrewarm])

  const startPrewarm = useCallback(() => {
    cancelPrewarm()
    armed.current = true
    intentRef.current = ++latestPrewarmIntent
  }, [cancelPrewarm])

  const notePointerMove = useCallback(() => {
    if (!armed.current || timer.current != null) {
      return
    }

    const intent = intentRef.current
    timer.current = window.setTimeout(() => {
      timer.current = null
      armed.current = false
      if (latestPrewarmIntent === intent) {
        prewarmRef.current()
      }
    }, PREWARM_DWELL_MS)
  }, [])

  return { cancelPrewarm, notePointerMove, startPrewarm }
}

/**
 * pointerenter/pointerleave handlers that pre-warm `profile`'s pool backend
 * after a short hover dwell (see prewarmProfileBackend in store/profile).
 * Consumers merge these with their own pointer handlers.
 */
export function useProfilePrewarm(profile: string | null | undefined) {
  const profileRef = useRef(profile)
  profileRef.current = profile

  return usePrewarmIntent(() => prewarmProfileBackend(profileRef.current || 'default'))
}
