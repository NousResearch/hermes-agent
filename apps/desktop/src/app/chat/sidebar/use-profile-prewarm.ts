import { useCallback, useEffect, useRef } from 'react'

import { prewarmProfileBackend } from '@/store/profile'

// Dwell before firing: long enough that sweeping the pointer across the rail
// or a mixed-profile session list doesn't spawn a backend for every element
// passed through, short enough to beat the click by hundreds of ms.
const PREWARM_DWELL_MS = 120

/**
 * pointerenter/pointerleave/pointermove handlers that pre-warm `profile`'s
 * pool backend after a short hover dwell (see prewarmProfileBackend).
 *
 * `pointerenter` alone is not intent: virtualized session lists and running-arc
 * re-renders fire it when a *stationary* cursor sits over the sidebar and a
 * different profile's row slides under the pointer (#100548). Arm on enter,
 * start the dwell only after a real pointermove on that visit, cancel on leave.
 * Consumers merge these with their own pointer handlers.
 */
export function useProfilePrewarm(profile: string | null | undefined) {
  const timer = useRef<null | number>(null)
  const armed = useRef(false)
  const profileRef = useRef(profile)
  profileRef.current = profile

  const cancelPrewarm = useCallback(() => {
    armed.current = false
    if (timer.current != null) {
      clearTimeout(timer.current)
      timer.current = null
    }
  }, [])

  useEffect(() => cancelPrewarm, [cancelPrewarm])

  const startPrewarm = useCallback(() => {
    cancelPrewarm()
    armed.current = true
  }, [cancelPrewarm])

  const notePointerMove = useCallback(() => {
    if (!armed.current || timer.current != null) {
      return
    }

    timer.current = window.setTimeout(() => {
      timer.current = null
      armed.current = false
      prewarmProfileBackend(profileRef.current || 'default')
    }, PREWARM_DWELL_MS)
  }, [])

  return { cancelPrewarm, notePointerMove, startPrewarm }
}
