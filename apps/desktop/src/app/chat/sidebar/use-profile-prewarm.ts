import { useCallback, useEffect, useRef } from 'react'

import { prewarmProfileBackend } from '@/store/profile'

// Dwell before firing: long enough that sweeping the pointer across the rail
// or a mixed-profile session list doesn't spawn a backend for every element
// passed through, short enough to beat the click by hundreds of ms.
const PREWARM_DWELL_MS = 220

// A pointer can cross several rows while a virtualized rail is mounting or
// reflowing. Only the last intent represents where the user actually stopped;
// letting each row keep an independent timer turns that one gesture into a
// burst of speculative backend launches.
let latestPrewarmIntent = 0

/**
 * Delay one speculative prewarm and invalidate older hover intents globally.
 * Profile squares, session rows, and Bot Mode rows all use this primitive so
 * a single pointer can queue at most one background connection attempt.
 */
export function usePrewarmIntent(prewarm: () => void) {
  const timer = useRef<null | number>(null)
  const intentRef = useRef(0)
  const prewarmRef = useRef(prewarm)
  prewarmRef.current = prewarm

  const cancelPrewarm = useCallback(() => {
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
    const intent = ++latestPrewarmIntent
    intentRef.current = intent

    timer.current = window.setTimeout(() => {
      timer.current = null

      if (latestPrewarmIntent === intent) {
        prewarmRef.current()
      }
    }, PREWARM_DWELL_MS)
  }, [cancelPrewarm])

  return { cancelPrewarm, startPrewarm }
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
