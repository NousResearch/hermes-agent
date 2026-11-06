import { useEffect } from 'react'

import { refreshFleetRoster } from '@/store/fleet-roster'
import { $gatewayState } from '@/store/session'

/** Slow live-refresh cadence for the rail's work dots. The roster store's own
 *  60s stale window throttles this to one enumeration a minute. */
const FLEET_WORK_REFRESH_MS = 30_000

/**
 * Keep the fleet roster fresh for the profile rail while the rail shows more
 * than one profile: pull on mount, when the window regains focus or visibility,
 * immediately when the connection registry changes, and when the gateway
 * (re)opens. A slow 30s tick is the one exception to "no timer": worker
 * heartbeats never reach the gateway sockets, so a bot going mid-card (or
 * dying) is invisible without a poll. The store's 60s stale window throttles
 * the tick to one enumeration a minute, and pooled backends + the SSH inventory
 * cache mean the tick never spawns a gateway.
 *
 * The open edge matters on a cold start: the first pull races `gateway
 * ensure`, whose boot outlives the roster's per-source deadline, so This
 * device enumerates as unreachable (one square instead of its profiles) and
 * nothing else would ask again until a window focus.
 */
export function useFleetRoster(enabled: boolean): void {
  useEffect(() => {
    if (!enabled) {
      return
    }

    void refreshFleetRoster()

    const onFocus = () => void refreshFleetRoster()

    const onVisibility = () => {
      if (document.visibilityState === 'visible') {
        void refreshFleetRoster()
      }
    }

    window.addEventListener('focus', onFocus)
    document.addEventListener('visibilitychange', onVisibility)
    const offRegistry = window.hermesDesktop?.connections?.onChanged?.(() => void refreshFleetRoster({ force: true }))
    let wasOpen = $gatewayState.get() === 'open'

    const offGateway = $gatewayState.listen(state => {
      const open = state === 'open'

      if (open && !wasOpen) {
        void refreshFleetRoster({ force: true })
      }

      wasOpen = open
    })

    // Work dots need a clock of their own: a worker that stops heartbeating
    // must flip without the user touching the window. Hidden documents skip
    // the tick; the next focus refresh catches up.
    const workTick =
      typeof window === 'undefined'
        ? null
        : window.setInterval(() => {
            if (document.visibilityState === 'visible') {
              void refreshFleetRoster()
            }
          }, FLEET_WORK_REFRESH_MS)

    return () => {
      window.removeEventListener('focus', onFocus)
      document.removeEventListener('visibilitychange', onVisibility)
      offRegistry?.()
      offGateway()

      if (workTick !== null) {
        window.clearInterval(workTick)
      }
    }
  }, [enabled])
}
