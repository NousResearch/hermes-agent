import { useEffect } from 'react'

import { $activeConnectionId } from '@/store/connections'
import { refreshFleetRoster } from '@/store/fleet-roster'

/**
 * Keep the fleet roster fresh for the profile rail while more than one
 * gateway is registered: pull on mount, when the window regains focus or
 * visibility, immediately when the connection registry changes, and
 * immediately after the active gateway changes — a switch is what dials (and
 * pools) a source that was connect-on-demand a moment ago, and the last-used
 * write it ends with raises no registry event. No timer — the
 * multi-connection contract rules out periodic fleet polling from the
 * sidebar, and a 60s stale window in the store absorbs focus churn.
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
    // listen (not subscribe): the mount pull above already covers the current
    // source; this fires only when the active gateway actually moves.
    const offActive = $activeConnectionId.listen(() => void refreshFleetRoster({ force: true }))

    return () => {
      window.removeEventListener('focus', onFocus)
      document.removeEventListener('visibilitychange', onVisibility)
      offRegistry?.()
      offActive()
    }
  }, [enabled])
}
