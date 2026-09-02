import { useEffect } from 'react'

import { refreshFleetRoster } from '@/store/fleet-roster'

/**
 * Keep the fleet roster fresh for the profile rail while more than one
 * gateway is registered: pull on mount, when the window regains focus or
 * visibility, and immediately when the connection registry changes. No timer
 * — the multi-connection contract rules out periodic fleet polling from the
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

    return () => {
      window.removeEventListener('focus', onFocus)
      document.removeEventListener('visibilitychange', onVisibility)
      offRegistry?.()
    }
  }, [enabled])
}
