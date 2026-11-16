import { useEffect } from 'react'

import { transitionProfileKey } from '@/store/backend-transition'
import { reconnectGateway } from '@/store/gateway-reconnect'
import { endLauncherTransitionOnOpen, launcherPluginFor, probeLauncherTransition } from '@/store/launcher-transition'
import { $activeGatewayProfile } from '@/store/profile'
import { $profileRemoteOverrides } from '@/store/profile-remote-override'
import { $gatewayState } from '@/store/session'

/**
 * When the active profile runs behind a plugin's launcher and its connection
 * drops, ask the launcher whether the sandbox is restarting on purpose
 * (launcher-transition.ts). A planned restart shows as a sandbox state and,
 * once the launcher reports the backend running again, reconnects at once
 * instead of waiting out the reconnect backoff. The socket opening again ends
 * the transition.
 */
export function useLauncherTransitions(): void {
  useEffect(() => {
    const reconnectNow = () => void reconnectGateway({ source: 'restart-followthrough' }).catch(() => undefined)

    const check = () => {
      const profile = transitionProfileKey($activeGatewayProfile.get())

      if (!launcherPluginFor(profile)) {
        return
      }

      const state = $gatewayState.get()

      if (state === 'open') {
        endLauncherTransitionOnOpen(profile)

        return
      }

      if (state !== 'idle') {
        void probeLauncherTransition(profile, reconnectNow)
      }
    }

    const offs = [
      $gatewayState.listen(check),
      $activeGatewayProfile.listen(check),
      $profileRemoteOverrides.listen(check)
    ]

    return () => offs.forEach(off => off())
  }, [])
}
