import { useEffect } from 'react'

import { transitionProfileKey } from '@/store/backend-transition'
import { reconnectGateway } from '@/store/gateway-reconnect'
import { endLauncherTransitionOnOpen, launcherPluginFor, probeLauncherTransition } from '@/store/launcher-transition'
import { $activeGatewayProfile, foregroundShows } from '@/store/profile'
import { $profileRemoteOverrides } from '@/store/profile-remote-override'
import { $gatewayState } from '@/store/session'

/**
 * When the active profile runs behind a plugin's launcher and its connection
 * drops, ask the launcher whether the sandbox is restarting on purpose
 * (launcher-transition.ts). A planned restart shows as a sandbox state and,
 * once the launcher reports the backend running again, reconnects at once
 * instead of waiting out the reconnect backoff. The socket opening again ends
 * the transition.
 *
 * That reconnect closes and redials the socket the window shows when it runs,
 * so it runs only while the window still shows the profile that restarted.
 */
export function useLauncherTransitions(): void {
  useEffect(() => {
    const reconnectNow = async (profile: string) => {
      if (await foregroundShows(profile)) {
        await reconnectGateway({ source: 'restart-followthrough' })
      }
    }

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
        void probeLauncherTransition(profile, () => void reconnectNow(profile).catch(() => undefined))
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
