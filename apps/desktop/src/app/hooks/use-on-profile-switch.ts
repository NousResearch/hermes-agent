import { useEffect, useRef } from 'react'

import { $activeGatewayProfile } from '@/store/profile'

/** Run `onSwitch` when the active gateway profile changes — never on first
 *  mount. For dropping per-profile view state (probes, cached usage, drafts)
 *  when the backend the app talks to swaps underneath a still-mounted view. */
export function useOnProfileSwitch(onSwitch: () => void): void {
  const onSwitchRef = useRef(onSwitch)
  const profileRef = useRef($activeGatewayProfile.get())
  onSwitchRef.current = onSwitch

  useEffect(
    () =>
      $activeGatewayProfile.subscribe(profile => {
        if (profile === profileRef.current) {
          return
        }

        profileRef.current = profile
        onSwitchRef.current()
      }),
    []
  )
}
