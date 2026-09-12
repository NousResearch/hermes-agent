import { useStore } from '@nanostores/react'
import { computed } from 'nanostores'
import { useEffect } from 'react'

import { $activeConnectionId } from '@/store/connections'
import {
  $continuationRefresh,
  $freeTierContinuation,
  continuationKey,
  continuationRequester,
  continuationRequired,
  continuationSuppressed,
  continuationTarget,
  refreshContinuation
} from '@/store/free-tier-continuation'
import { $freeTierSignIn } from '@/store/free-tier-sign-in'
import { $desktopOnboarding } from '@/store/onboarding'
import { $onboardingGate } from '@/store/onboarding-gate'
import { $onboardingSurfaces } from '@/store/onboarding-presence'
import { $activeGatewayProfile, $newChatProfile, $newChatRoute } from '@/store/profile'
import { $currentModel, $currentProvider, $gatewayState } from '@/store/session'
import {
  $focusedRuntimeId,
  $focusedSessionState,
  $focusedStoredSessionId,
  $sessionTiles
} from '@/store/session-states'

// Coarse facts only: token/tool deltas must not re-render the setup surface.
const $focusedContinuationActivity = computed($focusedSessionState, state =>
  JSON.stringify([Boolean(state?.busy || state?.turnLive || state?.awaitingResponse), state?.model, state?.provider, Boolean(state)])
)

export function useContinuation() {
  const runtimeId = useStore($focusedRuntimeId)
  const storedId = useStore($focusedStoredSessionId)
  const activity = useStore($focusedContinuationActivity)
  const gatewayState = useStore($gatewayState)
  const refresh = useStore($continuationRefresh)
  const signIn = useStore($freeTierSignIn).status
  const setup = useStore($desktopOnboarding).flow.status
  useStore($activeConnectionId)
  useStore($activeGatewayProfile)
  useStore($newChatProfile)
  useStore($newChatRoute)
  useStore($currentModel)
  useStore($currentProvider)
  useStore($sessionTiles)
  useStore($onboardingGate)
  useStore($onboardingSurfaces)
  useStore($freeTierContinuation)

  const target = continuationTarget(runtimeId ?? storedId)
  const key = target ? continuationKey(target) : ''
  const suppressed = continuationSuppressed()
  const [running, , , hydrated] = JSON.parse(activity) as [boolean, string?, string?, boolean?]
  const busy = running || Boolean((runtimeId || storedId) && !hydrated)

  useEffect(() => {
    if (!target || suppressed || busy || gatewayState !== 'open') return

    const request = continuationRequester(target)
    const pull = () => void refreshContinuation(target, request)
    pull()
    const onFocus = () => pull()
    window.addEventListener('focus', onFocus)

    return () => {
      window.removeEventListener('focus', onFocus)
    }
    // The serialized key contains the complete owner + route. No mutable
    // ambient requester survives a profile/connection or focused-tile switch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, activity, gatewayState, refresh, suppressed, busy, signIn, setup])

  return { target, required: gatewayState === 'open' && !suppressed && !busy && continuationRequired(target) }
}
