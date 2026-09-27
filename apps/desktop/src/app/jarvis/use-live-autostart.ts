import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { getRealtimeVoiceStatus } from '@/api/voice-realtime'
import { requestVoiceConversationStart } from '@/store/composer'
import { $activeConnectionId } from '@/store/connections'
import { $desktopOnboarding } from '@/store/onboarding'
import { $activeGatewayProfile } from '@/store/profile'

// Once per profile and app launch: navigating back must not reopen a call the user ended.
const attempted = new Set<string>()

export function useLiveAutostart(connected: boolean) {
  const onboarding = useStore($desktopOnboarding)
  const connectionId = useStore($activeConnectionId)
  const profile = useStore($activeGatewayProfile)
  useEffect(() => {
    const scope = `${connectionId || 'local'}:${profile || 'default'}`

    if (
      !connected ||
      !onboarding.configured ||
      onboarding.requested ||
      attempted.has(scope) ||
      !window.hermesDesktop?.api
    ) {
      return
    }

    let cancelled = false
    void getRealtimeVoiceStatus()
      .then(status => {
        if (cancelled || !status.available || attempted.has(scope)) {
          return
        }

        attempted.add(scope)
        requestVoiceConversationStart()
      })
      .catch(() => {
        // The explicit conversation button reports connection errors and permits retry.
      })

    return () => {
      cancelled = true
    }
  }, [connected, onboarding.configured, onboarding.requested, profile, connectionId])
}
