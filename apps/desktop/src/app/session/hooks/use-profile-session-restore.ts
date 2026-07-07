import { useStore } from '@nanostores/react'
import type { MutableRefObject } from 'react'
import { useCallback, useEffect, useRef } from 'react'

import { $activeGatewayProfile, $freshSessionRequest, $profileSwitchTarget, normalizeProfileKey } from '@/store/profile'
import { $sessions } from '@/store/session'

interface UseProfileSessionRestoreArgs {
  resumeSession: (id: string, replaceRoute?: boolean) => Promise<void>
  selectedStoredSessionId: string | null
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  startFreshSessionDraft: () => void
}

export function useProfileSessionRestore({
  resumeSession,
  selectedStoredSessionId,
  selectedStoredSessionIdRef,
  startFreshSessionDraft
}: UseProfileSessionRestoreArgs): void {
  const activeGatewayProfile = useStore($activeGatewayProfile)
  const freshSessionRequest = useStore($freshSessionRequest)
  const lastFreshRef = useRef(freshSessionRequest)
  const lastSessionByProfileRef = useRef(new Map<string, string>())

  const rememberSession = useCallback(
    (sessionId: string | null) => {
      if (!sessionId) {
        return
      }

      const session = $sessions.get().find(row => row.id === sessionId || row._lineage_root_id === sessionId)
      const profile = normalizeProfileKey(session?.profile ?? activeGatewayProfile)
      lastSessionByProfileRef.current.set(profile, sessionId)
    },
    [activeGatewayProfile]
  )

  useEffect(() => {
    rememberSession(selectedStoredSessionId)
  }, [rememberSession, selectedStoredSessionId])

  useEffect(() => {
    if (freshSessionRequest === lastFreshRef.current) {
      return
    }

    lastFreshRef.current = freshSessionRequest
    rememberSession(selectedStoredSessionIdRef.current)

    const target = $profileSwitchTarget.get()
    $profileSwitchTarget.set(null)

    if (target) {
      const rememberedSession = lastSessionByProfileRef.current.get(normalizeProfileKey(target))

      if (rememberedSession) {
        void resumeSession(rememberedSession, true)

        return
      }
    }

    startFreshSessionDraft()
  }, [freshSessionRequest, rememberSession, resumeSession, selectedStoredSessionIdRef, startFreshSessionDraft])
}
