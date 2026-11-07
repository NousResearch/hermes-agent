import { useEffect, useRef } from 'react'

import type { ProfileSessionRecoveryScope } from '@/store/profile-session-recovery'

export type OnSessionsRefreshed = (listener: (scope: ProfileSessionRecoveryScope) => void) => () => void

export function useGatewayScopeRefresh(
  activeConnectionId: null | string,
  activeGatewayProfile: string,
  refreshCurrentModel: (force: boolean) => Promise<unknown>,
  refreshHermesConfig: (force: boolean) => Promise<unknown>,
  refreshSessions: () => Promise<unknown>,
  restoreMostRecentSession: (isCurrent: () => boolean) => void | Promise<unknown>,
  onSessionsRefreshed: OnSessionsRefreshed,
  getRecoveryIntent: () => string,
  onScopeRefresh?: () => void
) {
  const gatewayScope = `${activeConnectionId ?? ''}\0${activeGatewayProfile}`
  const lastGatewayScopeRef = useRef(gatewayScope)

  const callbacksRef = useRef({
    onScopeRefresh,
    refreshCurrentModel,
    refreshHermesConfig,
    refreshSessions,
    restoreMostRecentSession,
    onSessionsRefreshed,
    getRecoveryIntent
  })

  callbacksRef.current = {
    onScopeRefresh,
    refreshCurrentModel,
    refreshHermesConfig,
    refreshSessions,
    restoreMostRecentSession,
    onSessionsRefreshed,
    getRecoveryIntent
  }

  // eslint-disable-next-line no-restricted-syntax -- scope sentinel is not an atom mirror
  useEffect(() => {
    if (gatewayScope === lastGatewayScopeRef.current) {
      return
    }

    lastGatewayScopeRef.current = gatewayScope
    let cancelled = false
    const callbacks = callbacksRef.current
    const [connectionId, profile] = gatewayScope.split('\0')
    const intent = callbacks.getRecoveryIntent()
    const isCurrent = () => !cancelled && callbacksRef.current.getRecoveryIntent() === intent

    const unsubscribe = callbacks.onSessionsRefreshed(scope => {
      if ((scope.connectionId ?? '') !== connectionId || scope.profile !== profile) {
        return
      }

      // One valid publication consumes the attempt, even with no history or a draft.
      unsubscribe()

      if (isCurrent()) {
        void Promise.resolve(callbacksRef.current.restoreMostRecentSession(isCurrent)).catch(() => undefined)
      }
    })

    void callbacks.refreshCurrentModel(true)
    void callbacks.refreshHermesConfig(true)
    callbacks.onScopeRefresh?.()
    // Completion alone proves nothing: a newer background refresh can supersede this one.
    void callbacks.refreshSessions().catch(() => undefined)

    return () => {
      cancelled = true
      unsubscribe()
    }
  }, [gatewayScope])
}
