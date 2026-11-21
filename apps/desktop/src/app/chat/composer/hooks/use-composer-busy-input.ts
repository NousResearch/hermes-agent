import { useCallback } from 'react'

import type { GatewayRequester } from '@/app/contrib/types'
import { useBusyInputMode } from '@/app/session/hooks/use-busy-input-mode'
import type { HermesGateway } from '@/hermes'

interface ComposerBusyInputOptions {
  gateway: HermesGateway | null | undefined
  sessionId: null | string | undefined
  queueSessionKey: null | string | undefined
}

/** What a send does while a turn runs (the backend's busy-input policy for this chat), and the
 *  gateway requester the composer's own config reads go through. */
export function useComposerBusyInput({ gateway, queueSessionKey, sessionId }: ComposerBusyInputOptions) {
  const requestBusyConfig = useCallback<GatewayRequester>(
    (method, params) => (gateway ? gateway.request(method, params) : Promise.reject(new Error('Gateway unavailable'))),
    [gateway]
  )

  const busyInputMode = useBusyInputMode({
    sessionId: sessionId ?? null,
    storedSessionId: queueSessionKey ?? null,
    requestGateway: requestBusyConfig
  })

  return { busyInputMode, requestBusyConfig }
}
