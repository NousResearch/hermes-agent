import { type MutableRefObject, useCallback } from 'react'

import { setActiveSessionId } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import type { AmbientGatewayRequest } from '../../contrib/session-rpc-dispatcher'
import type { ClientSessionState } from '../../types'
import { compressSession, ownedSessionRequest } from '../compress-session'

interface CompressConversationDeps {
  activeSessionIdRef: MutableRefObject<null | string>
  requestGateway: AmbientGatewayRequest
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: null | string
  ) => unknown
}

/** The controller's "compact this conversation" verb for the context popover.
 *
 *  Compression itself is `compressSession` (shared with `/compress`); this only
 *  binds it to the session the user clicked: the RPC goes to that session's
 *  OWNER, and a stale-runtime recovery republishes the live id only while that
 *  session is still the one on screen. */
export function useCompressConversationAction({
  activeSessionIdRef,
  requestGateway,
  updateSessionState
}: CompressConversationDeps) {
  return useCallback(
    async (sessionId: string) => {
      const storedSessionId = $sessionStates.get()[sessionId]?.storedSessionId ?? null

      return compressSession(
        {
          isActive: id => activeSessionIdRef.current === id,
          onRecovered: recoveredId => {
            if (activeSessionIdRef.current === sessionId) {
              activeSessionIdRef.current = recoveredId
              setActiveSessionId(recoveredId)
            }
          },
          request: ownedSessionRequest(sessionId, requestGateway),
          updateSessionState
        },
        { sessionId, storedSessionId }
      )
    },
    [activeSessionIdRef, requestGateway, updateSessionState]
  )
}
