import { pendingClarifyToolPayload } from '@/app/session/hooks/use-session-actions/restore-pending-clarify'
import { settlePendingClarifyToolCall } from '@/lib/chat-messages'
import { $clarifyRequests, clearClarifyRequest } from '@/store/clarify'
import { $mcpSetupRequests, clearMcpSetupRequest } from '@/store/mcp-setup'
import {
  $approvalRequests,
  $secretRequests,
  $sudoRequests,
  $vaultCodeRequests,
  $vaultSaveLoginRequests,
  $vaultUnlockRequests,
  clearApprovalRequest,
  clearSecretRequest,
  clearSudoRequest,
  clearVaultCodeRequest,
  clearVaultSaveLoginRequest,
  clearVaultUnlockRequest
} from '@/store/prompts'
import { forgetServerRequest } from '@/store/server-requests'

import type { GatewayEventContext } from './types'

/** The blocking-input family arrives as server→client REQUESTS (see
 *  `server-requests.ts`); the one EVENT in the family is `request.cancel`, the
 *  backend withdrawing an open request (timeout / interrupt / session close):
 *  tear down whichever parked card carries that id. Cancel is request-correlated:
 *  a delayed cancel for an older prompt must not erase a newer one the same
 *  session raised. */
export function handleInputRequestEvent(ctx: GatewayEventContext): boolean {
  const { deps, event, payload, sessionId, occurredAt } = ctx

  if (event.type !== 'request.cancel') {
    return false
  }

  const id = typeof payload?.id === 'string' ? payload.id : ''

  if (!id) {
    return true
  }

  forgetServerRequest(id)

  const key = sessionId ?? ''

  if ($clarifyRequests.get()[key]?.requestId === id) {
    const request = $clarifyRequests.get()[key]

    clearClarifyRequest(id, sessionId)

    if (sessionId && request) {
      deps.updateSessionState(sessionId, state => {
        const projection = settlePendingClarifyToolCall(
          state.messages,
          pendingClarifyToolPayload(request),
          state.busy,
          occurredAt
        )

        return {
          ...state,
          messages: projection.messages,
          needsInput: false,
          streamId: state.busy ? (projection.streamId ?? state.streamId) : null
        }
      })
    }

    return true
  }

  if ($approvalRequests.get()[key]?.serverRequestId === id) {
    clearApprovalRequest(sessionId, $approvalRequests.get()[key]?.requestId)
  } else if ($sudoRequests.get()[key]?.requestId === id) {
    clearSudoRequest(sessionId, id)
  } else if ($secretRequests.get()[key]?.requestId === id) {
    clearSecretRequest(sessionId, id)
  } else if ($vaultCodeRequests.get()[key]?.requestId === id) {
    clearVaultCodeRequest(sessionId, id)
  } else if ($vaultSaveLoginRequests.get()[key]?.requestId === id) {
    clearVaultSaveLoginRequest(sessionId, id)
  } else if ($vaultUnlockRequests.get()[key]?.requestId === id) {
    clearVaultUnlockRequest(sessionId, id)
  } else if ($mcpSetupRequests.get()[key]?.requestId === id) {
    clearMcpSetupRequest(id, sessionId)
  }

  return true
}
