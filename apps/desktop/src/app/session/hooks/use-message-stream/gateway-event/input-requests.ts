import type { ConnectionRequestPayload, ConnectionUpdatePayload, GatewayEvent } from '@hermes/shared'

import { applyAccountConnectionUpdate } from '@/app/capabilities/connectors/data/account-operations'
import { abortPreviewTyping } from '@/app/chat/right-rail/preview-typing-abort'
import { pendingClarifyToolPayload } from '@/app/session/hooks/use-session-actions/restore-pending-clarify'
import { connectionRequestToolPayload } from '@/app/session/hooks/use-session-actions/restore-pending-connection'
import { translateNow } from '@/i18n'
import { settlePendingClarifyToolCall, textPart } from '@/lib/chat-messages'
import { $clarifyRequests, clearClarifyRequest } from '@/store/clarify'
import { normalizeConnectionRequest, setConnectionRequest, updateConnectionRequest } from '@/store/connection-request'
import { dispatchNativeNotification } from '@/store/native-notifications'
import { notify } from '@/store/notifications'
import {
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
  clearVaultUnlockRequest,
  sessionApprovalRequests
} from '@/store/prompts'
import { requestRoute } from '@/store/recovery-requests'
import { forgetServerRequest } from '@/store/server-requests'

import type { GatewayEventContext } from './types'

/** Settings → Safety, where `approvals.timeout` lives (settings/constants.ts). */
const SAFETY_SETTINGS_ROUTE = '/settings?tab=config:safety'

type ConnectionRequestEvent = GatewayEvent<'connection.request'> & { payload: ConnectionRequestPayload }
type ConnectionUpdateEvent = GatewayEvent<'connection.update'> & { payload: ConnectionUpdatePayload }

const isConnectionRequestEvent = (event: GatewayEvent): event is ConnectionRequestEvent =>
  event.type === 'connection.request' && event.payload !== undefined

const isConnectionUpdateEvent = (event: GatewayEvent): event is ConnectionUpdateEvent =>
  event.type === 'connection.update' && event.payload !== undefined

/** `approval.settled` / `clarify.settled {prompt_id}`: take down the card
 *  parked under that server request id. Returns false for other events. */
function handlePromptSettledEvent(ctx: GatewayEventContext): boolean {
  const { deps, event, payload, sessionId } = ctx

  if (event.type !== 'approval.settled' && event.type !== 'clarify.settled') {
    return false
  }

  const promptId = (payload as { prompt_id?: unknown } | undefined)?.prompt_id

  if (typeof promptId !== 'string' || !promptId) {
    return true
  }

  forgetServerRequest(promptId)
  const key = sessionId ?? ''

  if (event.type === 'clarify.settled') {
    if ($clarifyRequests.get()[key]?.requestId === promptId) {
      clearClarifyRequest(promptId, sessionId)

      if (sessionId) {
        deps.updateSessionState(sessionId, state => ({ ...state, needsInput: false }))
      }
    }

    return true
  }

  const approval = sessionApprovalRequests(sessionId ?? null)
    .get()
    .find(request => request.serverRequestId === promptId)

  if (approval) {
    clearApprovalRequest(sessionId, approval.requestId)
  }

  return true
}

/** The blocking-input family arrives as server→client REQUESTS (see
 *  `server-requests.ts`); the one EVENT in the family is `request.cancel`, the
 *  backend withdrawing an open request (timeout / interrupt / session close):
 *  tear down whichever parked card carries that id. Cancel is request-correlated:
 *  a delayed cancel for an older prompt must not erase a newer one the same
 *  session raised. */
export function handleInputRequestEvent(ctx: GatewayEventContext): boolean {
  const { deps, event, payload, sessionId, occurredAt } = ctx

  if (isConnectionRequestEvent(event)) {
    // An interrupted/deleted session's runtime has no turn left to consent to a
    // connection; the backend withdraws its request on the same boundary. Drop
    // the frame rather than parking a stale consent card (#75587).
    if (sessionId && deps.sessionInterrupted(sessionId)) {
      return true
    }

    // Park per-session and upsert a stable tool row so the card renders even if tool.start was missed.
    const request = normalizeConnectionRequest(event.payload, sessionId ?? null)

    if (request) {
      setConnectionRequest(request)

      if (sessionId) {
        deps.upsertToolCall(sessionId, connectionRequestToolPayload(request), 'running')
        deps.updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
      }

      dispatchNativeNotification({
        body: request.targets.map(target => target.name).join(', '),
        kind: 'input',
        sessionId,
        title: translateNow('notifications.native.inputTitle')
      })
    }

    return true
  }

  if (isConnectionUpdateEvent(event)) {
    if (event.payload.owner.type === 'account') {
      applyAccountConnectionUpdate(event.payload)

      return true
    }

    updateConnectionRequest(sessionId ?? null, event.payload)

    if (event.payload.settled && sessionId) {
      deps.updateSessionState(sessionId, state => ({ ...state, needsInput: false }))
    }

    return true
  }

  // Canonical gateways settle a shared prompt with `approval.settled` /
  // `clarify.settled {prompt_id}` once ANY attached viewer answered it (or the
  // turn ended). The prompt id is the server request id this window parked the
  // card under. Only the parked card comes down: the answer itself reaches the
  // transcript through the tool's own completion, so no "skipped" projection.
  if (handlePromptSettledEvent(ctx)) {
    return true
  }

  if (event.type !== 'request.cancel') {
    return false
  }

  const id = typeof payload?.id === 'string' ? payload.id : ''

  if (!id) {
    return true
  }

  // preview.act has no card. A timeout or interrupt still has to stop keystrokes
  // already queued for that type.
  abortPreviewTyping(id, typeof payload?.reason === 'string' ? payload.reason : 'interrupted')

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

  const approval = sessionApprovalRequests(sessionId ?? null)
    .get()
    .find(request => request.serverRequestId === id)

  if (approval) {
    clearApprovalRequest(sessionId, approval.requestId)

    // The Run/Reject bar vanishing is the only thing the user would otherwise
    // see; the tool row then shows a model-facing "BLOCKED" result. Say what
    // happened in human terms and point at the setting that controls the wait.
    if (payload?.reason === 'timeout' && sessionId) {
      const line = translateNow('assistant.approval.timedOutSystemLine')

      deps.flushQueuedDeltas(sessionId)
      deps.updateSessionState(sessionId, state => ({
        ...state,
        messages: [
          ...state.messages,
          { id: `approval-timeout-${id}`, role: 'system', parts: [textPart(line, occurredAt)], timestamp: occurredAt }
        ]
      }))
      notify({
        kind: 'warning',
        message: line,
        action: {
          label: translateNow('assistant.approval.openSafetySettings'),
          onClick: () => requestRoute(SAFETY_SETTINGS_ROUTE)
        }
      })
    }
  } else {
    cancelParkedCredentialRequest(key, sessionId, id)
  }

  return true
}

/** Withdraw the sudo / secret / vault card parked under `id`, first match wins. */
function cancelParkedCredentialRequest(key: string, sessionId: GatewayEventContext['sessionId'], id: string): void {
  if ($sudoRequests.get()[key]?.requestId === id) {
    clearSudoRequest(sessionId, id)
  } else if ($sudoRequests.get()['']?.requestId === id) {
    clearSudoRequest(null, id) // the app-level Bot Screen install card: not owned by any chat
  } else if ($secretRequests.get()[key]?.requestId === id) {
    clearSecretRequest(sessionId, id)
  } else if ($vaultCodeRequests.get()[key]?.requestId === id) {
    clearVaultCodeRequest(sessionId, id)
  } else if ($vaultSaveLoginRequests.get()[key]?.requestId === id) {
    clearVaultSaveLoginRequest(sessionId, id)
  } else if ($vaultUnlockRequests.get()[key]?.requestId === id) {
    clearVaultUnlockRequest(sessionId, id)
  }
}
