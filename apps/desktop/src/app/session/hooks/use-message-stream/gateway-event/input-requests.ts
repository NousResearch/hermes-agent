import type { ConnectionRequestPayload, ConnectionUpdatePayload, GatewayEvent } from '@hermes/shared'

import { pendingClarifyToolPayload } from '@/app/session/hooks/use-session-actions/restore-pending-clarify'
import { connectionRequestToolPayload } from '@/app/session/hooks/use-session-actions/restore-pending-connection'
import { translateNow } from '@/i18n'
import { settlePendingClarifyToolCall } from '@/lib/chat-messages'
import { $clarifyRequests, clearClarifyRequest } from '@/store/clarify'
import { normalizeConnectionRequest, setConnectionRequest, updateConnectionRequest } from '@/store/connection-request'
import type { ScopedServerRequest } from '@/store/gateway'
import { dispatchNativeNotification } from '@/store/native-notifications'
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

import { clarify as handleClarifyServerRequest } from './server-requests'
import type { GatewayEventContext } from './types'

type ConnectionRequestEvent = GatewayEvent<'connection.request'> & { payload: ConnectionRequestPayload }
type ConnectionUpdateEvent = GatewayEvent<'connection.update'> & { payload: ConnectionUpdatePayload }

const isConnectionRequestEvent = (event: GatewayEvent): event is ConnectionRequestEvent =>
  event.type === 'connection.request' && event.payload !== undefined

const isConnectionUpdateEvent = (event: GatewayEvent): event is ConnectionUpdateEvent =>
  event.type === 'connection.update' && event.payload !== undefined

/** Settle the parked clarify tool call carrying `id` (clear the card and seal
 *  its transcript row with a result). Returns whether `id` matched the parked
 *  clarify — shared by `request.cancel` (v0.21.3) and the legacy `clarify.expire`
 *  plain event (v0.21.2). Correlated by id so a delayed settle for an older
 *  prompt cannot erase a newer one the same session raised. */
function settleClarifyToolCall(ctx: GatewayEventContext, id: string): boolean {
  const { deps, sessionId, occurredAt } = ctx
  const key = sessionId ?? ''
  const request = $clarifyRequests.get()[key]

  if (request?.requestId !== id) {
    return false
  }

  clearClarifyRequest(id, sessionId)

  if (sessionId) {
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

/** Re-shape a v0.21.2 `clarify.request` plain event into the `ScopedServerRequest`
 *  the clarify server-request handler already understands.
 *
 *  A plain-event clarify carries no JSON-RPC request frame, so there is no id the
 *  shared channel can answer over the wire. The desktop cannot fabricate a
 *  response frame for an id it never issued (apps/desktop-only shim), so the card
 *  is display-only: an unanswered v0.21.2 clarify still resolves via the backend's
 *  own timeout, which arrives here as `clarify.expire` and seals the row. */
function legacyClarifyRequest(
  requestId: string,
  payload: Record<string, unknown>,
  sessionId: string | null,
  profile: string | undefined
): ScopedServerRequest {
  return {
    fail: () => undefined,
    id: requestId,
    method: 'clarify',
    params: { ...payload, ...(sessionId ? { session_id: sessionId } : {}) },
    profile: profile || 'default',
    replayed: false,
    respond: () => undefined
  }
}

/** The blocking-input family arrives as server→client REQUESTS (see
 *  `server-requests.ts`); the EVENTS in the family are `request.cancel` (v0.21.3
 *  withdrawing an open request on timeout / interrupt / session close) and the
 *  legacy `clarify.request` / `clarify.expire` pair a v0.21.2 backend emits
 *  instead of the JSON-RPC clarify request/cancel frames. */
export function handleInputRequestEvent(ctx: GatewayEventContext): boolean {
  const { deps, event, payload, sessionId } = ctx

  if (isConnectionRequestEvent(event)) {
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
    updateConnectionRequest(sessionId ?? null, event.payload)

    if (event.payload.settled && sessionId) {
      deps.updateSessionState(sessionId, state => ({ ...state, needsInput: false }))
    }

    return true
  }

  // Legacy v0.21.2 event types are not in the current gateway contract, so the
  // comparisons below read `event.type` as a plain string.
  const eventType: string = event.type

  // Backwards compat: v0.21.2 backends emit clarify as a plain `clarify.request`
  // event (via `_emit`) rather than as a JSON-RPC server request (the srq-<n>
  // frame the v0.21.3 desktop registers a handler for). Re-shape it into that
  // handler so the card still appears. Newer backends never send this event.
  if (eventType === 'clarify.request') {
    const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

    if (requestId) {
      handleClarifyServerRequest({
        deps,
        isActiveSession: ctx.isActiveEvent,
        request: legacyClarifyRequest(requestId, payload ?? {}, sessionId, event.profile),
        sessionId: sessionId ?? ''
      })
    }

    return true
  }

  // The v0.21.2 counterpart of `request.cancel` for clarify: settle the sealed
  // transcript row with a result so it renders the outcome instead of the
  // "Result unavailable" a bare timeout leaves behind (bug 2).
  if (eventType === 'clarify.expire') {
    const expireId =
      (typeof payload?.request_id === 'string' && payload.request_id) ||
      (typeof payload?.id === 'string' && payload.id) ||
      ''

    if (expireId) {
      forgetServerRequest(expireId)
      settleClarifyToolCall(ctx, expireId)
    }

    return true
  }

  if (event.type !== 'request.cancel') {
    return false
  }

  const id = typeof payload?.id === 'string' ? payload.id : ''

  if (!id) {
    return true
  }

  forgetServerRequest(id)

  const key = sessionId ?? ''

  if (settleClarifyToolCall(ctx, id)) {
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
  }

  return true
}
