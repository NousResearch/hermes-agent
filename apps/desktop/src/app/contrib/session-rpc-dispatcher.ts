/**
 * The window's ONE session-scoped RPC dispatcher, factored out of the contrib
 * wiring controller so the exact production routing (not a re-implementation)
 * can be driven by integration tests alongside the real session/prompt hooks.
 *
 * Route each RPC by the session IT targets, not by whatever tile is focused.
 * `requestGateway` is one shared closure used for every session RPC in the
 * window; keying the owner off $focusedStoredSessionId sent a NON-focused
 * tile's RPC (any bot chat while another pane is active) to the focused tile's
 * backend. That is the Bot Mode bug: a bot's prompt.submit carried its own
 * session_id but ran on the default backend (served via ?profile= from the
 * default's state.db), or 4001'd when the default backend didn't hold the
 * runtime session.
 *
 * params.session_id is a RUNTIME id, while tiles and session rows key on the
 * STORED id, so translate first (state cache, then a reverse scan of the
 * stored->runtime map, then the persisted tile map — the same ladder
 * use-session-tile-delegate uses, plus the tile rung that survives a reload
 * when the state cache is cold). A miss on ALL rungs means the id is already a
 * stored id (several RPCs pass stored ids directly), so use it as-is. Only an
 * RPC with no session_id at all (ambient/config calls) keeps the focused-tile
 * route.
 *
 * Session-scoped RPCs route to the backend that OWNS the session — never to
 * whatever is "active" (active is presentation only). The owner ladder is
 * resolveSessionRpcOwner (tile route → exact unique owner hint → row profile),
 * then a cross-profile REST probe for a hidden/unlisted session. Only a
 * request with NO session at all falls to the ambient socket.
 */
import type { MutableRefObject } from 'react'

import { resolveSessionProfile } from '@/app/session/hooks/use-session-actions/utils'
import type { ClientSessionState } from '@/app/types'
import { $sessions, getSessionOwnerHint, knownSessionOwner } from '@/store/session'
import { requestForSessionProfile, type SessionOwnerScope } from '@/store/session-request-router'
import { $focusedStoredSessionId, sessionTileOwnerRoute, storedSessionIdForRuntimeId } from '@/store/session-states'

import { findStoredIdForRuntimeId, resolveRoutingSessionId, resolveSessionRpcOwner } from './wiring-routing'

export type AmbientGatewayRequest = <T>(
  method: string,
  params?: Record<string, unknown>,
  timeoutMs?: number,
  signal?: AbortSignal
) => Promise<T>

export interface SessionRpcDispatcherDeps {
  ambientRequest: AmbientGatewayRequest
  runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>>
  selectedStoredSessionIdRef: MutableRefObject<null | string>
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
}

export function createSessionRpcDispatcher(deps: SessionRpcDispatcherDeps): AmbientGatewayRequest {
  const { ambientRequest, runtimeIdByStoredSessionIdRef, selectedStoredSessionIdRef, sessionStateByRuntimeIdRef } = deps

  return async <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number, signal?: AbortSignal) => {
    const paramSessionId = typeof params?.session_id === 'string' && params.session_id ? params.session_id : undefined

    const routingSessionId = resolveRoutingSessionId({
      focusedStoredSessionId: $focusedStoredSessionId.get(),
      paramSessionId,
      selectedStoredSessionId: selectedStoredSessionIdRef.current,
      storedIdForRuntime: runtimeId =>
        sessionStateByRuntimeIdRef.current.get(runtimeId)?.storedSessionId ??
        findStoredIdForRuntimeId(runtimeIdByStoredSessionIdRef.current, runtimeId) ??
        storedSessionIdForRuntimeId(runtimeId) ??
        undefined
    })

    let owner: SessionOwnerScope = resolveSessionRpcOwner({
      routingSessionId,
      sessionOwnerHint: storedSessionId => getSessionOwnerHint(storedSessionId),
      sessionRowOwner: storedSessionId => knownSessionOwner($sessions.get(), storedSessionId),
      tileOwnerRoute: sessionTileOwnerRoute
    })

    if (!owner && routingSessionId) {
      // Unknown owner for a REAL session: probe across profiles (REST, not the
      // gateway socket, so no recursion) rather than defaulting to active. A
      // hit stamps ownership + caches a hint; a miss leaves owner undefined
      // and the request falls to ambient, exactly as an unroutable session did
      // before — but only after we tried, never as a silent active fallback.
      const probed = await resolveSessionProfile(routingSessionId)

      if (probed) {
        owner = probed
      }
    }

    return requestForSessionProfile<T>(owner, ambientRequest, method, params ?? {}, timeoutMs, signal)
  }
}
