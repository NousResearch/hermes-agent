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
 * resolveSessionRpcOwner (tile route → exact unique owner hint → the row's
 * owner: exact when connection-tagged, else its profile), then a
 * cross-profile REST probe for a hidden/unlisted session. A request with a
 * session whose owner STILL cannot be named fails closed with an explicit
 * SessionOwnerResolutionError rather than riding the ambient socket (the one
 * exception: the legacy single-backend Desktop, where ambient IS the owner).
 * Only a request with NO session at all falls to the ambient socket.
 *
 * `session.resume` is the one method where failing closed is not the end of the
 * story: #94724 gave the tile-delegate path a read-only stored-transcript
 * recovery, but this dispatcher kept the bare gate, so an orphaned / owner-less
 * tile whose resume surfaced HERE dead-ended on "Couldn't open this session"
 * with a Retry that re-ran the same resume forever (#102618). The resume
 * dispatch is therefore wrapped in the same `resumeWithStoredTranscriptFallback`
 * recovery: an unresolvable owner marks the stored id read-only and answers
 * from the id-only REST transcript read (which routes no live session at all),
 * so the tile paints history instead of latching an unrecoverable error.
 */
import type { MutableRefObject } from 'react'

import { fetchStoredTranscriptAcrossBackends } from '@/api/sessions'
import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import type { ClientSessionState } from '@/app/types'
import { readOnlyRuntimeIdFor, resumeWithStoredTranscriptFallback } from '@/store/read-only-transcript'
import { isSessionGoneForBackgroundPolling } from '@/store/runtime-gone'
import { getSessionOwnerHint, knownSessionOwner, ownerLookupSessionRows, requestSessionResume } from '@/store/session'
import { assertSessionOwnerResolved } from '@/store/session-owner-resolution'
import { requestForSessionProfile, type SessionOwnerScope } from '@/store/session-request-router'
import { $focusedStoredSessionId, sessionTileOwnerRoute, storedSessionIdForRuntimeId } from '@/store/session-states'
import type { SessionResumeResponse } from '@/types/hermes'

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

/** A `session.resume` payload synthesized from a stored transcript read, for
 *  the no-owner recovery. `session_id` is the synthetic read-only runtime id
 *  (never a gateway runtime), and the stored id stays latched in
 *  `$readOnlyStoredTranscripts` so write surfaces refuse to submit into a
 *  session that has no routable runtime. */
export function readOnlyResumeResponse(
  storedSessionId: string,
  messages: SessionResumeResponse['messages']
): SessionResumeResponse {
  return {
    message_count: messages.length,
    messages,
    resumed: storedSessionId,
    running: false,
    session_id: readOnlyRuntimeIdFor(storedSessionId)
  }
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
      sessionRowOwner: storedSessionId => knownSessionOwner(ownerLookupSessionRows(), storedSessionId),
      tileOwnerRoute: sessionTileOwnerRoute
    })

    if (!owner && routingSessionId) {
      // Unknown owner for a REAL session: probe across profiles (REST, not the
      // gateway socket, so no recursion) rather than defaulting to active. A
      // hit stamps ownership on the row (exact when the row came back
      // connection-tagged); a miss leaves owner undefined.
      const probed = await resolveSessionOwner(routingSessionId)

      if (probed) {
        owner = probed
      }
    }

    const dispatch = async <TResult>(): Promise<TResult> => {
      try {
        return await requestForSessionProfile<TResult>(owner, ambientRequest, method, params ?? {}, timeoutMs, signal)
      } catch (error) {
        // A missed session.reclaimed leaves later RPCs answering 4001 against a
        // still-resumable stored row. Prompt actions already retry their own
        // calls; this seam covers the other session-scoped callers and wakes
        // route-resume for the visible main session only. Do not retry the
        // failing RPC — it may be destructive, and a fresh binding is async.
        // A session the user just deleted is filtered by requestSessionResume,
        // which drops resume requests for a removal-pending id.
        if (
          method !== 'session.resume' &&
          method !== 'session.activate' &&
          paramSessionId &&
          routingSessionId &&
          routingSessionId === selectedStoredSessionIdRef.current &&
          isSessionGoneForBackgroundPolling(error)
        ) {
          requestSessionResume(routingSessionId, typeof owner === 'object' && owner ? owner : undefined)
        }

        throw error
      }
    }

    // A request that names a session but whose owner nobody can name must not
    // ride the ambient socket: that turns missing metadata into a misleading
    // backend "session not found" on a backend that never held the runtime.
    //
    // `session.resume` is the exception, and only in the recovery direction
    // (#102618): failing closed here is right, but dead-ending is not — the
    // stored transcript is reachable over an id-only REST read that routes no
    // live session, so recover into a read-only open exactly as the
    // tile-delegate path already does (#94724). Every other method, and every
    // other failure mode of resume, keeps the bare fail-closed gate.
    const gatedSessionId = paramSessionId ? routingSessionId : null

    if (method === 'session.resume' && gatedSessionId) {
      const outcome = await resumeWithStoredTranscriptFallback(
        gatedSessionId,
        async () => {
          assertSessionOwnerResolved(owner, { method, sessionId: gatedSessionId })

          return dispatch<T>()
        },
        async () => {
          const stored = await fetchStoredTranscriptAcrossBackends(gatedSessionId)

          if (!stored) {
            throw new Error('stored transcript unavailable on every reachable backend')
          }

          return stored
        }
      )

      return outcome.mode === 'live'
        ? outcome.resumed
        : (readOnlyResumeResponse(gatedSessionId, outcome.transcript.messages ?? []) as T)
    }

    assertSessionOwnerResolved(owner, { method, sessionId: gatedSessionId })

    return dispatch<T>()
  }
}
