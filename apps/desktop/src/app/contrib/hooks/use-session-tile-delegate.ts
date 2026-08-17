import { useEffect } from 'react'

import { getLatestSessionMessages, PROMPT_SUBMIT_REQUEST_TIMEOUT_MS } from '@/hermes'
import { toChatMessages } from '@/lib/chat-messages'
import { publishSessionState, setSessionTileDelegate } from '@/store/session-states'
import type { SessionResumeResponse } from '@/types/hermes'

import type { usePromptActions } from '../../session/hooks/use-prompt-actions'
import { markSessionRecentlyInterrupted, withSessionNotFoundResume } from '../../session/hooks/use-prompt-actions/utils'
import { resolveSessionProfile } from '../../session/hooks/use-session-actions/utils'
import type { useSessionStateCache } from '../../session/hooks/use-session-state-cache'
import type { GatewayRequester } from '../types'

type SessionStateCache = ReturnType<typeof useSessionStateCache>

interface SessionTileDelegateParams {
  archiveSession: (storedSessionId: string) => Promise<unknown>
  branchStoredSession: (storedSessionId: string) => Promise<unknown>
  executeSlashCommand: ReturnType<typeof usePromptActions>['executeSlashCommand']
  removeSession: (storedSessionId: string) => Promise<unknown>
  requestGateway: GatewayRequester
  runtimeIdByStoredSessionIdRef: SessionStateCache['runtimeIdByStoredSessionIdRef']
  sessionStateByRuntimeIdRef: SessionStateCache['sessionStateByRuntimeIdRef']
  updateSessionState: SessionStateCache['updateSessionState']
}

/**
 * Publishes the session-tile delegate: resume / submit / interrupt / slash for
 * tiled sessions WITHOUT touching the primary view ($activeSessionId /
 * $messages stay the main thread's). Resume reuses a live runtime binding when
 * one exists (incl. the main thread's own session); a cold tile binds +
 * hydrates the cache, which publishSessionState mirrors to the tile.
 */
export function useSessionTileDelegate({
  archiveSession,
  branchStoredSession,
  executeSlashCommand,
  removeSession,
  requestGateway,
  runtimeIdByStoredSessionIdRef,
  sessionStateByRuntimeIdRef,
  updateSessionState
}: SessionTileDelegateParams): void {
  useEffect(() => {
    // A tile's runtime binding can die the same way the foreground's does
    // (sleep/wake, backend restart). The cache maps stored -> runtime, so walk
    // it backwards to find the durable id this runtime belongs to.
    const storedSessionIdForRuntime = (runtimeId: string): null | string => {
      const cached = sessionStateByRuntimeIdRef.current.get(runtimeId)?.storedSessionId

      if (cached) {
        return cached
      }

      for (const [storedId, mapped] of runtimeIdByStoredSessionIdRef.current) {
        if (mapped === runtimeId) {
          return storedId
        }
      }

      return null
    }

    // Repoint the stored -> runtime mapping at the recovered id so subsequent
    // tile actions use the live binding instead of re-recovering every call.
    const rebindTileRuntime = (deadRuntimeId: string) => (recoveredId: string) => {
      const storedId = storedSessionIdForRuntime(deadRuntimeId)

      if (storedId) {
        runtimeIdByStoredSessionIdRef.current.set(storedId, recoveredId)
      }
    }

    setSessionTileDelegate({
      archiveSession: async storedSessionId => {
        await archiveSession(storedSessionId)
      },
      branchSession: async storedSessionId => {
        await branchStoredSession(storedSessionId)
      },
      deleteSession: async storedSessionId => {
        await removeSession(storedSessionId)
      },
      executeSlash: async (rawCommand, sessionId) => {
        await executeSlashCommand(rawCommand, { sessionId })
      },
      // Gateway reconnect (sleep/wake, backend respawn): every stored→runtime
      // binding recorded pre-reconnect points at a runtime id the respawned
      // backend no longer knows. Drop the map so resumeTile's warm path can't
      // re-bind a tile to a dead runtime; live bindings re-record from
      // post-reconnect events and fresh resumes.
      invalidateRuntimeBindings: () => {
        runtimeIdByStoredSessionIdRef.current.clear()
      },
      interruptSession: async runtimeId => {
        // Same cooldown as the primary chat's Stop (#83855): the gateway may
        // still be winding down after this interrupt, so a quick edit/resend
        // on the tile must go interrupt-first even though busy already reads
        // false. Mark the runtime id (and any recovered id) before the RPC so
        // the window covers the whole wind-down.
        markSessionRecentlyInterrupted(runtimeId)
        await withSessionNotFoundResume(
          runtimeId,
          storedSessionIdForRuntime(runtimeId),
          liveId => requestGateway('session.interrupt', { session_id: liveId }),
          {
            requestGateway,
            onRecovered: recoveredId => {
              markSessionRecentlyInterrupted(recoveredId)
              rebindTileRuntime(runtimeId)(recoveredId)
            }
          }
        )
      },
      resumeTile: async storedSessionId => {
        const existing = runtimeIdByStoredSessionIdRef.current.get(storedSessionId)
        const cached = existing ? sessionStateByRuntimeIdRef.current.get(existing) : undefined

        // Warm path: reuse a live binding — but only when it still carries a
        // transcript (or is mid-turn, where messages legitimately stream in).
        // A binding whose cached state has no messages is either a released
        // transcript or a stale pre-reconnect survivor; reusing it painted the
        // post-sleep/wake tile permanently empty. Fall through to a real
        // resume instead — it's idempotent for a genuinely live session.
        if (
          existing &&
          cached?.storedSessionId === storedSessionId &&
          (cached.busy || cached.messages.length > 0)
        ) {
          publishSessionState(existing, cached)

          return existing
        }

        // Resolve the owning profile before binding a runtime. A tile can open a
        // session from any profile, not just the active one; resuming (or
        // reading messages) without a profile lets the gateway fall back to the
        // launch-profile DB and fork the conversation into the wrong profile —
        // the same cross-profile bleed the recovery resumes had (#67603).
        const profile = await resolveSessionProfile(storedSessionId)

        const [prefetch, resumed] = await Promise.all([
          getLatestSessionMessages(storedSessionId, profile).catch(() => null),
          requestGateway<SessionResumeResponse>('session.resume', {
            session_id: storedSessionId,
            cols: 96,
            omit_messages: true,
            ...(profile ? { profile } : {})
          })
        ])

        const runtimeId = resumed?.session_id

        if (!runtimeId) {
          throw new Error('resume returned no session id')
        }

        updateSessionState(
          runtimeId,
          state => ({
            ...state,
            busy: Boolean(resumed?.info?.running),
            messages:
              state.messages.length > 0 ? state.messages : toChatMessages(prefetch?.messages ?? resumed?.messages ?? [])
          }),
          storedSessionId
        )

        return runtimeId
      },
      submitToSession: async (runtimeId, text) => {
        await withSessionNotFoundResume(
          runtimeId,
          storedSessionIdForRuntime(runtimeId),
          liveId => requestGateway('prompt.submit', { session_id: liveId, text }, PROMPT_SUBMIT_REQUEST_TIMEOUT_MS),
          { requestGateway, onRecovered: rebindTileRuntime(runtimeId) }
        )
      },
      updateSession: (runtimeId, updater) => updateSessionState(runtimeId, updater)
    })
  }, [
    archiveSession,
    branchStoredSession,
    executeSlashCommand,
    removeSession,
    requestGateway,
    runtimeIdByStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    updateSessionState
  ])
}
