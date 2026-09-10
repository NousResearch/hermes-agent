import { useEffect } from 'react'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import {
  fetchStoredTranscriptAcrossBackends,
  getLatestSessionMessages,
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS
} from '@/hermes'
import { translateNow } from '@/i18n/runtime'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { notify } from '@/store/notifications'
import {
  isReadOnlyRuntimeId,
  readOnlyRuntimeIdFor,
  resumeWithStoredTranscriptFallback
} from '@/store/read-only-transcript'
import { knownSessionOwner, ownerLookupSessionRows } from '@/store/session'
import { assertSessionOwnerResolved } from '@/store/session-owner-resolution'
import { requestForSessionProfile, type SessionOwnerScope } from '@/store/session-request-router'
import {
  $sessionTiles,
  publishSessionState,
  sessionTileOwnerRoute,
  setSessionTileDelegate
} from '@/store/session-states'
import type { SessionResumeResponse } from '@/types/hermes'

import type { usePromptActions } from '../../session/hooks/use-prompt-actions'
import { singleFlightSessionResume } from '../../session/hooks/use-prompt-actions/single-flight-resume'
import { markSessionRecentlyInterrupted, withSessionNotFoundResume } from '../../session/hooks/use-prompt-actions/utils'
import {
  chatMessageArraysEquivalent,
  preserveLocalPendingTurnMessages,
  reconcileResumeMessages,
  resolveSessionOwner
} from '../../session/hooks/use-session-actions/utils'
import type { useSessionStateCache } from '../../session/hooks/use-session-state-cache'
import type { GatewayRequester } from '../types'

type SessionStateCache = ReturnType<typeof useSessionStateCache>

function mergeTileTranscript(
  previous: ChatMessage[],
  prefetchMessages: SessionResumeResponse['messages'] | undefined,
  streamId?: null | string
): ChatMessage[] {
  const prefetched = toChatMessages(prefetchMessages ?? [])

  if (!prefetched.length) {
    return previous
  }

  const persisted = graftRefreshedTailOntoBackfill(prefetched, previous)

  // The known stream belongs to this turn even when its text repeats an older
  // answer; the generic reconnect reconciler only has text/ordinal heuristics.
  const stream = previous.find(message => message.id === streamId)

  const merged = preserveLocalPendingTurnMessages(
    reconcileResumeMessages(persisted, previous),
    stream ? previous.filter(message => message !== stream) : previous
  )

  if (!stream) {
    return merged
  }

  // Compaction shifts global assistant ordinals. Anchor this turn at its user
  // row instead; an older answer sharing the stream's prefix is not a match.
  const beforeStream = previous.slice(0, previous.indexOf(stream))
  const user = beforeStream.findLast(message => message.role === 'user')

  let anchor = user
    ? persisted.findIndex(
        message => message.id === user.id || (user.rowId !== undefined && message.rowId === user.rowId)
      )
    : -1

  if (user && anchor < 0) {
    const matches = persisted.filter(
      message => message.role === 'user' && chatMessageText(message) === chatMessageText(user)
    )

    anchor = matches.length === 1 ? persisted.indexOf(matches[0]) : -1
  }

  const turn = anchor < 0 ? [] : persisted.slice(anchor + 1)
  const nextUser = turn.findIndex(message => message.role === 'user')

  const ordinal = user
    ? beforeStream.slice(beforeStream.indexOf(user) + 1).filter(message => message.role === 'assistant').length
    : 0

  const counterpart =
    persisted.find(
      message => message.id === stream.id || (stream.rowId !== undefined && message.rowId === stream.rowId)
    ) ?? (nextUser < 0 ? turn : turn.slice(0, nextUser)).filter(message => message.role === 'assistant')[ordinal]

  const localText = chatMessageText(stream)
  const storedText = counterpart ? chatMessageText(counterpart) : ''

  if (!counterpart || !(storedText.startsWith(localText) || localText.startsWith(storedText))) {
    return [...merged, stream]
  }

  // Keep the stream id for subsequent deltas, but use REST's fuller answer.
  const reply =
    storedText.length > localText.length
      ? { ...reconcileResumeMessages([counterpart], [stream])[0], id: stream.id }
      : stream

  return merged.map((message, index) => (index === persisted.indexOf(counterpart) ? reply : message))
}

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

    // Same ladder as the window's session-RPC dispatcher: tile route → the
    // row's owner (exact when connection-tagged, else the hint / profile) →
    // the async cross-profile probe (exact when the resolved row is tagged).
    const ownerForStoredSession = async (storedSessionId: string): Promise<SessionOwnerScope> => {
      const owner =
        sessionTileOwnerRoute(storedSessionId) ??
        knownSessionOwner(ownerLookupSessionRows(), storedSessionId) ??
        (await resolveSessionOwner(storedSessionId))

      return owner
    }

    const requestForStoredSession = async <T>(
      storedSessionId: string,
      method: string,
      params: Record<string, unknown>,
      timeoutMs?: number
    ): Promise<T> => {
      const owner = await ownerForStoredSession(storedSessionId)

      return requestForSessionProfile<T>(owner, requestGateway, method, params, timeoutMs)
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
      invalidateRuntimeBindings: preserveStoredSessionIds => {
        for (const storedSessionId of runtimeIdByStoredSessionIdRef.current.keys()) {
          if (!preserveStoredSessionIds?.has(storedSessionId)) {
            runtimeIdByStoredSessionIdRef.current.delete(storedSessionId)
          }
        }
      },
      // Reconnect reconcile (#93059): retire an orphaned runtime's busy claim
      // through updateSessionState so the cache, focused view, busyRef and
      // tile mirrors settle together. A runtime this cache never held reports
      // false instead of minting an entry; the store downgrades its mirror.
      retireBusyClaim: runtimeId => {
        const cached = sessionStateByRuntimeIdRef.current.get(runtimeId)

        if (!cached || (!cached.busy && !cached.awaitingResponse)) {
          return false
        }

        updateSessionState(runtimeId, state => ({ ...state, awaitingResponse: false, busy: false }))

        return true
      },
      interruptSession: async runtimeId => {
        // Read-only stored-transcript tiles have no live turn to interrupt.
        if (isReadOnlyRuntimeId(runtimeId)) {
          return
        }

        // Same cooldown as the primary chat's Stop (#83855): the gateway may
        // still be winding down after this interrupt, so a quick edit/resend
        // on the tile must go interrupt-first even though busy already reads
        // false. Mark the runtime id (and any recovered id) before the RPC so
        // the window covers the whole wind-down.
        markSessionRecentlyInterrupted(runtimeId)

        const storedSessionId = storedSessionIdForRuntime(runtimeId)

        const routedRequest = storedSessionId
          ? <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) =>
              requestForStoredSession<T>(storedSessionId, method, params ?? {}, timeoutMs)
          : requestGateway

        await withSessionNotFoundResume(
          runtimeId,
          storedSessionId,
          liveId => routedRequest('session.interrupt', { session_id: liveId }),
          {
            requestGateway: routedRequest,
            onRecovered: recoveredId => {
              markSessionRecentlyInterrupted(recoveredId)
              rebindTileRuntime(runtimeId)(recoveredId)
            }
          }
        )
      },
      resumeTile: async (storedSessionId, options) => {
        // A retained tile can still own its runtime after the primary view drops
        // its reverse lookup. Reconnect invalidates both bindings.
        const existing =
          runtimeIdByStoredSessionIdRef.current.get(storedSessionId) ??
          $sessionTiles.get().find(tile => tile.storedSessionId === storedSessionId)?.runtimeId

        const cached = existing ? sessionStateByRuntimeIdRef.current.get(existing) : undefined
        const refreshTranscript = options?.refreshTranscript === true

        // Warm path: reuse a live binding — but only when it still carries a
        // transcript (or is mid-turn, where messages legitimately stream in).
        // A binding whose cached state has no messages is either a released
        // transcript or a stale pre-reconnect survivor; reusing it painted the
        // post-sleep/wake tile permanently empty. Fall through to a real
        // resume instead — it's idempotent for a genuinely live session.
        //
        // Explicit reopen (`refreshTranscript`) must still REST-merge: the
        // warm snapshot is whatever the tile last painted, and cron bot-chat
        // deliveries that landed while the panel's WS was down never arrive
        // as realtime events (#96183).
        if (
          existing &&
          cached?.storedSessionId === storedSessionId &&
          (cached.busy || cached.messages.length > 0) &&
          !refreshTranscript
        ) {
          publishSessionState(existing, cached)

          return existing
        }

        // Resolve the owning profile before binding a runtime. A tile can open a
        // session from any profile, not just the active one; resuming (or
        // reading messages) without a profile lets the gateway fall back to the
        // launch-profile DB and fork the conversation into the wrong profile —
        // the same cross-profile bleed the recovery resumes had (#67603).
        const owner = await ownerForStoredSession(storedSessionId)

        const restScope =
          owner && typeof owner === 'object'
            ? { connectionId: owner.connectionId, profile: owner.targetProfile || owner.profile }
            : owner

        const prefetchPromise = getLatestSessionMessages(storedSessionId, restScope).catch(() => null)

        if (existing && cached?.storedSessionId === storedSessionId && (cached.busy || cached.messages.length > 0)) {
          const prefetch = await prefetchPromise
          // Deltas and completion may land while REST is in flight.
          updateSessionState(
            existing,
            state => {
              const merged = mergeTileTranscript(state.messages, prefetch?.messages, state.streamId ?? cached.streamId)

              return chatMessageArraysEquivalent(state.messages, merged) ? state : { ...state, messages: merged }
            },
            storedSessionId
          )

          return existing
        }

        // #94724 no-owner recovery: dispatching the resume through the same
        // fail-closed gate as the window's RPC dispatcher keeps an unknown
        // owner off the ambient socket, and the wrapper opens the stored
        // transcript read-only instead of dead-ending the tile — the id-only
        // REST read routes no live session at all.
        const outcome = await resumeWithStoredTranscriptFallback(
          storedSessionId,
          () => {
            assertSessionOwnerResolved(owner, { method: 'session.resume', sessionId: storedSessionId })

            return singleFlightSessionResume(storedSessionId, () =>
              requestForSessionProfile<SessionResumeResponse>(owner, requestGateway, 'session.resume', {
                session_id: storedSessionId,
                cols: 96,
                omit_messages: true,
                ...(owner ? { profile: typeof owner === 'string' ? owner : owner.profile } : {})
              })
            )
          },
          async () => {
            const stored = (await prefetchPromise) ?? (await fetchStoredTranscriptAcrossBackends(storedSessionId))

            if (!stored) {
              throw new Error('stored transcript unavailable on every reachable backend')
            }

            return stored
          }
        )

        const prefetch = await prefetchPromise

        if (outcome.mode === 'read-only') {
          const readOnlyId = readOnlyRuntimeIdFor(storedSessionId)

          updateSessionState(
            readOnlyId,
            state => ({
              ...state,
              busy: false,
              awaitingResponse: false,
              messages: state.messages.length > 0 ? state.messages : toChatMessages(outcome.transcript?.messages ?? [])
            }),
            storedSessionId
          )

          notify({
            kind: 'info',
            title: translateNow('desktop.readOnlyTranscriptTitle'),
            message: translateNow('desktop.readOnlyTranscriptBody')
          })

          return readOnlyId
        }

        const resumed = outcome.resumed

        const runtimeId = resumed?.session_id

        if (!runtimeId) {
          throw new Error('resume returned no session id')
        }

        const info = resumed?.info

        updateSessionState(
          runtimeId,
          state => ({
            ...state,
            busy: Boolean(info?.running),
            // Persist the session's own model/provider from resume so the tile
            // pill does not wait on a chrome-scoped catalog read (#93892).
            ...(typeof info?.model === 'string' ? { model: info.model } : {}),
            ...(typeof info?.provider === 'string' ? { provider: info.provider } : {}),
            ...(typeof info?.reasoning_effort === 'string' ? { reasoningEffort: info.reasoning_effort } : {}),
            ...(typeof info?.fast === 'boolean' ? { fast: info.fast } : {}),
            messages:
              state.messages.length > 0 ? state.messages : toChatMessages(prefetch?.messages ?? resumed?.messages ?? [])
          }),
          storedSessionId
        )

        return runtimeId
      },
      submitToSession: async (runtimeId, text) => {
        // A read-only stored-transcript tile has no live runtime to submit
        // into (#94724). Refuse with the explanation instead of minting a
        // misrouted prompt on a backend that never owned the session.
        if (isReadOnlyRuntimeId(runtimeId)) {
          notify({ kind: 'info', message: translateNow('desktop.readOnlyTranscriptSendBlocked') })

          return
        }

        const storedSessionId = storedSessionIdForRuntime(runtimeId)

        const routedRequest = storedSessionId
          ? <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) =>
              requestForStoredSession<T>(storedSessionId, method, params ?? {}, timeoutMs)
          : requestGateway

        await withSessionNotFoundResume(
          runtimeId,
          storedSessionId,
          liveId => routedRequest('prompt.submit', { session_id: liveId, text }, PROMPT_SUBMIT_REQUEST_TIMEOUT_MS),
          { requestGateway: routedRequest, onRecovered: rebindTileRuntime(runtimeId) }
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
