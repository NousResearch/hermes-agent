import type { MutableRefObject } from 'react'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { isGatewaySystemMarker, reconcileResumeMessages } from '@/app/session/hooks/use-session-actions/utils'
import type { ClientSessionState } from '@/app/types'
import {
  getApiRequestConnection,
  getApiRequestProfile,
  getSessionMessages,
  LATEST_SESSION_MESSAGES_LIMIT
} from '@/hermes'
import { type ChatMessage, chatMessageText, preserveLocalAssistantErrors, toChatMessages } from '@/lib/chat-messages'
import { latestSessionTodos } from '@/lib/todos'
import { $sessions, sessionMatchesStoredId } from '@/store/session'
import { assertSessionOwnerResolved } from '@/store/session-owner-resolution'
import { knownOwnerForSession } from '@/store/session-states'
import { clearSessionTodos, setSessionTodos, todosForHydration } from '@/store/todos'
import { recordTranscriptTail } from '@/store/transcript-tail'

interface PostTurnHydrationOptions {
  activeSessionIdRef: MutableRefObject<string | null>
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
  runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}

const sameRow = (a: ChatMessage, b: ChatMessage) => a.id === b.id || (a.rowId !== undefined && a.rowId === b.rowId)

/** Keep unpersisted turns, not arbitrary obsolete history. Pair only within
 * the tail AFTER its preceding reply; global text matching eats repeated turns. */
function mergePostTurnMessages(stored: ChatMessage[], baseline: ClientSessionState, state: ClientSessionState) {
  const current = state.messages
  const next = graftRefreshedTailOntoBackfill(stored, current)
  const baselineIds = new Set(baseline.messages.map(message => message.id))

  let start = current.findIndex(
    message => message.role === 'user' && !isGatewaySystemMarker(message) && !baselineIds.has(message.id)
  )

  // A queued submit may already be seeded when the old turn requests hydration.
  // Protect that baseline's LIVE user too, but never resurrect a settled warm
  // transcript merely because its rows still have optimistic ids.
  const baselineUser =
    baseline.busy || baseline.awaitingResponse || baseline.streamId
      ? baseline.messages.findLast(
          message =>
            message.role === 'user' &&
            !isGatewaySystemMarker(message) &&
            message.rowId === undefined &&
            message.id.startsWith('user-')
        )
      : undefined

  const optimistic = baselineUser ? current.findIndex(message => sameRow(message, baselineUser)) : -1

  if (optimistic >= 0) {
    start = start < 0 ? optimistic : Math.min(start, optimistic)
  }

  if (start < 0) {
    return reconcileResumeMessages(next, current)
  }

  // REST folds tool-bearing assistant rows together. Only accepted USER
  // boundaries survive that projection; assistant ordinals are not turn ids.
  const acceptedUser = (message: ChatMessage) => message.role === 'user' && !isGatewaySystemMarker(message)

  const endOfTurn = (from: number) => {
    const nextUser = next.findIndex((message, index) => index > from && acceptedUser(message))

    return nextUser < 0 ? next.length : nextUser
  }

  const correspondingUser = (storedUser: ChatMessage, localUser: ChatMessage) =>
    acceptedUser(storedUser) &&
    storedUser.rowId !== undefined &&
    (localUser.rowId !== undefined
      ? storedUser.rowId === localUser.rowId
      : chatMessageText(storedUser) === chatMessageText(localUser))

  let localAnchor = start - 1
  let storedAnchor = -1

  while (localAnchor >= 0) {
    const rowId = current[localAnchor].rowId
    storedAnchor = rowId === undefined ? -1 : next.findIndex(message => message.rowId === rowId)

    if (storedAnchor >= 0) {
      break
    }

    localAnchor -= 1
  }

  if (storedAnchor < 0) {
    // An adopted running reply may precede this renderer's first user row.
    // Recover only the first DURABLE turn, with its reply inside that turn.
    // Idless/re-written history has no safe overlap: leave this concurrent
    // read alone. A later idle read (no protected tail) remains authoritative.
    const firstUser = next.findIndex(acceptedUser)
    const firstTurnEnd = firstUser < 0 ? 0 : endOfTurn(firstUser)

    const replyText = (messages: ChatMessage[]) =>
      messages
        .filter(message => message.role === 'assistant')
        .map(chatMessageText)
        .join('')

    const localReply = replyText(current.slice(0, start))

    if (
      current.slice(0, start).some(acceptedUser) ||
      firstUser < 0 ||
      next[firstUser].rowId === undefined ||
      !localReply ||
      replyText(next.slice(firstUser + 1, firstTurnEnd)) !== localReply
    ) {
      return current
    }

    storedAnchor = firstUser
    localAnchor = start - 1
  }

  let boundary = endOfTurn(storedAnchor)

  // Advance by actual accepted users, never by a count of assistant rows or
  // by searching past a mismatching user (identical prompts are legitimate).
  for (const user of current.slice(localAnchor + 1, start).filter(acceptedUser)) {
    if (boundary >= next.length || !correspondingUser(next[boundary], user)) {
      return current
    }

    boundary = endOfTurn(boundary)
  }

  const localTail = current.slice(start)
  const prefix = reconcileResumeMessages(next.slice(0, boundary), current.slice(0, start))

  // The live stream owns even its RAF-buffered tail. Do not seed stored B
  // before mutateStream's first flush, or the flush will create a second B.
  if (state.busy || state.awaitingResponse || state.streamId) {
    return [...prefix, ...localTail]
  }

  const result = [...prefix]
  let localStart = start

  while (localStart < current.length) {
    const nextUser = current.findIndex((message, index) => index > localStart && acceptedUser(message))
    const localEnd = nextUser < 0 ? current.length : nextUser

    if (boundary >= next.length || !correspondingUser(next[boundary], current[localStart])) {
      // An idless or mismatching user is not proof of persistence. Defer
      // this ambiguous stored tail until a later idle read; do not duplicate
      // it alongside the accepted local tail or silently consume that tail.
      return [...result, ...current.slice(localStart)]
    }

    const storedEnd = endOfTurn(boundary)
    const storedTurn = next.slice(boundary, storedEnd)
    const localTurn = current.slice(localStart, localEnd)
    result.push(...reconcileResumeMessages(storedTurn, localTurn))

    // A persisted prompt without its answer is not a persisted whole turn.
    if (!storedTurn.some(message => message.role === 'assistant')) {
      result.push(...localTurn.filter(message => message.role === 'assistant'))
    }

    boundary = storedEnd
    localStart = localEnd
  }

  return [...result, ...next.slice(boundary)]
}

function hydrationScope(runtimeSessionId: string) {
  const owner = knownOwnerForSession(runtimeSessionId)

  try {
    assertSessionOwnerResolved(owner, { method: 'post-turn history', sessionId: runtimeSessionId })
  } catch {
    return null
  }

  return {
    connectionId:
      typeof owner === 'object' && owner
        ? owner.connectionId
        : typeof owner === 'string'
          ? 'local'
          : getApiRequestConnection() || 'local',
    profile: (typeof owner === 'string' ? owner : owner?.profile) || getApiRequestProfile() || 'default',
    targetProfile: typeof owner === 'object' && owner ? owner.targetProfile : undefined,
    mode: typeof owner === 'object' && owner ? owner.mode : undefined
  }
}

/** Stored-history fallback used by the stream's terminal transitions. */
export function createPostTurnHydrator({
  activeSessionIdRef,
  selectedStoredSessionIdRef,
  sessionStateByRuntimeIdRef,
  runtimeIdByStoredSessionIdRef,
  updateSessionState
}: PostTurnHydrationOptions) {
  const requests = new Map<string, symbol>()

  return async (
    attempts = 1,
    storedSessionId = selectedStoredSessionIdRef.current,
    runtimeSessionId = activeSessionIdRef.current
  ) => {
    if (!storedSessionId || !runtimeSessionId) {
      return
    }

    const cache = sessionStateByRuntimeIdRef.current
    const baseline = cache.get(runtimeSessionId)

    if (!baseline || baseline.storedSessionId !== storedSessionId) {
      return
    }

    const scope = hydrationScope(runtimeSessionId)

    if (!scope) {
      return
    }

    const backendScope = { connectionId: scope.connectionId, profile: scope.targetProfile || scope.profile }

    const lineage = () =>
      $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))?._lineage_root_id

    const lineageRoot = lineage()
    const request = Symbol()
    requests.set(runtimeSessionId, request)

    const isCurrent = () => {
      const state = cache.get(runtimeSessionId)
      const owner = hydrationScope(runtimeSessionId)

      return (
        requests.get(runtimeSessionId) === request &&
        sessionStateByRuntimeIdRef.current === cache &&
        state?.storedSessionId === storedSessionId &&
        state.transcriptAuthorityEpoch === baseline.transcriptAuthorityEpoch &&
        runtimeIdByStoredSessionIdRef.current.get(storedSessionId) === runtimeSessionId &&
        owner?.connectionId === scope.connectionId &&
        owner?.profile === scope.profile &&
        owner?.targetProfile === scope.targetProfile &&
        owner?.mode === scope.mode &&
        lineage() === lineageRoot
      )
    }

    try {
      for (let index = 0; index < Math.max(1, attempts); index += 1) {
        if (!isCurrent()) {
          return
        }

        try {
          // Unlike getLatestSessionMessages, this read has no tail-bookkeeping
          // side effect before the ownership/freshness check below.
          const latest = await getSessionMessages(storedSessionId, backendScope, {
            limit: LATEST_SESSION_MESSAGES_LIMIT,
            order: 'latest',
            includeCompacted: true
          })

          if (!isCurrent() || latest.session_id !== storedSessionId) {
            return
          }

          const messages = toChatMessages(latest.messages)
          const current = cache.get(runtimeSessionId)!
          const restoreTodos = !current.busy && !current.awaitingResponse && current.messages === baseline.messages
          recordTranscriptTail(storedSessionId, latest, backendScope)

          const updated = updateSessionState(runtimeSessionId, state => ({
            ...state,
            messages: preserveLocalAssistantErrors(mergePostTurnMessages(messages, baseline, state), state.messages)
          }))

          if (restoreTodos && isCurrent() && cache.get(runtimeSessionId) === updated) {
            const restored = todosForHydration(latestSessionTodos(updated.messages))

            if (restored) {
              setSessionTodos(runtimeSessionId, restored)
            } else {
              clearSessionTodos(runtimeSessionId)
            }
          }

          return
        } catch {
          // Best-effort fallback when live stream payloads are empty.
        }

        if (index < attempts - 1) {
          await new Promise(resolve => window.setTimeout(resolve, 250))
        }
      }
    } finally {
      if (requests.get(runtimeSessionId) === request) {
        requests.delete(runtimeSessionId)
      }
    }
  }
}
