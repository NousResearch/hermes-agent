import type { MutableRefObject } from 'react'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { reconcileResumeMessages } from '@/app/session/hooks/use-session-actions/utils'
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
  let start = current.findIndex(message => message.role === 'user' && !baselineIds.has(message.id))

  // A queued submit may already be seeded when the old turn requests hydration.
  // Protect that baseline's LIVE user too, but never resurrect a settled warm
  // transcript merely because its rows still have optimistic ids.
  const baselineUser =
    baseline.busy || baseline.awaitingResponse || baseline.streamId
      ? baseline.messages.findLast(
          message => message.role === 'user' && message.rowId === undefined && message.id.startsWith('user-')
        )
      : undefined

  const optimistic = baselineUser ? current.findIndex(message => sameRow(message, baselineUser)) : -1

  if (optimistic >= 0) {
    start = start < 0 ? optimistic : Math.min(start, optimistic)
  }

  if (start < 0) {
    return reconcileResumeMessages(next, current)
  }

  // The nearest durable row anchors role ordinals even with backfill or an
  // adopted reply whose user wasn't in this renderer's cache yet.
  let localAnchor = start - 1
  let storedAnchor = -1

  while (localAnchor >= 0) {
    storedAnchor = next.findIndex(message => sameRow(message, current[localAnchor]))

    if (storedAnchor >= 0) {
      break
    }

    localAnchor -= 1
  }

  let boundary = storedAnchor
  const preceding = current[start - 1]

  if (preceding && localAnchor < start - 1) {
    const ordinal = current.slice(localAnchor + 1, start).filter(message => message.role === preceding.role).length

    const candidates = next
      .map((message, index) => ({ message, index }))
      .filter(({ message, index }) => index > storedAnchor && message.role === preceding.role)

    const candidate = candidates[ordinal - 1]
    // An unmatched prefix is a history rewrite, not evidence that the new
    // local user is represented. Retain that user's tail after server history.
    boundary =
      candidate && chatMessageText(candidate.message) === chatMessageText(preceding) ? candidate.index : next.length - 1
  }

  const localTail = current.slice(start)
  const prefix = reconcileResumeMessages(next.slice(0, boundary + 1), current.slice(0, start))

  // A live turn owns its tail, including deltas still queued for the next RAF.
  // Installing a stored reply before that first flush makes mutateStream seed
  // a second reply. Hydrate the settled prefix now; adopt this tail when idle.
  if (state.busy || state.awaitingResponse || state.streamId) {
    return [...prefix, ...localTail]
  }

  const storedTail = next.slice(boundary + 1)
  const reconciledTail = reconcileResumeMessages(storedTail, localTail)
  const counts = new Map<ChatMessage['role'], number>()
  const positions = new Map<ChatMessage['role'], number[]>()
  storedTail.forEach((message, index) => positions.set(message.role, [...(positions.get(message.role) ?? []), index]))

  for (const message of localTail) {
    const ordinal = counts.get(message.role) ?? 0
    counts.set(message.role, ordinal + 1)
    const index = positions.get(message.role)?.[ordinal]

    if (index === undefined) {
      reconciledTail.push(message)
    }
  }

  return [...prefix, ...reconciledTail]
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
    profile: (typeof owner === 'string' ? owner : owner?.profile) || getApiRequestProfile() || 'default'
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
          const latest = await getSessionMessages(storedSessionId, scope, {
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
          recordTranscriptTail(storedSessionId, latest, scope)

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
