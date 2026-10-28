/**
 * How a resume snapshot is reconciled with the transcript already on screen.
 *
 * A reconnect re-resumes the open chat. The backend cuts that snapshot when it
 * answers, but the new socket is already attached, so the turn's
 * `message.complete` can reach the renderer first. Everything the resume
 * computes from (its in-flight projection, a REST page read before the reply
 * was stored) then predates the settled reply, and reconciling it over the
 * live transcript rewound the reply to the partial text the snapshot caught,
 * often just its first chunk. The live transcript is the newer truth for that
 * session, so the resume keeps it; durable row identity arrives with the next
 * transcript refresh.
 */

import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage, SessionResumeResult } from '@/types/hermes'

import { reconcilePersistedLiveTurn } from './persisted-live-turn'
import { turnSettledSince } from './settled-turns'
import { appendLiveSessionProjection, reconcileDurableHistory } from './utils'

/**
 * Whether a re-resume of the chat on screen must keep the live transcript: a
 * turn of the resumed runtime settled after the resume started (`startedAtS`,
 * seconds). A cold open has no live transcript of its own to keep.
 */
export function liveTurnSettledDuringResume(
  resumed: Pick<SessionResumeResult, 'session_id'>,
  startedAtS: number,
  sameSession: boolean
): boolean {
  return sameSession && turnSettledSince(resumed.session_id, startedAtS * 1000)
}

export function reconcileAuthoritativeChatMessages(
  authoritativeMessages: ChatMessage[],
  previousMessages: ChatMessage[],
  liveProjection?: Pick<SessionResumeResult, 'inflight' | 'queued' | 'session_id'>,
  sourceRows?: SessionMessage[]
): ChatMessage[] {
  if (liveProjection && sourceRows) {
    const reconciled = reconcilePersistedLiveTurn(authoritativeMessages, previousMessages, sourceRows, liveProjection)

    if (reconciled) {
      return reconciled
    }
  }

  return reconcileDurableHistory(
    liveProjection ? appendLiveSessionProjection(authoritativeMessages, liveProjection) : authoritativeMessages,
    previousMessages
  )
}

export function reconcileAuthoritativeMessages(
  authoritativeMessages: SessionResumeResult['messages'],
  previousMessages: ChatMessage[],
  liveProjection?: Pick<SessionResumeResult, 'inflight' | 'queued' | 'session_id'>
): ChatMessage[] {
  return reconcileAuthoritativeChatMessages(
    toChatMessages(authoritativeMessages),
    previousMessages,
    liveProjection,
    authoritativeMessages
  )
}

/** Session-state patch for a restored blocking prompt row; the first non-null projection is used. */
export function livePromptStreamId(
  ...projections: ({ streamId: string } | null)[]
): { awaitingResponse: false; sawAssistantPayload: true; streamId: string } | Record<string, never> {
  const live = projections.find(Boolean)

  return live ? { awaitingResponse: false, sawAssistantPayload: true, streamId: live.streamId } : {}
}

/**
 * The row a resumed running turn's next deltas extend: the first candidate id
 * the resumed transcript holds, else the backend projection's live row
 * (`assistant-stream-<runtime id>`). A cold resume (window reload, app restart)
 * has no stream target of its own, and keeping that empty one made the next
 * delta open a second bubble beside the row the resume had just restored, so
 * the reply split in two and its final text drew twice until the next refresh.
 */
export function resumedStreamTarget(
  messages: ChatMessage[],
  runtimeId: string,
  candidates: (null | string | undefined)[]
): null | string {
  const ids = new Set(messages.map(message => message.id))
  const held = [...candidates, `assistant-stream-${runtimeId}`].find(id => id && ids.has(id))

  return held ?? candidates.find(Boolean) ?? null
}
