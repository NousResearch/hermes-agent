import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import {
  type ChatMessage,
  preserveLocalAssistantErrors
} from '@/lib/chat-messages'
import type { SessionResumeResponse } from '@/types/hermes'

import {
  appendLiveSessionProjection,
  preserveLocalPendingTurnMessages,
  reconcileResumeMessages
} from './utils'

/**
 * Reconciles a runtime snapshot with ephemeral runtime projection. Persisted
 * display reads use the separate helper below because they preserve backfill.
 */
export function reconcileRuntimeAuthoritativeChatMessages(
  authoritativeMessages: ChatMessage[],
  previousMessages: ChatMessage[],
  liveProjection?: Pick<SessionResumeResponse, 'inflight' | 'queued' | 'session_id'>
): ChatMessage[] {
  const withLiveProjection = liveProjection
    ? appendLiveSessionProjection(authoritativeMessages, liveProjection)
    : authoritativeMessages

  const reconciled = reconcileResumeMessages(withLiveProjection, previousMessages)
  const withPendingTurn = preserveLocalPendingTurnMessages(reconciled, previousMessages)

  return preserveLocalAssistantErrors(withPendingTurn, previousMessages)
}

/**
 * Reconciles a persisted display page without changing its existing backfill
 * or local-pending preservation semantics. Kept separate from runtime
 * reconciliation so the conditional REST path has an observable boundary.
 */
export function reconcileAuthoritativeChatMessages(
  authoritativeMessages: ChatMessage[],
  currentMessages: ChatMessage[],
  backfillBase: ChatMessage[]
): ChatMessage[] {
  return preserveLocalPendingTurnMessages(
    graftRefreshedTailOntoBackfill(authoritativeMessages, backfillBase),
    currentMessages
  )
}
