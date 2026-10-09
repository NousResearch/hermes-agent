import type { ChatMessage } from '@/lib/chat-messages'
import type { SessionMessage, SessionResumeResult } from '@/types/hermes'

import { finiteTurnStartedAt } from './live-session-projection'
import { isInflightPromptRow, reconcilePersistedLiveTurn } from './persisted-live-turn'

type LiveTurnProjection = Pick<SessionResumeResult, 'inflight' | 'queued' | 'session_id' | 'turn_started_at'>

/**
 * Index of the live turn's first source row, or -1 when none is durable yet.
 *
 * An idle submit records turn_started_at before its prompt row. The gateway's
 * queue drain (`_drain_queued_prompt`) does the reverse: it re-places the
 * drained prompt's row, then starts the turn, so that prompt sits just before
 * the boundary. Cut off, the occurrence resolver has no anchor and the legacy
 * projection paints the live reply and tool cards a second time.
 */
function liveTurnStart(rows: SessionMessage[], startedAt: number, projection: LiveTurnProjection): number {
  const first = rows.findIndex(row => row.timestamp !== undefined && row.timestamp >= startedAt)
  const boundary = first < 0 ? rows.length : first
  const before = rows[boundary - 1]
  const inflight = projection.inflight

  if (!before || !inflight) {
    return first
  }

  const drainedPrompt =
    isInflightPromptRow(before, inflight) && (before.user_originated === false) === (inflight.user_originated === false)

  return drainedPrompt ? boundary - 1 : first
}

export function reconcilePersistedSessionTurn(
  messages: ChatMessage[],
  previous: ChatMessage[],
  rows: SessionMessage[],
  projection: LiveTurnProjection
): ChatMessage[] | null {
  const startedAt = finiteTurnStartedAt(projection)
  const hasBoundary = startedAt !== null
  const currentStart = hasBoundary ? liveTurnStart(rows, startedAt, projection) : 0

  if (currentStart < 0) {
    return null
  }

  const currentRows = rows.slice(currentStart)

  // The occurrence resolver predates canonical user provenance. Do not let a
  // runtime notice occupy a human prompt slot, even when its prose is equal.
  if (
    projection.inflight?.user_originated !== false &&
    currentRows.some(row => row.role === 'user' && row.user_originated === false)
  ) {
    return null
  }

  const reconciled = reconcilePersistedLiveTurn(messages, previous, currentRows, projection)

  if (!reconciled || projection.inflight?.user_originated !== false || !hasBoundary) {
    return reconciled
  }

  // A visible runtime wake can use source-row occurrence reconciliation too,
  // but its projected reply must retain the journal's backend-owned boundary.
  // Hidden/system wakes have no user anchor and use the legacy projection path.
  const boundary = reconciled.findIndex(
    message =>
      message.role === 'user' &&
      message.userOriginated === false &&
      message.timestamp !== undefined &&
      message.timestamp >= startedAt
  )

  if (boundary < 0) {
    return null
  }

  return reconciled.map((message, index) =>
    index >= boundary && message.id !== `user-queued-${projection.session_id}`
      ? { ...message, runtimeTurnStartedAt: startedAt }
      : message
  )
}
