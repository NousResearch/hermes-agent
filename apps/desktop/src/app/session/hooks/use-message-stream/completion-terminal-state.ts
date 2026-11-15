import type { PersistedTurn } from '@hermes/shared'

import type { ErrorSurface } from '@/lib/error-surface'

import type { ClientSessionState } from '../../../types'

import { bindStoppedTurnReceipt, extendInterruptedReply } from './interrupted-reply'
import { isContentPolicyRefusal, replaceTurnWithContentRefused } from './utils'

interface TerminalCompletionArgs {
  text: string
  occurredAt: number
  persistedTurn?: PersistedTurn | null
  failure?: { error: string; partial: boolean; surface?: ErrorSurface | null }
  status?: string
  failureReason?: string
}

/**
 * A completion that short-circuits the normal settle path with a fully-formed
 * terminal state, or `null` when the normal path should run.
 *
 * Two cases collapse the turn without the dedupe/merge machinery below:
 *
 *  • Late completion from an already-cancelled turn: cancelRun has already
 *    finalized the bubble (kept the partial text, dropped it if empty).
 *    Re-running the dedupe would replace the partial with the just-cancelled
 *    full text, so we settle and bail instead — only extending the bubble to
 *    the partial the agent persisted (#121594).
 *
 *  • A provider safety-filter refusal replaces the whole turn with the same
 *    greyed content-refused row stored hydration produces — never a normal
 *    error card beside the untouched user bubble. The live frame is enough;
 *    a reload would only rebuild this same row, so the caller skips the extra
 *    hydrate.
 */
export function terminalCompletionState(
  state: ClientSessionState,
  { text, occurredAt, persistedTurn, failure, status, failureReason }: TerminalCompletionArgs
): ClientSessionState | null {
  if (state.interrupted) {
    return {
      ...state,
      messages: bindStoppedTurnReceipt(
        status === 'interrupted' ? extendInterruptedReply(state.messages, text, occurredAt) : state.messages,
        persistedTurn
      ),
      awaitingResponse: false,
      busy: false,
      needsInput: false,
      pendingBranchGroup: null,
      streamId: null,
      turnStartedAt: null,
      turnLive: false
    }
  }

  if (isContentPolicyRefusal(failure, failureReason)) {
    return {
      ...state,
      messages: replaceTurnWithContentRefused(state.messages, occurredAt),
      adoptedRunningTurn: false,
      heartbeatSettledStreamId: null,
      streamId: null,
      pendingBranchGroup: null,
      awaitingResponse: false,
      busy: false,
      needsInput: false,
      interimBoundaryPending: false,
      turnStartedAt: null,
      turnLive: false
    }
  }

  return null
}
