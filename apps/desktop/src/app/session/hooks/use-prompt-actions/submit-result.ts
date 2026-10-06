import type { PromptSubmitResult } from '@hermes/shared'

import { toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import type { ClientSessionState } from '../../../types'

export interface PromptSubmitCompletionFence extends Pick<ClientSessionState, 'streamId' | 'turnStartedAt' | 'turnLive'> {
  state: ClientSessionState
}

export function capturePromptSubmitCompletionFence(state: ClientSessionState): PromptSubmitCompletionFence {
  return { state, streamId: state.streamId, turnStartedAt: state.turnStartedAt, turnLive: state.turnLive }
}

export function createPromptSubmitReconciler(optimisticId: string) {
  let fence: null | PromptSubmitCompletionFence = null
  let settled = false

  return {
    capture(state: ClientSessionState): ClientSessionState {
      fence = capturePromptSubmitCompletionFence(state)

      return state
    },
    reconcile(state: ClientSessionState, result: null | PromptSubmitResult | undefined): ClientSessionState {
      const reconciled = reconcilePromptSubmitResult(state, result, optimisticId, fence)

      settled = reconciled.settled

      return reconciled.state
    },
    get settled() { return settled }
  }
}

export function isCompletedPromptSubmit(result: null | PromptSubmitResult | undefined): boolean {
  return result?.duplicate === true && result.status === 'complete' && Array.isArray(result.messages)
}

function hasNewerPromptTurn(
  state: ClientSessionState,
  fence: null | PromptSubmitCompletionFence,
  snapshot: ClientSessionState['messages']
): boolean {
  if (
    !fence ||
    (state.streamId !== null && state.streamId !== fence.streamId) ||
    (state.turnStartedAt !== null && state.turnStartedAt !== fence.turnStartedAt) ||
    (state.turnLive && (!fence.turnLive || state !== fence.state))
  ) {
    return true
  }

  const latestRowId = snapshot.reduce((latest, message) => Math.max(latest, message.rowId ?? 0), 0)

  return state.messages.some(message => (message.rowId ?? 0) > latestRowId)
}

export function reconcilePromptSubmitResult(
  state: ClientSessionState,
  result: null | PromptSubmitResult | undefined,
  optimisticId: string,
  completionFence: null | PromptSubmitCompletionFence
): { settled: boolean; state: ClientSessionState } {
  if (isCompletedPromptSubmit(result)) {
    // The retry observed a finished turn after resume saw it streaming. Its
    // atomic history snapshot is the completion frame this window missed.
    const messages = toChatMessages(result!.messages as unknown as SessionMessage[])

    // A later queue drain or another surface can start a turn while this RPC
    // reply is in flight. The older snapshot must not erase its rows or timer.
    if (hasNewerPromptTurn(state, completionFence, messages)) {
      return { settled: false, state }
    }

    return {
      settled: true,
      state: {
        ...state,
        messages,
        busy: false,
        awaitingResponse: false,
        streamId: null,
        pendingBranchGroup: null,
        sawAssistantPayload: messages.some(message => message.role === 'assistant'),
        adoptedRunningTurn: false,
        needsInput: false,
        interimBoundaryPending: false,
        turnStartedAt: null,
        turnLive: false
      }
    }
  }

  const rowId = result?.user_row_id

  if (typeof rowId !== 'number' || !Number.isSafeInteger(rowId) || rowId <= 0) {
    return { settled: false, state }
  }

  // A normal acknowledgement can arrive after a live completion event. Bind
  // only this send's optimistic row and preserve the live turn's state.
  const index = state.messages.findIndex(message => message.id === optimisticId && message.role === 'user')

  if (index < 0 || state.messages[index].rowId === rowId) {
    return { settled: false, state }
  }

  return {
    settled: false,
    state: {
      ...state,
      messages: state.messages.map((message, i) => (i === index ? { ...message, rowId } : message))
    }
  }
}
