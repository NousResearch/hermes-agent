import type { ClientSessionState } from '@/app/types'
import { type ChatMessage, chatMessageText, finalizeInterruptedMessages, sealOpenToolParts } from '@/lib/chat-messages'
import type { ErrorSurface } from '@/lib/error-surface'

// Raised only after the backend confirmed the turn is over and no reply reached
// this window, so Retry cannot run the prompt twice.
const NO_REPLY_SURFACE: ErrorSurface = { code: 'no_reply', layer: 'runtime', retryable: true }
const NO_REPLY_ERROR = 'Hermes ended this turn without a reply.'

function turnHasReply(messages: ChatMessage[]): boolean {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]

    if (message.hidden) {
      continue
    }

    if (message.role === 'user') {
      return false
    }

    if (message.role === 'assistant' && (message.error || chatMessageText(message).trim())) {
      return true
    }
  }

  return false
}

function withNoReplyNotice(messages: ChatMessage[]): ChatMessage[] {
  const last = messages.findLast(message => !message.hidden)

  // A turn that ran tools but never wrote text carries the notice on its own bubble.
  if (last?.role === 'assistant') {
    return messages.map(message =>
      message === last ? { ...message, error: NO_REPLY_ERROR, errorSurface: NO_REPLY_SURFACE } : message
    )
  }

  const occurredAt = Date.now() / 1000

  return [
    ...messages,
    {
      completedAt: occurredAt,
      error: NO_REPLY_ERROR,
      errorSurface: NO_REPLY_SURFACE,
      id: `assistant-no-reply-${Date.now()}`,
      parts: [],
      pending: false,
      role: 'assistant',
      timestamp: occurredAt
    }
  ]
}

/** Settle a live turn the backend reported over whose end events never
 *  arrived, like the running=false edge: nothing is interrupted, the kept
 *  stream bubble is remembered so a late message.complete settles onto it. */
export function settleEndedTurn(state: ClientSessionState, occurredAt: number): ClientSessionState {
  const messages = sealOpenToolParts(finalizeInterruptedMessages(state.messages, state.streamId, occurredAt))

  return {
    ...state,
    awaitingResponse: false,
    busy: false,
    heartbeatSettledStreamId:
      state.streamId && messages.some(message => message.id === state.streamId) ? state.streamId : null,
    messages,
    pendingBranchGroup: null,
    streamId: null,
    turnLive: false,
    turnStartedAt: null
  }
}

/** Stamp the retry card on an ended turn that has no reply, never an
 *  intentional interruption. A newer turn has its own reply. */
export function stampNoReply(state: ClientSessionState): ClientSessionState {
  return state.interrupted || turnHasReply(state.messages)
    ? state
    : { ...state, messages: withNoReplyNotice(state.messages) }
}
