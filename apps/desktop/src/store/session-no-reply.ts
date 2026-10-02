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

function withNoReplyNotice(messages: ChatMessage[], noticeId: string): { messages: ChatMessage[]; noticeId: string } {
  const last = messages.findLast(message => !message.hidden)

  // A turn that ran tools but never wrote text carries the notice on its own bubble.
  if (last?.role === 'assistant') {
    return {
      messages: messages.map(message =>
        message === last ? { ...message, error: NO_REPLY_ERROR, errorSurface: NO_REPLY_SURFACE } : message
      ),
      noticeId: last.id
    }
  }

  const occurredAt = Date.now() / 1000

  return {
    messages: [
      ...messages,
      {
        completedAt: occurredAt,
        error: NO_REPLY_ERROR,
        errorSurface: NO_REPLY_SURFACE,
        id: noticeId,
        parts: [],
        pending: false,
        role: 'assistant',
        timestamp: occurredAt
      }
    ],
    noticeId
  }
}

/** Settle a live turn the backend reported over whose end events never
 *  arrived, like the running=false edge: nothing is interrupted. Its
 *  `heartbeatSettledStreamId` names the kept stream bubble, or is reserved when
 *  settling dropped an empty one, so the notice binds to this turn. History
 *  recovery may await I/O; a newer message.start clears the id first. */
export function settleEndedTurn(state: ClientSessionState, occurredAt: number): ClientSessionState {
  const messages = sealOpenToolParts(finalizeInterruptedMessages(state.messages, state.streamId, occurredAt))

  return {
    ...state,
    awaitingResponse: false,
    busy: false,
    heartbeatSettledStreamId:
      state.streamId && messages.some(message => message.id === state.streamId)
        ? state.streamId
        : `assistant-no-reply-${Math.round(occurredAt * 1000)}`,
    messages,
    pendingBranchGroup: null,
    streamId: null,
    turnLive: false,
    turnStartedAt: null
  }
}

/** Stamp the retry card on the settled turn when it still has no reply. Never
 *  an intentional interruption, and never once a newer turn took the id. */
export function stampNoReply(state: ClientSessionState, settledId: string): ClientSessionState {
  if (state.heartbeatSettledStreamId !== settledId || state.interrupted || turnHasReply(state.messages)) {
    return state
  }

  const stamped = withNoReplyNotice(state.messages, settledId)

  return { ...state, heartbeatSettledStreamId: stamped.noticeId, messages: stamped.messages }
}
