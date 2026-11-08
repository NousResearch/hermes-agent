import type { ClientSessionState } from '@/app/types'
import { type ChatMessage, chatMessageText, finalizeInterruptedMessages, sealOpenToolParts } from '@/lib/chat-messages'
import type { ErrorSurface } from '@/lib/error-surface'

// Raised only after the backend confirmed the turn is over and no reply reached
// this window, so Retry cannot run the prompt twice.
const NO_REPLY_SURFACE: ErrorSurface = { code: 'no_reply', layer: 'runtime', retryable: true }
const NO_REPLY_ERROR = 'Hermes ended this turn without a reply.'

/** The inferred card this module stamps, not a failure the backend reported:
 *  both fields must match, since a later bare error replaces only the text. */
function isNoReplyNotice(message: ChatMessage): boolean {
  return message.error === NO_REPLY_ERROR && message.errorSurface?.code === NO_REPLY_SURFACE.code
}

/** The bubble without its inferred card: what a real outcome of the turn
 *  (answer text, a reported failure) starts from. */
export function withoutNoReplyNotice(message: ChatMessage, outcome = true): ChatMessage {
  return outcome && isNoReplyNotice(message) ? { ...message, error: undefined, errorSurface: undefined } : message
}

/** A terminal frame with an answer or a reported failure ends the turn's card. */
export function turnOutcome(finalText: string, completionError: null | string | undefined): boolean {
  return Boolean(finalText || completionError)
}

/** What a completion leaves as the settled id. Only one case keeps it: an
 *  empty frame of the settled turn itself (no answer, no failure, no newer
 *  turn live or armed) while that turn still has no reply, because the
 *  watchdog's pending history recovery owes it a no-reply card. A stream the
 *  turn's own late progress opened is not a newer turn. A turn whose kept
 *  bubble already holds its reply needs no card, and the id would let a later
 *  turn's output land on that bubble. */
export function settledIdAfterCompletion(state: ClientSessionState, outcome: boolean): null | string {
  const owed = !outcome && !state.turnLive && !state.awaitingResponse && !turnHasReply(state.messages)

  return owed ? (state.heartbeatSettledStreamId ?? null) : null
}

/** A card bubble that now holds answer text is the turn's reply, not a card.
 *  Reasoning and tool rows are not answer text, so they keep the card. */
export function dropAnsweredNoReplyNotice(message: ChatMessage): ChatMessage {
  return isNoReplyNotice(message) && chatMessageText(message).trim() ? withoutNoReplyNotice(message) : message
}

/** Late output of the turn the watchdog or a running=false heartbeat settled
 *  goes to that turn's bubble (its card included). Only while nothing newer
 *  is live or armed: a running=true heartbeat or a submitted prompt keeps the
 *  id until its message.start, and that turn's output is its own. A reserved
 *  id that names no bubble yet falls through to the ordinary stream. */
export function settledTurnTarget(state: ClientSessionState): null | string {
  const settledId = state.heartbeatSettledStreamId

  if (!settledId || state.streamId || state.turnLive || state.awaitingResponse) {
    return null
  }

  return state.messages.some(message => message.id === settledId) ? settledId : null
}

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
