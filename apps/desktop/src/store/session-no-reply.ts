import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import type { ErrorSurface } from '@/lib/error-surface'

// Raised only after the backend confirmed the turn is over and no reply reached
// this window, so Retry cannot run the prompt twice.
const NO_REPLY_SURFACE: ErrorSurface = { code: 'no_reply', layer: 'runtime', retryable: true }
const NO_REPLY_ERROR = 'Hermes ended this turn without a reply.'

export function turnHasReply(messages: ChatMessage[]): boolean {
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

export function withNoReplyNotice(messages: ChatMessage[]): ChatMessage[] {
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

// Interrupted settles that never got a reasoned message.complete still must not go blank.
// Non-retryable: Stop / system cancel is intentional, not a connection failure (#131205 family).
const INTERRUPTED_EMPTY_SURFACE: ErrorSurface = { code: 'interrupted', layer: 'runtime', retryable: false }
const INTERRUPTED_EMPTY_ERROR = 'Operation interrupted: turn stopped with no reply.'

/** Interrupted turn with no user-visible text and no message.complete reason yet. */
export function withInterruptedEmptyNotice(messages: ChatMessage[]): ChatMessage[] {
  const last = messages.findLast(message => !message.hidden)

  if (last?.role === 'assistant') {
    return messages.map(message =>
      message === last
        ? {
            ...message,
            error: INTERRUPTED_EMPTY_ERROR,
            errorSurface: INTERRUPTED_EMPTY_SURFACE,
            interrupted: true
          }
        : message
    )
  }

  const occurredAt = Date.now() / 1000

  return [
    ...messages,
    {
      completedAt: occurredAt,
      error: INTERRUPTED_EMPTY_ERROR,
      errorSurface: INTERRUPTED_EMPTY_SURFACE,
      id: `assistant-interrupted-empty-${Date.now()}`,
      interrupted: true,
      parts: [],
      pending: false,
      role: 'assistant',
      timestamp: occurredAt
    }
  ]
}
