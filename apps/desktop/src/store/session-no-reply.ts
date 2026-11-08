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

export function withNoReplyNotice(messages: ChatMessage[], runtimeId: string): ChatMessage[] {
  const last = messages.findLast(message => !message.hidden)
  // The runtime id is the one fact that lets a copied error-details blob be
  // traced to this turn in agent.log.
  const errorSurface: ErrorSurface = { ...NO_REPLY_SURFACE, session: runtimeId }

  // A turn that ran tools but never wrote text carries the notice on its own bubble.
  if (last?.role === 'assistant') {
    return messages.map(message => (message === last ? { ...message, error: NO_REPLY_ERROR, errorSurface } : message))
  }

  const occurredAt = Date.now() / 1000

  return [
    ...messages,
    {
      completedAt: occurredAt,
      error: NO_REPLY_ERROR,
      errorSurface,
      id: `assistant-no-reply-${Date.now()}`,
      parts: [],
      pending: false,
      role: 'assistant',
      timestamp: occurredAt
    }
  ]
}
