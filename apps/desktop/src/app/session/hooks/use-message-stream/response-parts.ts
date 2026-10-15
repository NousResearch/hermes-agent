import { type ChatMessage, type ChatMessagePart, mergeFinalAssistantText } from '@/lib/chat-messages'

/** One live bubble may hold several model responses when an equal interim
 * callback is suppressed. The last tool round, not the bubble, bounds the
 * response that an interim/final frame is allowed to replace. */
export function currentResponseParts(parts: ChatMessagePart[]): ChatMessagePart[] {
  return parts.slice(parts.findLastIndex(part => part.type === 'tool-call') + 1)
}

export function mergeCurrentResponseText(parts: ChatMessagePart[], text: string, timestamp: number): ChatMessagePart[] {
  const boundary = parts.findLastIndex(part => part.type === 'tool-call') + 1

  return [...parts.slice(0, boundary), ...mergeFinalAssistantText(parts.slice(boundary), text, timestamp)]
}

/** Hydration can replace a streaming id before its terminal frame arrives.
 * The receipt still names the same stored response, even if a background reply
 * follows it. A folded bubble's final text source takes precedence over its
 * first row address so an earlier response cannot replace its final suffix. */
export function persistedFinalBubbleIndex(
  messages: readonly ChatMessage[],
  lastUserIndex: number,
  finalRowId: number | null | undefined
): number {
  if (lastUserIndex < 0 || typeof finalRowId !== 'number' || !Number.isSafeInteger(finalRowId) || finalRowId <= 0) {
    return -1
  }

  return messages.findLastIndex((message, index) => {
    if (index <= lastUserIndex || message.role !== 'assistant' || message.hidden) {
      return false
    }

    const finalTextPart = message.parts.findLast(part => part.type === 'text')

    return (finalTextPart?.sourceRowId ?? message.rowId) === finalRowId
  })
}
