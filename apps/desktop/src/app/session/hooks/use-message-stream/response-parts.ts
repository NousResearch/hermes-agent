import { type ChatMessagePart, mergeFinalAssistantText, normalizeWs } from '@/lib/chat-messages'

const joinedText = (parts: ChatMessagePart[]) =>
  normalizeWs(
    parts
      .filter((part): part is Extract<ChatMessagePart, { type: 'text' }> => part.type === 'text')
      .map(part => part.text)
      .join('')
  )

/** One live bubble may hold several model responses when an equal interim
 * callback is suppressed. The last tool round, not the bubble, bounds the
 * response that an interim/final frame is allowed to replace. */
export function currentResponseParts(parts: ChatMessagePart[]): ChatMessagePart[] {
  return parts.slice(parts.findLastIndex(part => part.type === 'tool-call') + 1)
}

export function mergeCurrentResponseText(parts: ChatMessagePart[], text: string, timestamp: number): ChatMessagePart[] {
  const boundary = parts.findLastIndex(part => part.type === 'tool-call') + 1
  const prefix = parts.slice(0, boundary)
  const suffix = parts.slice(boundary)

  // A bubble that ends on its tool round has no post-tool text to merge into.
  // When the final restates the text already streamed before that tool call
  // (#130396), there is nothing new to add; appending would paint it twice.
  const prefixText = joinedText(prefix)

  if (prefixText && !joinedText(suffix) && prefixText === normalizeWs(text)) {
    return parts
  }

  return [...prefix, ...mergeFinalAssistantText(suffix, text, timestamp)]
}
