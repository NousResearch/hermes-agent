import { type ChatMessagePart, mergeFinalAssistantText } from '@/lib/chat-messages'

/** One live bubble may hold several model responses when an equal interim
 * callback is suppressed. The last tool round, not the bubble, bounds the
 * response that an interim/final frame is allowed to replace. */
export function currentResponseParts(parts: ChatMessagePart[]): ChatMessagePart[] {
  return parts.slice(parts.findLastIndex(part => part.type === 'tool-call') + 1)
}

export function mergeCurrentResponseText(
  parts: ChatMessagePart[],
  text: string,
  timestamp: number
): ChatMessagePart[] {
  const boundary = parts.findLastIndex(part => part.type === 'tool-call') + 1

  return [...parts.slice(0, boundary), ...mergeFinalAssistantText(parts.slice(boundary), text, timestamp)]
}
