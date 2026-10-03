import { type ChatMessagePart, mergeFinalAssistantText } from '@/lib/chat-messages'

/** One live bubble may hold several model responses when an equal interim
 * callback is suppressed. The last tool round, not the bubble, bounds the
 * response that an interim/final frame is allowed to replace. */
export function currentResponseParts(parts: ChatMessagePart[]): ChatMessagePart[] {
  return parts.slice(parts.findLastIndex(part => part.type === 'tool-call') + 1)
}

export function mergeCurrentResponseText(parts: ChatMessagePart[], text: string, timestamp: number): ChatMessagePart[] {
  // Finals can be cumulative across a tool boundary. Let mergeFinalAssistantText
  // see the retained prefix so it can strip that prefix before replacing the
  // latest response instead of appending the cumulative text a second time.
  return mergeFinalAssistantText(parts, text, timestamp)
}
