import { type ChatMessagePart, mergeFinalAssistantText, partsText } from '@/lib/chat-messages'

/** One live bubble may hold several model responses when an equal interim
 * callback is suppressed. The last tool round, not the bubble, bounds the
 * response that an interim/final frame is allowed to replace. */
export function currentResponseParts(parts: ChatMessagePart[]): ChatMessagePart[] {
  return parts.slice(parts.findLastIndex(part => part.type === 'tool-call') + 1)
}

export function mergeCurrentResponseText(parts: ChatMessagePart[], text: string, timestamp: number): ChatMessagePart[] {
  const boundary = parts.findLastIndex(part => part.type === 'tool-call') + 1
  const earlier = parts.slice(0, boundary)
  const earlierText = partsText(earlier)

  // Some terminal frames carry cumulative text that re-includes the prose
  // streamed before the last tool round. Bounding the merge at the last tool
  // call keeps that pre-tool prose as a separate part while the cumulative
  // final re-includes it — the same paragraph renders twice. Strip the
  // pre-boundary prefix first, mirroring mergeFinalAssistantText's tool path.
  const responseText = earlierText && text.startsWith(earlierText) ? text.slice(earlierText.length) : text

  return [...earlier, ...mergeFinalAssistantText(parts.slice(boundary), responseText, timestamp)]
}
