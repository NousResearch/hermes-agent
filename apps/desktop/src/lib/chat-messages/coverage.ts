import { normalizeWs as normalizedText } from './parts'
import type { ChatMessage, ChatMessagePart } from './types'

function sameOccurrencePart(stored: ChatMessagePart, local: ChatMessagePart): boolean {
  if (stored.type === 'tool-call' && local.type === 'tool-call') {
    return Boolean(stored.toolCallId) && stored.toolCallId === local.toolCallId
  }

  if (stored.type === 'text' && local.type === 'text') {
    return normalizedText(stored.text) === normalizedText(local.text)
  }

  return false
}

/** Subtract an ordered, tool-anchored prefix within an already matched user
 * interval. Reasoning visibility and segmentation differ between live events
 * and durable display projections; only public text and tool IDs prove coverage.
 * Report exact covered tool pairs so callers can retain newer completion state. */
export function withoutCoveredAssistantPrefix(
  stored: ChatMessage[],
  local: ChatMessage[],
  coveredTools?: Map<ChatMessagePart, ChatMessagePart>
): ChatMessage[] {
  const parts = stored.flatMap(message =>
    message.role === 'assistant' ? message.parts.filter(part => part.type !== 'reasoning') : []
  )

  let cursor = 0
  let covered: { message: number; part: number } | undefined

  scan: for (const [messageIndex, message] of local.entries()) {
    if (message.role !== 'assistant' || message.error) {
      break
    }

    for (const [partIndex, part] of message.parts.entries()) {
      if (part.type === 'reasoning') {
        continue
      }

      if (!parts[cursor] || !sameOccurrencePart(parts[cursor], part)) {
        break scan
      }

      cursor += 1

      if (part.type === 'tool-call') {
        covered = { message: messageIndex, part: partIndex }
        coveredTools?.set(parts[cursor - 1], part)
      }
    }
  }

  // Commit only through the last shared tool. Equal prose or reasoning after
  // it can be a new occurrence and must survive until another tool anchors it.
  if (!covered) {
    return local
  }

  const message = local[covered.message]
  const suffix = message.parts.slice(covered.part + 1)

  return [...(suffix.length ? [{ ...message, parts: suffix }] : []), ...local.slice(covered.message + 1)]
}
