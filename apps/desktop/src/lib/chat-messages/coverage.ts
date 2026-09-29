import { normalizeWs as normalizedText } from './parts'
import type { ChatMessage, ChatMessagePart } from './types'

function sameOccurrencePart(stored: ChatMessagePart, local: ChatMessagePart): boolean {
  if (stored.type === 'tool-call' && local.type === 'tool-call') {
    return Boolean(stored.toolCallId) && stored.toolCallId === local.toolCallId
  }

  return stored.type === 'text' && local.type === 'text' && normalizedText(stored.text) === normalizedText(local.text)
}

/** Subtract an ordered, tool-anchored prefix within an already matched user
 * interval. Hydration can fold several live bubbles into one durable row;
 * bubble ordinals and equal text alone cannot establish that coverage. */
export function withoutCoveredAssistantPrefix(stored: ChatMessage[], local: ChatMessage[]): ChatMessage[] {
  // Narration is not an occurrence. A window that attached mid-turn holds the
  // answer text but never saw the `reasoning.delta` frames, so a durable row
  // that leads with reasoning would stall the walk on its first part and leave
  // the live bubble of an already-covered answer on screen as a duplicate.
  const parts = stored
    .flatMap(message => (message.role === 'assistant' ? message.parts : []))
    .filter(part => part.type !== 'reasoning')

  let cursor = 0
  let anchored = false
  let stopped = false
  const remaining: ChatMessage[] = []

  for (const message of local) {
    if (stopped || message.role !== 'assistant' || message.error) {
      stopped = true
      remaining.push(message)

      continue
    }

    let consumed = 0

    for (const part of message.parts) {
      if (!parts[cursor] || !sameOccurrencePart(parts[cursor], part)) {
        break
      }

      anchored ||= part.type === 'tool-call'
      cursor += 1
      consumed += 1
    }

    if (consumed < message.parts.length) {
      stopped = true
      remaining.push(consumed ? { ...message, parts: message.parts.slice(consumed) } : message)
    }
  }

  // A coincidentally equal paragraph, without the same tool occurrence after
  // it, is insufficient evidence to remove anything.
  return anchored ? remaining : local
}
