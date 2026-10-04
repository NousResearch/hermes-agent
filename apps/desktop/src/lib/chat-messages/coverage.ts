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
  // Narration is not an occurrence, on either side. A window that attached
  // mid-turn holds the answer text but never saw the `reasoning.delta` frames,
  // and a live row can lead with reasoning the durable row never received
  // (`appendReasoningDelta`'s replace branch attaches it before any text).
  // Either asymmetry would stall the walk before the shared tool anchor and
  // leave the live bubble of an already-covered answer on screen.
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

    const walkable = message.parts.filter(part => part.type !== 'reasoning')

    // A row that is nothing but narration has no occurrence to prove coverage
    // with — it is never folded.
    if (!walkable.length) {
      stopped = true
      remaining.push(message)

      continue
    }

    let consumed = 0
    let sliceFrom = 0

    for (let index = 0; index < message.parts.length; index += 1) {
      const part = message.parts[index]

      if (part.type === 'reasoning') {
        continue
      }

      if (!parts[cursor] || !sameOccurrencePart(parts[cursor], part)) {
        break
      }

      anchored ||= part.type === 'tool-call'
      cursor += 1
      consumed += 1
      sliceFrom = index + 1
    }

    if (consumed < walkable.length) {
      stopped = true
      remaining.push(consumed ? { ...message, parts: message.parts.slice(sliceFrom) } : message)
    }
  }

  // A coincidentally equal paragraph, without the same tool occurrence after
  // it, is insufficient evidence to remove anything.
  return anchored ? remaining : local
}
