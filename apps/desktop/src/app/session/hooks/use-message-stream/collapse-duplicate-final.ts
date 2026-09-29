import { type ChatMessage, chatMessageText, withUniqueToolCallIdsWithinMessage } from '@/lib/chat-messages'

export interface DuplicateFinalCollapse {
  /** The surviving bubble replaces the removed stream id at completion. */
  keptId: string
  messages: ChatMessage[]
}

/** An identical final belongs to the adjacent interim, not a second tool bubble. */
export function collapseDuplicateFinalAfterToolInterim(
  messages: ChatMessage[],
  streamIndex: number,
  options: {
    completeMessage: (message: ChatMessage) => ChatMessage
    finalText: string
    hasFailure: boolean
    interimBoundaryPending: boolean
  }
): DuplicateFinalCollapse | null {
  if (streamIndex < 0 || !options.interimBoundaryPending || options.hasFailure || !options.finalText) {
    return null
  }

  const live = messages[streamIndex]

  if (!live?.parts.some(part => part.type === 'tool-call')) {
    return null
  }

  const liveText = chatMessageText(live).trim()

  if (liveText && liveText !== options.finalText) {
    return null
  }

  const priorIndex = messages.findLastIndex(
    (message, index) => index < streamIndex && (!message.hidden || message.role === 'user')
  )

  const prior = messages[priorIndex]

  const priorText = chatMessageText(prior).trim()

  // A late interim can contain only the final response's tail when the
  // renderer flushes around the tool-fold boundary. It is still the same
  // durable reply, so do not leave that tail as a separate bubble.
  if (
    prior?.role !== 'assistant' ||
    !prior.interim ||
    !priorText ||
    (priorText !== options.finalText && !options.finalText.endsWith(priorText))
  ) {
    return null
  }

  const next = messages.slice()
  const merged = withUniqueToolCallIdsWithinMessage({
    ...prior,
    parts: [...prior.parts, ...live.parts.filter(part => part.type !== 'text')]
  })
  const textIndex = merged.parts.findIndex(part => part.type === 'text')
  const finalParts =
    textIndex < 0
      ? merged.parts
      : merged.parts.map((part, index) => (index === textIndex ? { ...part, text: options.finalText } : part))

  next[priorIndex] = options.completeMessage({ ...merged, parts: finalParts })
  next.splice(streamIndex, 1)

  return { keptId: prior.id, messages: next }
}
