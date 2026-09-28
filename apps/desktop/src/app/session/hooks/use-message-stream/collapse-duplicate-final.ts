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

  if (prior?.role !== 'assistant' || !prior.interim || chatMessageText(prior).trim() !== options.finalText) {
    return null
  }

  const next = messages.slice()
  next[priorIndex] = options.completeMessage(
    withUniqueToolCallIdsWithinMessage({
      ...prior,
      parts: [...prior.parts, ...live.parts.filter(part => part.type !== 'text')]
    })
  )
  next.splice(streamIndex, 1)

  return { keptId: prior.id, messages: next }
}

/**
 * Index of a sealed, text-only interim earlier in this occurrence that already
 * carries exactly the turn's final reply — `-1` when there is none.
 *
 * `message.interim` seals the segment the agent produced (a tool-call round's
 * commentary, or a verify-on-stop candidate). When the client has no live
 * bubble to seal at that moment — a superseded attempt's frames cleared the
 * stream, a reconnect dropped the deltas — the interim materializes its OWN
 * bubble from the text. The turn then streams that same text again and
 * completes, and the reply is on screen twice: the sealed interim (no footer)
 * above the settled bubble (with footer), while the store holds one row
 * (#123801).
 *
 * Byte-identical text is the discriminator: a second real segment never repeats
 * the reply verbatim, so an interim whose text IS the final text is this turn's
 * reply, not another segment. Tool-call parts disqualify it — that interim owns
 * rows the transcript must keep. Only the nearest sealed interim is considered
 * (the loop stops at the first one): anything further back is an earlier
 * segment of the turn.
 */
export function identicalInterimSiblingIndex(
  messages: ChatMessage[],
  boundaryIndex: number,
  finalText: string,
  excludeIndex = -1
): number {
  if (!finalText) {
    return -1
  }

  for (let index = messages.length - 1; index > boundaryIndex; index -= 1) {
    if (index === excludeIndex) {
      continue
    }

    const message = messages[index]

    if (message.role !== 'assistant' || message.hidden || message.interim !== true) {
      continue
    }

    if (message.parts.some(part => part.type === 'tool-call')) {
      return -1
    }

    return chatMessageText(message).trim() === finalText ? index : -1
  }

  return -1
}

/** The identical interim survives (it IS this reply); the live twin is dropped. */
export function collapseDuplicateFinalOntoIdenticalInterim(
  messages: ChatMessage[],
  streamIndex: number,
  interimIndex: number,
  options: {
    completeMessage: (message: ChatMessage) => ChatMessage
    finalText: string
    hasFailure: boolean
  }
): DuplicateFinalCollapse | null {
  if (streamIndex < 0 || interimIndex < 0 || options.hasFailure || !options.finalText) {
    return null
  }

  const live = messages[streamIndex]
  const interim = messages[interimIndex]

  if (!live || !interim || chatMessageText(live).trim() !== options.finalText) {
    return null
  }

  const next = messages.slice()
  next[interimIndex] = options.completeMessage(
    withUniqueToolCallIdsWithinMessage({
      ...interim,
      parts: [...interim.parts, ...live.parts.filter(part => part.type !== 'text')]
    })
  )
  next.splice(streamIndex, 1)

  return { keptId: interim.id, messages: next }
}
