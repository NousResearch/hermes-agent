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
    (message, index) => index < streamIndex && message.role !== 'system' && (!message.hidden || message.role === 'user')
  )

  const prior = messages[priorIndex]

  if (prior?.role !== 'assistant' || !prior.interim || chatMessageText(prior).trim() !== options.finalText) {
    return null
  }

  return mergeLiveIntoKept(messages, streamIndex, priorIndex, options.completeMessage)
}

/** Fold the live bubble's non-text parts into the kept bubble, then drop the live one. */
function mergeLiveIntoKept(
  messages: ChatMessage[],
  streamIndex: number,
  keptIndex: number,
  completeMessage: (message: ChatMessage) => ChatMessage
): DuplicateFinalCollapse {
  const kept = messages[keptIndex]
  const next = messages.slice()
  next[keptIndex] = completeMessage(
    withUniqueToolCallIdsWithinMessage({
      ...kept,
      parts: [...kept.parts, ...messages[streamIndex].parts.filter(part => part.type !== 'text')]
    })
  )
  next.splice(streamIndex, 1)

  return { keptId: kept.id, messages: next }
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
 *
 * `interimBoundaryPending` bounds the scan to the CURRENT occurrence. The
 * prompt row is not the only boundary: `message.start` also starts one while
 * keeping the prior messages (a chained turn, a prompt-less turn), and it
 * resets that flag. Without it, `interim('X') → message.start → delta('X') →
 * complete('X')` reaches back past the boundary, settles the PREVIOUS turn's
 * interim and deletes this turn's live bubble. The flag is exactly "an interim
 * was sealed since the last `message.start`", so a false value means the only
 * interims in range belong to an earlier occurrence and must not be touched.
 */
export function identicalInterimSiblingIndex(
  messages: ChatMessage[],
  boundaryIndex: number,
  finalText: string,
  options: { interimBoundaryPending: boolean; excludeIndex?: number; hasFailure?: boolean }
): number {
  if (!finalText || !options.interimBoundaryPending || options.hasFailure) {
    return -1
  }

  const excludeIndex = options.excludeIndex ?? -1

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
  }
): DuplicateFinalCollapse | null {
  // Callers pass a live streamIndex; a non-negative interimIndex already implies
  // a non-empty finalText and no failure.
  if (interimIndex < 0 || chatMessageText(messages[streamIndex]).trim() !== options.finalText) {
    return null
  }

  return mergeLiveIntoKept(messages, streamIndex, interimIndex, options.completeMessage)
}
