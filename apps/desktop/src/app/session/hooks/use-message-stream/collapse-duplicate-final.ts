import { transcriptRowIds } from '@/app/session/hooks/use-session-actions/pending-turn-identity'
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
 * The receipt's final row is already on screen as a stored bubble of this
 * occurrence, so the live stream is its twin (#123801).
 *
 * A history reconcile that lands after the gateway committed the reply but
 * before the client applied the rest of the turn folds the live bubble into
 * the stored row. The stream id survives it, so the next delta re-seeds a
 * bubble under that id and completion settles it: one stored row, a
 * `timestamp-index-assistant` root and an `assistant-stream-*` root. Settle
 * onto the stored row, which the receipt names, and drop the live twin.
 * Identity is the row id, never prose, so an identical reply of another
 * occurrence is untouched.
 */
export function collapseLiveTwinOfPersistedFinal(
  messages: ChatMessage[],
  streamIndex: number,
  options: {
    completeMessage: (message: ChatMessage) => ChatMessage
    finalRowId: null | number
    lastUserIndex: number
  }
): DuplicateFinalCollapse | null {
  const { finalRowId } = options

  if (streamIndex < 0 || finalRowId === null) {
    return null
  }

  const storedIndex = messages.findIndex(
    (message, index) =>
      index > options.lastUserIndex &&
      index !== streamIndex &&
      message.role === 'assistant' &&
      transcriptRowIds(message).includes(finalRowId)
  )

  if (storedIndex < 0) {
    return null
  }

  const next = messages.slice()
  next[storedIndex] = options.completeMessage(messages[storedIndex])
  next.splice(streamIndex, 1)

  return { keptId: messages[storedIndex].id, messages: next }
}
