import {
  type ChatMessage,
  type ChatMessagePart,
  chatMessageText,
  normalizeWs as normalizedText
} from '@/lib/chat-messages'
import { isLiveTailReplyId } from '@/lib/spoken-reply'

/** Row predicates and matching shared by the journal's tail capture and its
 *  human-turn and runtime-turn merges. */

export interface InFlightRecoveryResult {
  applied: boolean
  /** The base transcript already contains the journaled turn's completed
   *  reply — the journal entry is stale and has been cleared. */
  caughtUp: boolean
  messages: ChatMessage[]
  streamId: null | string
  turnStartedAt: null | number
}

function attachmentSignature(message: ChatMessage): string {
  return (message.attachmentRefs ?? []).join('\n')
}

function userMessageIdentityMatches(left: ChatMessage, right: ChatMessage): boolean {
  return (
    left.role === 'user' &&
    right.role === 'user' &&
    (left.userOriginated === undefined || right.userOriginated === undefined || left.userOriginated === right.userOriginated) &&
    (left.rowId === undefined || right.rowId === undefined || left.rowId === right.rowId) &&
    attachmentSignature(left) === attachmentSignature(right)
  )
}

export function userMessagesMatch(left: ChatMessage, right: ChatMessage): boolean {
  return userMessageIdentityMatches(left, right) &&
    normalizedText(chatMessageText(left)) === normalizedText(chatMessageText(right))
}

/** Tool steering persists a newline-joined batch inside this exact envelope
 * (agent/prompt_builder.py::steer_user_row). ChatMessage does not retain
 * display_kind, so never infer a batch from arbitrary multiline user prose. */
const STEER_OPEN = '[OUT-OF-BAND USER MESSAGE — a direct message from the user, delivered once at this position; not tool output and not a new delivery when replayed from conversation history]\n'
const STEER_CLOSE = '\n[/OUT-OF-BAND USER MESSAGE]'

export function coveredHumanOccurrences(candidate: ChatMessage, humans: ChatMessage[], start: number): number {
  const first = humans[start]

  if (!first) {
    return 0
  }

  if (userMessagesMatch(candidate, first)) {
    return 1
  }

  const text = chatMessageText(candidate)

  if (candidate.userOriginated === false || !text.startsWith(STEER_OPEN) || !text.endsWith(STEER_CLOSE)) {
    return 0
  }

  const body = text.slice(STEER_OPEN.length, -STEER_CLOSE.length)
  const inputs: string[] = []

  for (const human of humans.slice(start)) {
    // A next-turn queue is not part of the tool-steer drain. Retain the usual
    // provenance, durable-id and attachment guards for every grouped input.
    if (human.id.startsWith('user-queued-') || !userMessageIdentityMatches(candidate, human)) {
      return 0
    }

    inputs.push(chatMessageText(human).trim())
    const joined = inputs.join('\n')

    if (body === joined) {
      return inputs.length
    }

    if (!body.startsWith(`${joined}\n`)) {
      return 0
    }
  }

  return 0
}

function partHasRecoverableContent(part: ChatMessagePart): boolean {
  if (part.type === 'text' || part.type === 'reasoning') {
    return typeof part.text === 'string' && part.text.trim().length > 0
  }

  return part.type === 'tool-call'
}

export function assistantHasRecoverableContent(message: ChatMessage): boolean {
  return message.role === 'assistant' && (Boolean(message.error) || message.parts.some(partHasRecoverableContent))
}

/** A live-turn projection row (backend `inflight` via appendLiveSessionProjection,
 *  or a still-streaming local bubble) — as opposed to a completed transcript row. */
export function isLiveProjectionRow(message: ChatMessage): boolean {
  return Boolean(message.pending) || isLiveTailReplyId(message.id)
}

/** A row the transcript actually holds: neither a live projection nor a
 *  journal recovery that is still waiting for its durable counterpart. */
export function isCommittedRow(message: ChatMessage): boolean {
  return !isLiveProjectionRow(message) && !message.recovered
}

function assistantTextLength(message: ChatMessage): number {
  return chatMessageText(message).length
}

/** Merge the journal's last assistant row into the base's live projection row.
 *
 * The journal carries structure (tool calls, reasoning) the backend snapshot
 * lacks; the backend text may be newer than the journal's last throttled
 * write. Keep the journal's parts, but let the longer text win — and keep the
 * BASE row's id so live deltas keep appending to the row the stream handler
 * already targets.
 */
function hasStructuralParts(message: ChatMessage): boolean {
  return message.parts.some(part => part.type === 'reasoning' || part.type === 'tool-call')
}

export function overlayProjectionRow(projection: ChatMessage, journalRow: ChatMessage): ChatMessage {
  // A projected error (retained failed turn) must survive the overlay.
  const error = journalRow.error ?? projection.error

  const merged: ChatMessage = {
    ...journalRow,
    id: projection.id,
    pending: projection.pending,
    // The journal deliberately bounds tool payloads. A matching backend result
    // is richer, even when another result in this same bubble is still missing.
    parts: journalRow.parts.map(part => part.type === 'tool-call'
      ? projection.parts.find(candidate => candidate.type === 'tool-call' &&
          part.toolCallId !== undefined && candidate.toolCallId === part.toolCallId && candidate.result !== undefined) ?? part
      : part),
    ...(error ? { error } : {})
  }

  if (assistantTextLength(projection) <= assistantTextLength(journalRow)) {
    return merged
  }

  // Backend text is newer than the journal's last throttled write — swap it
  // into the journal's first text part, keeping tool calls and reasoning.
  // When the journal already carries structure, only accept a *strict*
  // extension of the answer text. A longer flat dump that starts with
  // thinking chatter must not overwrite / insert as answer text (#76444).
  const projectionText = chatMessageText(projection)
  const journalText = chatMessageText(journalRow).trim()

  if (hasStructuralParts(journalRow)) {
    const next = projectionText.trim()

    if (!journalText || !next.startsWith(journalText)) {
      return merged
    }
  }

  const parts: ChatMessagePart[] = []
  let textReplaced = false

  for (const part of merged.parts) {
    if (part.type !== 'text') {
      parts.push(part)
    } else if (!textReplaced) {
      parts.push({ ...part, text: projectionText })
      textReplaced = true
    }
  }

  if (!textReplaced) {
    parts.push({ type: 'text', text: projectionText })
  }

  return { ...merged, parts }
}

/** The backend boundary of a journaled runtime wake; human turns carry none. */
export function journaledRuntimeStartedAt(messages: ChatMessage[]): number | undefined {
  return messages.find(message => message.runtimeTurnStartedAt !== undefined)?.runtimeTurnStartedAt
}
