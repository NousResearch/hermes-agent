import type { ChatMessage } from '@/lib/chat-messages'
import { isLiveTailReplyId } from '@/lib/spoken-reply'

/** A hydrated bubble can contain several source rows, including the final reply. */
export function transcriptRowIds(message: ChatMessage): number[] {
  const ids = message.parts.flatMap(part => (part.sourceRowId !== undefined ? [part.sourceRowId] : []))

  return message.rowId === undefined ? ids : [message.rowId, ...ids]
}

/** Unknown identity is not a match, but remains eligible for legacy live projection. */
export function conflictingTranscriptIdentity(local: ChatMessage, authoritative: ChatMessage): boolean {
  if (
    local.clientMessageId &&
    authoritative.clientMessageId &&
    local.clientMessageId === authoritative.clientMessageId
  ) {
    return false
  }

  const localIds = transcriptRowIds(local)
  const authoritativeIds = transcriptRowIds(authoritative)

  if (localIds.length && authoritativeIds.length) {
    return !localIds.some(id => authoritativeIds.includes(id))
  }

  return Boolean(
    local.clientMessageId &&
      authoritative.clientMessageId &&
      local.clientMessageId !== authoritative.clientMessageId
  )
}

export function persistedTurnsEquivalent(a: ChatMessage['persistedTurn'], b: ChatMessage['persistedTurn']): boolean {
  return (
    a === b ||
    Boolean(
      a &&
      b &&
      a.complete === b.complete &&
      a.user_row_id === b.user_row_id &&
      a.final_assistant_row_id === b.final_assistant_row_id &&
      a.row_ids.length === b.row_ids.length &&
      a.row_ids.every((id, index) => id === b.row_ids[index])
    )
  )
}

/** Locate an acknowledged boundary on BOTH windows. Prose and clocks are not identity. */
export function acknowledgedTranscriptBoundary(next: ChatMessage[], previous: ChatMessage[]) {
  const byId = new Map(next.map((message, index) => [message.id, index]))
  const byRow = new Map<number, number>()
  const byClientMessageId = new Map<string, number>()

  next.forEach((message, index) => {
    for (const id of transcriptRowIds(message)) {
      byRow.set(id, index)
    }

    if (message.clientMessageId) {
      byClientMessageId.set(message.clientMessageId, index)
    }
  })

  for (let localIndex = previous.length - 1; localIndex >= 0; localIndex -= 1) {
    const local = previous[localIndex]

    if (local.role === 'assistant' && (local.pending || local.interim || local.persistedTurn?.complete === false)) {
      continue
    }

    // A live bubble can still hold unpersisted segments alongside a committed
    // source. Only the terminal receipt proves the whole bubble is covered.
    if (isLiveTailReplyId(local.id) && local.durableComplete !== true) {
      continue
    }

    const finalRowId = local.persistedTurn?.final_assistant_row_id ?? local.rowId

    const storedIndex =
      (finalRowId !== undefined ? byRow.get(finalRowId) : undefined) ??
      (local.clientMessageId ? byClientMessageId.get(local.clientMessageId) : undefined) ??
      byId.get(local.id)

    if (storedIndex === undefined) {
      continue
    }

    const authoritative = next[storedIndex]

    if (
      authoritative.role === local.role &&
      !authoritative.pending &&
      !authoritative.interim &&
      !conflictingTranscriptIdentity(local, authoritative)
    ) {
      return { localIndex, storedIndex }
    }
  }

  return { localIndex: -1, storedIndex: -1 }
}
