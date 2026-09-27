import type { ChatMessage } from '@/lib/chat-messages'
import { isLiveTailReplyId } from '@/lib/spoken-reply'

/** A hydrated bubble can contain several source rows, including the final reply. */
export function transcriptRowIds(message: ChatMessage): number[] {
  const ids = message.parts.flatMap(part => (part.sourceRowId !== undefined ? [part.sourceRowId] : []))

  return message.rowId === undefined ? ids : [message.rowId, ...ids]
}

/** Unknown identity is not a match, but remains eligible for legacy live projection. */
export function conflictingTranscriptIdentity(local: ChatMessage, authoritative: ChatMessage): boolean {
  const localIds = transcriptRowIds(local)
  const authoritativeIds = transcriptRowIds(authoritative)

  return Boolean(localIds.length && authoritativeIds.length && !localIds.some(id => authoritativeIds.includes(id)))
}

/**
 * Compaction carry (#117867): a rewrite re-inserts the live turn's rows under
 * NEW row ids, so disjoint ids prove the local row was SUPERSEDED, not that it
 * is a different row. Callers that use `conflictingTranscriptIdentity` to veto
 * candidates need this arm, or the veto empties the candidate list and every
 * text/tool guard behind it goes blind — the streamed reply is re-appended and
 * paints twice until the window reloads.
 *
 * Evidence is positive on purpose: EVERY id the local bubble names (its folded
 * row plus every row of its turn receipt) is gone from the page — a page that
 * still partly holds the turn has not rewritten it, and a genuinely newer equal
 * reply still holds its own row — and every candidate id is newer than all of
 * them, i.e. a carried generation rather than an older page.
 */
export function carriedOverTranscriptIdentity(
  local: ChatMessage,
  authoritative: ChatMessage,
  pageRowIds: ReadonlySet<number>
): boolean {
  const authoritativeIds = transcriptRowIds(authoritative)

  if (!authoritativeIds.length) {
    return false
  }

  const localIds = [...transcriptRowIds(local), ...(local.persistedTurn?.row_ids ?? [])]

  if (!localIds.length) {
    return false
  }

  if (localIds.some(id => authoritativeIds.includes(id)) || localIds.some(id => pageRowIds.has(id))) {
    return false
  }

  const newestLocal = Math.max(...localIds)

  return authoritativeIds.every(id => id > newestLocal)
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

  next.forEach((message, index) => {
    for (const id of transcriptRowIds(message)) {
      byRow.set(id, index)
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
    const storedIndex = finalRowId !== undefined ? byRow.get(finalRowId) : byId.get(local.id)

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
