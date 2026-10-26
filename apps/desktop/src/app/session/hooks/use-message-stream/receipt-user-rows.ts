import type { PersistedTurn } from '@hermes/shared'

import { transcriptRowIds } from '@/app/session/hooks/use-session-actions/pending-turn-identity'
import type { ChatMessage } from '@/lib/chat-messages'

const isRowId = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value > 0

/** The receipt's stored user rows in turn order (prompt first, then each steer/redirect row). */
function receiptUserRowIds(receipt: PersistedTurn): number[] {
  const listed = receipt.user_row_ids

  if (Array.isArray(listed) && listed.length && listed.every(isRowId)) {
    return listed
  }

  // A host that lists no user rows still names the prompt; on its own that is the
  // whole set only when the receipt vouches for the entire turn.
  return isRowId(receipt.user_row_id) && receipt.complete === true ? [receipt.user_row_id] : []
}

const isTurnUserBubble = (message: ChatMessage) =>
  message.role === 'user' && !message.hidden && !message.id.startsWith('user-queued-')

/**
 * Give the turn's on-screen user bubbles the stored row ids its completion
 * receipt names. A host that queues the turn (the session authority) writes the
 * prompt row only when the turn runs, and a steer/redirect row is written
 * mid-turn, so no submit acknowledgement can carry them. Without a row id, a
 * later transcript read keyed on stored rows (switch-back graft, peer refresh)
 * keeps the optimistic bubble beside its stored copy and paints the prompt twice.
 *
 * Pairing is positional within the turn and never by text:
 * - the turn starts at the sender's own optimistic prompt (`user-<submission_id>`);
 * - a window that did not send it (another window, a queued follow-up) starts
 *   after the last row that belongs to an earlier turn, and binds only when it
 *   shows exactly the receipt's user rows;
 * - a bubble that already carries a different row id means the pairing is
 *   wrong, and nothing is bound.
 */
export function bindReceiptUserRows(messages: ChatMessage[], receipt: PersistedTurn | null | undefined): ChatMessage[] {
  const ids = receipt ? receiptUserRowIds(receipt) : []

  if (!receipt || !ids.length) {
    return messages
  }

  const anchor =
    typeof receipt.submission_id === 'string' && receipt.submission_id
      ? messages.findIndex(message => message.role === 'user' && message.id === `user-${receipt.submission_id}`)
      : -1

  const turnRows = new Set(receipt.row_ids)

  const start =
    anchor >= 0
      ? anchor
      : messages.findLastIndex(message => {
          const rows = transcriptRowIds(message)

          return rows.length > 0 && !rows.some(id => turnRows.has(id))
        }) + 1

  const candidates = messages.flatMap((message, index) => (index >= start && isTurnUserBubble(message) ? [index] : []))

  // Unanchored, a missing bubble makes every later pairing ambiguous. Anchored,
  // the prompt is still certain even when a later bubble is absent.
  const paired =
    anchor >= 0 ? (candidates.length >= ids.length ? ids.length : 1) : candidates.length === ids.length ? ids.length : 0

  const binding = new Map<number, number>()

  for (let i = 0; i < paired; i += 1) {
    const current = messages[candidates[i]].rowId

    if (current !== undefined && current !== ids[i]) {
      return messages
    }

    if (current === undefined) {
      binding.set(candidates[i], ids[i])
    }
  }

  if (!binding.size) {
    return messages
  }

  return messages.map((message, index) => {
    const rowId = binding.get(index)

    return rowId === undefined ? message : { ...message, rowId }
  })
}
