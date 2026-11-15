/**
 * A refreshed page that holds a whole turn the window drew live supersedes the
 * window's copy of that turn, decided by the turn's completion receipt.
 *
 * The stored-id merge (transcript-backfill `mergeOverlappingTail`) pairs rows
 * by stored id. A live turn reaches it with stored ids on only some of its rows:
 * a stopped turn binds none, a steered turn's receipt is not `complete`, and the
 * narration the stream sealed before each tool never has one. Those rows used
 * to travel as "unstored" rows beside the stored copies the page brings, so the
 * whole turn, prompt included, rendered twice (#139741), or the narration
 * re-rendered under the final answer once the duplicate final collapsed
 * (#141771).
 *
 * `message.complete.persisted_turn` names every stored row of the turn
 * (`row_ids`) and the turn's own prompt (`submission_id`, `user_row_id(s)`).
 * The page is one contiguous run of the newest stored rows, so a page that
 * holds the turn's first row and reaches its last row holds the whole turn.
 * The window's rows from that prompt to the receipt bubble are then the live
 * copy of rows the page carries, and only the page's copy is kept. A local
 * notice drawn between them (see `isLocalSystemNotice`) is no row of the turn
 * and is never dropped.
 */

import type { PersistedTurn } from '@hermes/shared'

import { isRowId, isTurnUserBubble, receiptUserRowIds } from '@/app/session/hooks/use-message-stream/receipt-user-rows'
import { transcriptRowIds } from '@/app/session/hooks/use-session-actions/pending-turn-identity'
import { type ChatMessage, isLocalSystemNotice } from '@/lib/chat-messages'

/** Index of the turn's prompt in the window, by the receipt's own identity. */
function receiptPromptIndex(window: ChatMessage[], receipt: PersistedTurn, before: number): number {
  const submission = typeof receipt.submission_id === 'string' && receipt.submission_id ? receipt.submission_id : ''
  const userRows = [receipt.user_row_id, ...(receipt.user_row_ids ?? [])].filter(isRowId)

  for (let index = before; index >= 0; index -= 1) {
    const message = window[index]

    if (message.role !== 'user') {
      continue
    }

    if (
      (submission && message.id === `user-${submission}`) ||
      (message.rowId !== undefined && userRows.includes(message.rowId))
    ) {
      // A steer row of the same turn sits after the prompt: keep walking while
      // an earlier user row still belongs to this receipt.
      const earlier = receiptPromptIndex(window, receipt, index - 1)

      return earlier >= 0 && earlier < index ? earlier : index
    }
  }

  return -1
}

/** First and last stored row of the receipt on `message`, when it names any. */
function receiptBounds(message: ChatMessage): null | { first: number; last: number; receipt: PersistedTurn } {
  const receipt = message.persistedTurn
  const rows = (receipt?.row_ids ?? []).filter(isRowId)

  return receipt && rows.length ? { first: Math.min(...rows), last: Math.max(...rows), receipt } : null
}

/** Window indices of the turn ending at `end`, or none when its identity is incomplete. */
function windowTurnIndices(window: ChatMessage[], end: number, first: number, last: number, receipt: PersistedTurn) {
  const start = receiptPromptIndex(window, receipt, end)

  if (start < 0) {
    return []
  }

  const indices: number[] = []

  for (let index = start; index <= end; index += 1) {
    // A notice this window drew mid-turn (a /btw answer, a fallback switch) is
    // not a row of the turn: the page cannot carry it, so it stays.
    if (isLocalSystemNotice(window[index])) {
      continue
    }

    const ids = transcriptRowIds(window[index])

    // A row that carries a stored id outside the turn is not this turn's copy:
    // the pairing is wrong, so nothing is dropped.
    if (ids.some(id => id < first || id > last)) {
      return []
    }

    indices.push(index)
  }

  return indices
}

/** Stored ids of the reply text rows the page paints (an answer or a stopped partial, never a tool row). */
const pageReplyRows = (page: ChatMessage[]): Set<number> =>
  new Set(
    page.flatMap(message =>
      message.role === 'assistant'
        ? message.parts.flatMap(part =>
            part.type === 'text' && part.sourceRowId !== undefined ? [part.sourceRowId] : []
          )
        : []
    )
  )

/**
 * Whether the page's copy of the turn includes its reply. A receipt that is not `complete` can
 * address only the rows that were stored (agent/persisted_turn_receipt.py): when the final answer
 * failed to store, the turn's last stored row is a tool row and the window's bubble is the only copy
 * of the answer. Proven by a complete receipt, by the page painting the turn's last stored row as
 * reply text, or by a stopped turn (everything it produced before the stop is stored).
 */
function pageHoldsTheReply(message: ChatMessage, last: number, replyRows: Set<number>): boolean {
  return Boolean(message.persistedTurn?.complete || message.interrupted) || replyRows.has(last)
}

/** Window indices of the receipt turn ending at `end` that the page holds whole. */
function receiptTurnIndices(
  window: ChatMessage[],
  end: number,
  pageRows: Set<number>,
  replyRows: Set<number>
): number[] {
  const bounds = receiptBounds(window[end])

  if (
    !bounds ||
    !pageRows.has(bounds.first) ||
    Math.max(...pageRows) < bounds.last ||
    !pageHoldsTheReply(window[end], bounds.last, replyRows)
  ) {
    return []
  }

  return windowTurnIndices(window, end, bounds.first, bounds.last, bounds.receipt)
}

/** The window without the turns the page holds whole, and where their notices go. */
export interface ReceiptTurnDrop {
  window: ChatMessage[]
  /**
   * A local notice drawn inside a dropped turn, by message id, mapped to the
   * stored id of the next dropped row of that turn (a steer prompt, the final
   * answer). The page carries that row, so the notice goes in front of the
   * page's copy and stays inside its own turn. Anchored to the next row the
   * window still holds instead, it slid below a later turn whose prompt only
   * the page carries (a background-process completion turn).
   */
  noticeAnchors: Map<string, number>
}

/**
 * Stored id of the first dropped row after `at`. A steer prompt the window
 * drew before its row was written carries no id yet; when the turn's user
 * bubbles pair one-to-one with the receipt's user rows, it takes its row's id,
 * so the notice stays above the steer instead of sliding to the final answer.
 */
function nextDroppedRowId(
  window: ChatMessage[],
  indices: number[],
  at: number,
  receipt: PersistedTurn
): number | undefined {
  const userIndices = indices.filter(index => isTurnUserBubble(window[index]))
  const userRows = receiptUserRowIds(receipt)
  const paired = userIndices.length === userRows.length

  for (const index of indices) {
    if (index <= at) {
      continue
    }

    const ids = transcriptRowIds(window[index])

    if (ids.length) {
      return ids[0]
    }

    const userRow = paired ? userRows[userIndices.indexOf(index)] : undefined

    if (userRow !== undefined) {
      return userRow
    }
  }

  return undefined
}

/** Drop the window rows whose turn the page holds whole, by the turn's receipt. */
export function dropReceiptTurnsThePageHolds(window: ChatMessage[], page: ChatMessage[]): ReceiptTurnDrop {
  const pageRows = new Set(page.flatMap(transcriptRowIds))
  const noticeAnchors = new Map<string, number>()

  if (!pageRows.size) {
    return { window, noticeAnchors }
  }

  const dropped = new Set<number>()
  const replyRows = pageReplyRows(page)

  window.forEach((message, index) => {
    if (!message.persistedTurn || message.pending) {
      return
    }

    const indices = receiptTurnIndices(window, index, pageRows, replyRows)

    indices.forEach(at => dropped.add(at))

    for (let at = indices[0] ?? index; at < index; at += 1) {
      const anchor = isLocalSystemNotice(window[at])
        ? nextDroppedRowId(window, indices, at, message.persistedTurn)
        : undefined

      if (anchor !== undefined) {
        noticeAnchors.set(window[at].id, anchor)
      }
    }
  })

  return { window: dropped.size ? window.filter((_, index) => !dropped.has(index)) : window, noticeAnchors }
}

/**
 * A turn stopped during its tool run ends on stored rows no page paints: the
 * tool result folds into its call and the empty closing assistant row is
 * skipped. A page that holds the turn therefore never reaches its last row,
 * the window's live copy was never dropped, and the page's copy of the
 * narration and tool drew beside it. When the page holds the stopped turn's
 * first row but not its last, the window's live copy (its receipt bound by the
 * interrupted completion, carrying the "Response stopped" mark) is the whole
 * turn, and the page's rows inside the receipt are the partial copy that goes.
 */
export function dropPageRowsOfStoppedTurns(window: ChatMessage[], page: ChatMessage[]): ChatMessage[] {
  const pageRows = new Set(page.flatMap(transcriptRowIds))

  if (!pageRows.size) {
    return page
  }

  const pageLast = Math.max(...pageRows)
  const liveRows = new Set<number>()

  window.forEach((message, index) => {
    const bounds = message.interrupted && !message.pending ? receiptBounds(message) : null

    if (
      bounds &&
      pageRows.has(bounds.first) &&
      pageLast < bounds.last &&
      windowTurnIndices(window, index, bounds.first, bounds.last, bounds.receipt).length
    ) {
      bounds.receipt.row_ids.filter(isRowId).forEach(id => liveRows.add(id))
    }
  })

  const partial = (message: ChatMessage) => {
    const rows = transcriptRowIds(message)

    return rows.length > 0 && rows.every(id => liveRows.has(id))
  }

  return liveRows.size && page.some(partial) ? page.filter(message => !partial(message)) : page
}

/**
 * The newest page can start inside an older tool turn. History folds that turn
 * under its first stored row, so the page's fold carries only the rows from
 * the page boundary on, under a different first row than the window's fold of
 * the same turn. Paired by first row, both folds rendered and the turn's later
 * narration and answer appeared again under the whole turn (#141771). A page
 * fold whose stored rows the window's fold all carries, while the window's
 * fold also carries rows older than the page, is that turn cut by the page
 * boundary: the window's whole fold stays and the page's partial copy goes.
 */
export function dropPageFoldsCutByTheBoundary(window: ChatMessage[], page: ChatMessage[]): ChatMessage[] {
  const pageRows = page.flatMap(transcriptRowIds)

  if (!pageRows.length) {
    return page
  }

  const pageFirst = Math.min(...pageRows)

  const wholeFolds = window
    .map(message => new Set(transcriptRowIds(message)))
    .filter(rows => [...rows].some(id => id < pageFirst))

  const cut = (message: ChatMessage) => {
    const rows = transcriptRowIds(message)

    return rows.length > 0 && wholeFolds.some(whole => rows.every(id => whole.has(id)))
  }

  return page.some(cut) ? page.filter(message => !cut(message)) : page
}
