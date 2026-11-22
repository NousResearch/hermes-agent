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
import { type ChatMessage, type ChatMessagePart, isLocalSystemNotice } from '@/lib/chat-messages'

const receiptUserRows = (receipt: PersistedTurn) =>
  [receipt.user_row_id, ...(receipt.user_row_ids ?? [])].filter(isRowId)

/**
 * Index of the turn's prompt in the window, by the receipt's own identity. A backend-started
 * turn's prompt row (an async delegation or process completion) is stored as a user row that
 * history paints as a system card: its stored id, not its role, makes it the prompt.
 */
function receiptPromptIndex(window: ChatMessage[], receipt: PersistedTurn, before: number): number {
  const submission = typeof receipt.submission_id === 'string' && receipt.submission_id ? receipt.submission_id : ''
  const userRows = receiptUserRows(receipt)

  for (let index = before; index >= 0; index -= 1) {
    const message = window[index]

    if (message.role !== 'user' && message.role !== 'system') {
      continue
    }

    if (
      (submission && message.role === 'user' && message.id === `user-${submission}`) ||
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

/**
 * Where a receipt turn whose prompt the window cannot name starts (a backend-started turn on a
 * runtime that stamps no stream turn): the row after the one that was newest when the turn's own
 * message.start arrived (`receiptTurnAnchorId`, local to this window, never a wire identity). The
 * boundary is proven only while every row between it and the receipt bubble can be nothing but
 * this turn's live copy: its sealed narration (interim) and the notices this window drew. A user
 * row there, a stored non-assistant row, or a reply that settled, failed or still streams may be
 * another turn's, and an unnamed user prompt at the anchor is this turn's own prompt that the
 * receipt failed to prove. Any of those, or an anchor no longer on screen, proves nothing (-1).
 */
function anchoredTurnStart(window: ChatMessage[], end: number): number {
  const anchorId = window[end].receiptTurnAnchorId
  const anchor = anchorId ? window.findIndex(message => message.id === anchorId) : -1

  if (anchor < 0 || anchor >= end || (window[anchor].role === 'user' && window[anchor].rowId === undefined)) {
    return -1
  }

  for (let index = anchor + 1; index < end; index += 1) {
    const message = window[index]

    if (isLocalSystemNotice(message)) {
      continue
    }

    if (message.role !== 'assistant' || !message.interim || message.pending || message.error || message.persistedTurn) {
      return -1
    }
  }

  return anchor + 1
}

/** Window indices of the turn ending at `end`, or none when its identity is incomplete. */
function windowTurnIndices(
  window: ChatMessage[],
  end: number,
  first: number,
  last: number,
  receipt: PersistedTurn,
  { anchored = false } = {}
) {
  const prompt = receiptPromptIndex(window, receipt, end)
  const start = prompt < 0 && anchored ? anchoredTurnStart(window, end) : prompt

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
 * reply text, or by a stopped turn (everything it produced before the stop is stored). A receipt that
 * names its final row needs that very row on the page: a later turn's rows reaching past it are no copy.
 */
function pageHoldsTheReply(message: ChatMessage, last: number, replyRows: Set<number>): boolean {
  const final = message.persistedTurn?.final_assistant_row_id

  if (isRowId(final)) {
    return replyRows.has(final)
  }

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

  // A hidden starter has no rendered row. An earlier visible row establishes that the
  // contiguous page starts before it; a page starting inside the turn proves nothing.
  const reachesStarter =
    bounds && (pageRows.has(bounds.first) || (window[end].receiptTurnAnchorId && Math.min(...pageRows) < bounds.first))

  if (
    !bounds ||
    !reachesStarter ||
    Math.max(...pageRows) < bounds.last ||
    !pageHoldsTheReply(window[end], bounds.last, replyRows)
  ) {
    return []
  }

  // The page holds the whole turn and its reply; with no prompt identity on screen, the local
  // start boundary may stand in (only here: a stopped turn's partial copy keeps its prompt rule).
  return windowTurnIndices(window, end, bounds.first, bounds.last, bounds.receipt, { anchored: true })
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

type ToolCallPart = Extract<ChatMessagePart, { type: 'tool-call' }>

/** A part addressed by its message and part index. */
interface PartAt {
  at: number
  part: number
}

const partKey = ({ at, part }: PartAt) => `${at}:${part}`

/** A live row whose own state the page cannot carry: its text and its card stay. */
const keepsItsReply = (message: ChatMessage) => Boolean(message.pending || message.error || message.interrupted)

/** One stored row of the turn as the page paints it: its text part and its calls, in order. */
interface StoredRow {
  text?: PartAt
  calls: PartAt[]
}

/**
 * Which stored parts are the turn's own, by exact provenance only: the receipt's row ids (each text
 * part's `sourceRowId`, each call's `sourceCallRowId`), or the stream turn the row was written by
 * (`storedTurnId`, agent/session_persistence.STREAM_TURN_ATTR). A provider's tool-call id, the
 * rows' order and their words prove nothing: the next turn can reuse `call_0` and say the same.
 */
function storedTurnRows(page: ChatMessage[], turnId: string, receipt: PersistedTurn | undefined): StoredRow[] {
  const receiptRows = new Set((receipt?.row_ids ?? []).filter(isRowId))
  const rows = new Map<number, StoredRow>()

  page.forEach((message, at) => {
    if (message.role !== 'assistant') {
      return
    }

    message.parts.forEach((part, index) => {
      const source =
        part.type === 'text' ? part.sourceRowId : part.type === 'tool-call' ? part.sourceCallRowId : undefined

      if (source === undefined || !(receiptRows.has(source) || (part.storedTurnId && part.storedTurnId === turnId))) {
        return
      }

      const row = rows.get(source) ?? { calls: [] }

      if (part.type === 'text') {
        row.text ??= { at, part: index }
      } else {
        row.calls.push({ at, part: index })
      }

      rows.set(source, row)
    })
  })

  return [...rows.values()]
}

/** The live rows' parts of one stream turn, in the order they were drawn. */
function liveTurnParts(window: ChatMessage[], indices: number[]): PartAt[] {
  return indices.flatMap(at =>
    window[at].parts.flatMap((part, index) => (part.type === 'reasoning' ? [] : [{ at, part: index }]))
  )
}

const partOf = (messages: ChatMessage[], { at, part }: PartAt) => messages[at].parts[part]
const trimmedText = (part: ChatMessagePart) => (part.type === 'text' ? part.text.trim() : '')

/** The id a stored call's live card carries: the provider's, when hydration renamed a repeat. */
const liveIdOf = (part: ToolCallPart) => part.providerToolCallId ?? part.toolCallId

const resultKey = (result: unknown) => (typeof result === 'string' ? result : JSON.stringify(result))

/**
 * Which live call a stored call represents when the two sides have different occurrences of an
 * id: only the one the evidence singles out, by its stored row or the same finished result.
 * Either side may have missed a round, so order alone cannot decide.
 */
function provenOccurrence(window: ChatMessage[], live: PartAt[], stored: ToolCallPart, candidates: number[]) {
  const cards = candidates.map(index => ({ index, card: partOf(window, live[index]) as ToolCallPart }))
  const row = stored.sourceCallRowId
  const byRow = row === undefined ? [] : cards.filter(({ card }) => card.sourceCallRowId === row)

  if (byRow.length === 1) {
    return byRow[0].index
  }

  if (stored.result === undefined) {
    return -1
  }

  const byResult = cards.filter(
    ({ card }) =>
      (row === undefined || card.sourceCallRowId === undefined || card.sourceCallRowId === row) &&
      card.result !== undefined &&
      Boolean(card.isError) === Boolean(stored.isError) &&
      resultKey(card.result) === resultKey(stored.result)
  )

  return byResult.length === 1 ? byResult[0].index : -1
}

/** Walk matching tool-call occurrences in transcript order, without reusing a live call. */
function matchToolOccurrences(window: ChatMessage[], page: ChatMessage[], live: PartAt[], stored: StoredRow[]) {
  const storedLeft = new Map<string, number>()
  const uncertainIds = new Set<string>()

  for (const call of stored.flatMap(row => row.calls)) {
    const id = liveIdOf(partOf(page, call) as ToolCallPart)

    if (id) {
      storedLeft.set(id, (storedLeft.get(id) ?? 0) + 1)
    }
  }

  return (call: PartAt, cursor: number): number => {
    const storedCall = partOf(page, call) as ToolCallPart
    const id = liveIdOf(storedCall)

    if (!id) {
      return -1
    }

    const candidates = live.flatMap((piece, index) => {
      const part = partOf(window, piece)

      return index >= cursor && part.type === 'tool-call' && liveIdOf(part) === id ? [index] : []
    })

    const left = storedLeft.get(id) ?? 0
    storedLeft.set(id, left - 1)

    // Walking past a round that is missing from either side does not make order reliable again.
    if (candidates.length !== left) {
      uncertainIds.add(id)
    }

    return uncertainIds.has(id) ? provenOccurrence(window, live, storedCall, candidates) : (candidates[0] ?? -1)
  }
}

/**
 * Pair the turn's stored parts with the live parts that drew them, occurrence by occurrence:
 * - a live text part bound to a stored row (the final, by its receipt) pairs with that row's text;
 * - each stored call pairs with the next live call of that id (its provider id, when hydration
 *   renamed a repeat inside one fold), in order, so a repeated id pairs one occurrence each and
 *   nothing is matched out of order; when the counts disagree on either side, a stored call pairs
 *   only with the one its evidence proves (`provenOccurrence`), or none;
 * - a stored row's text pairs only with the live text drawn just before that row's first paired
 *   call (a row without calls: just after the previous row's last paired call), and only when the
 *   words agree. The words verify the structurally chosen piece; they never choose it, so equal
 *   narration elsewhere in the turn is a different occurrence and stays.
 * A live piece still streaming, failed or stopped pairs with nothing but its calls.
 */
function pairLiveParts(window: ChatMessage[], page: ChatMessage[], live: PartAt[], stored: StoredRow[]) {
  const pairs = new Map<string, PartAt>()
  const used = new Set<string>()

  const pair = (storedPiece: PartAt, livePiece: PartAt) => {
    pairs.set(partKey(livePiece), storedPiece)
    used.add(partKey(livePiece))
  }

  const textMatches = (storedPiece: PartAt, index: number) => {
    const livePiece = live[index]
    const part = livePiece && partOf(window, livePiece)

    return Boolean(
      part?.type === 'text' &&
      !used.has(partKey(livePiece)) &&
      part.sourceRowId === undefined &&
      !keepsItsReply(window[livePiece.at]) &&
      trimmedText(part) &&
      trimmedText(part) === trimmedText(partOf(page, storedPiece))
    )
  }

  for (const row of stored) {
    const storedText = row.text && partOf(page, row.text)
    const bound = storedText?.type === 'text' ? storedText.sourceRowId : undefined

    const boundPiece = live.find(piece => {
      const part = partOf(window, piece)

      return part.type === 'text' && bound !== undefined && part.sourceRowId === bound
    })

    if (row.text && boundPiece) {
      pair(row.text, boundPiece)
    }
  }

  const matchCall = matchToolOccurrences(window, page, live, stored)
  let cursor = 0

  for (const row of stored) {
    const before = cursor
    let firstCall = -1

    for (const call of row.calls) {
      const found = matchCall(call, cursor)

      if (found < 0) {
        continue
      }

      pair(call, live[found])
      firstCall = firstCall < 0 ? found : firstCall
      cursor = found + 1
    }

    // A row whose calls the stream never drew anchors no text either.
    if (!row.text || [...pairs.values()].includes(row.text) || (row.calls.length > 0 && firstCall < 0)) {
      continue
    }

    const candidate = firstCall >= 0 ? firstCall - 1 : before

    if (textMatches(row.text, candidate)) {
      pair(row.text, live[candidate])
      cursor = Math.max(cursor, candidate + 1)
    }
  }

  return pairs
}

/** A stored call slot the page has no result for yet takes the live card's result. */
function withResultOf(stored: ToolCallPart, live: ToolCallPart): ToolCallPart {
  if (stored.result !== undefined || live.result === undefined) {
    return stored
  }

  return {
    ...stored,
    result: live.result,
    ...(live.isError !== undefined ? { isError: live.isError } : {}),
    ...(live.artifact !== undefined ? { artifact: live.artifact } : {}),
    ...(live.completedAt !== undefined ? { completedAt: live.completedAt } : {}),
    ...(live.toolResultMetadata !== undefined ? { toolResultMetadata: live.toolResultMetadata } : {})
  }
}

/** The same stored call (its row and its id): a refresh keeps a result an earlier one filled in. */
function carryFilledResults(window: ChatMessage[], page: ChatMessage[]): ChatMessage[] {
  const filled = new Map<string, ToolCallPart>()

  for (const part of window.flatMap(message => message.parts)) {
    if (part.type === 'tool-call' && part.sourceCallRowId !== undefined && part.result !== undefined) {
      filled.set(`${part.sourceCallRowId}:${part.toolCallId}`, part)
    }
  }

  if (!filled.size) {
    return page
  }

  let changed = false

  const next = page.map(message => {
    const parts = message.parts.map(part => {
      const earlier =
        part.type === 'tool-call' && part.result === undefined && part.sourceCallRowId !== undefined
          ? filled.get(`${part.sourceCallRowId}:${part.toolCallId}`)
          : undefined

      return part.type === 'tool-call' && earlier ? withResultOf(part, earlier) : part
    })

    if (parts.some((part, index) => part !== message.parts[index])) {
      changed = true

      return { ...message, parts }
    }

    return message
  })

  return changed ? next : page
}

/** The live parts of one turn that the page's stored parts drew, removed or merged into the page. */
function reconcileLiveTurn(window: ChatMessage[], page: ChatMessage[], turnId: string, indices: number[]) {
  const receipt = indices.map(index => window[index].persistedTurn).findLast(Boolean)
  const stored = storedTurnRows(page, turnId, receipt)

  if (!stored.length) {
    return null
  }

  const pairs = pairLiveParts(window, page, liveTurnParts(window, indices), stored)

  if (!pairs.size) {
    return null
  }

  const pagePatches = new Map<string, ToolCallPart>()

  for (const [liveKey, storedPiece] of pairs) {
    const [at, part] = liveKey.split(':').map(Number)
    const livePart = window[at].parts[part]
    const storedPart = partOf(page, storedPiece)

    if (livePart.type === 'tool-call' && storedPart.type === 'tool-call') {
      const filled = withResultOf(storedPart, livePart)

      if (filled !== storedPart) {
        pagePatches.set(partKey(storedPiece), filled)
      }
    }
  }

  const nextPage = pagePatches.size
    ? page.map((message, at) =>
        message.parts.some((_, part) => pagePatches.has(partKey({ at, part })))
          ? {
              ...message,
              parts: message.parts.map((part, index) => pagePatches.get(partKey({ at, part: index })) ?? part)
            }
          : message
      )
    : page

  const rows = new Map<number, ChatMessage | null>()

  for (const at of indices) {
    const message = window[at]
    const parts = message.parts.filter((_, part) => !pairs.has(partKey({ at, part })))

    if (parts.length === message.parts.length) {
      continue
    }

    // Reasoning alone is no reply of its own once everything it led to is stored.
    const kept = keepsItsReply(message) || parts.some(part => part.type !== 'reasoning')
    rows.set(at, kept ? { ...message, parts } : null)
  }

  return { page: nextPage, rows }
}

/**
 * The live rows of a streamed turn (`streamTurnId`) give up every piece the page stores for that
 * turn, by provenance the producer wrote (see `storedTurnRows`), receipt or not: a running turn's
 * committed rounds, the committed rounds of a turn whose receipt covers only part of it, a turn
 * whose receipt was lost. What the page has no proven copy of (the answer still streaming, a final
 * the receipt does not cover, a later round, a result only the stream delivered) stays where it
 * is; a result only the live card has fills the page's slot for that call instead of drawing the
 * call twice. A turn whose stored rows carry no such provenance (an older runtime with no receipt)
 * is left as it is: unknown rows are kept, never guessed at.
 */
export function reconcileLiveTurnsWithPage(
  window: ChatMessage[],
  page: ChatMessage[]
): { window: ChatMessage[]; page: ChatMessage[] } {
  const turns = new Map<string, number[]>()

  window.forEach((message, index) => {
    if (message.streamTurnId && message.role === 'assistant') {
      turns.set(message.streamTurnId, [...(turns.get(message.streamTurnId) ?? []), index])
    }
  })

  let nextPage = carryFilledResults(window, page)
  const rows = new Map<number, ChatMessage | null>()

  for (const [turnId, indices] of turns) {
    const reconciled = reconcileLiveTurn(window, nextPage, turnId, indices)

    if (reconciled) {
      nextPage = reconciled.page
      reconciled.rows.forEach((row, at) => rows.set(at, row))
    }
  }

  const nextWindow = rows.size
    ? window.flatMap((message, index) => {
        const next = rows.has(index) ? rows.get(index) : message

        return next ? [next] : []
      })
    : window

  return { window: nextWindow, page: nextPage }
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
