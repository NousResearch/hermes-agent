// Folded-turn coverage — the durable-fold family of preserveLocalPendingTurnMessages.
//
// History folds a tool-heavy turn into one committed bubble (narration, tool
// rounds, answer joined under one row id), while the renderer's live tail
// sealed each segment as its own row. Every guard here answers one question:
// does the committed window already carry a live row's content, so the local
// copy is a stale duplicate that must retire instead of append beside its
// durable twin (#131500, #119540, #118670)?
import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import type { ChatMessage } from '@/lib/chat-messages'
import { chatMessageText } from '@/lib/chat-messages'
import { isLiveTailReplyId } from '@/lib/spoken-reply'

import { transcriptRowIds } from './pending-turn-identity'

/** A live-turn row, as opposed to a committed transcript row (see utils.ts). */
const isLiveTailRow = (message: ChatMessage): boolean =>
  message.pending === true || isLiveTailReplyId(message.id) || message.interim === true

export const isGatewaySystemMarker = (message: ChatMessage): boolean =>
  message.role === 'user' && chatMessageText(message).trimStart().startsWith('[System:')

export const toolCallIdsOf = (message: ChatMessage) =>
  message.parts.flatMap(part => (part.type === 'tool-call' ? [part.toolCallId] : []))

const textPartsOf = (message: ChatMessage) =>
  message.parts.flatMap(part => {
    const text = part.type === 'text' ? textWithoutReferenceLines(part.text).trim() : ''

    return text ? [text] : []
  })

export const isPrompt = (message: ChatMessage) => message.role === 'user' && !isGatewaySystemMarker(message)

/** Text of the response that follows a folded tool round, not the commentary before it. */
function lastFoldedResponseText(message: ChatMessage): string {
  let afterTool = false
  let text = ''

  for (const part of message.parts) {
    if (part.type === 'tool-call') {
      afterTool = true
      text = ''

      continue
    }

    if (afterTool && part.type === 'text') {
      text = textWithoutReferenceLines(part.text).trim()
    }
  }

  return afterTool ? text : ''
}

/**
 * Committed rows folding the local turn around `index`, for ANY turn, not just
 * the latest: rows sharing a tool call id with that turn, plus the assistant
 * rows after its prompt's durable twin. Only the latest turn may fall back to
 * position (everything after the last stored prompt).
 */
export function committedFoldsOfLocalTurn(
  candidates: ChatMessage[],
  previous: ChatMessage[],
  index: number
): ChatMessage[] {
  const start = previous.findLastIndex((row, at) => at < index && isPrompt(row))
  const end = previous.findIndex((row, at) => at > index && isPrompt(row))

  const turnToolIds = new Set(
    previous
      .slice(start + 1, end < 0 ? undefined : end)
      .flatMap(toolCallIdsOf)
      .filter(Boolean)
  )

  const owner = previous[start]
  const ownerRowIds = owner ? transcriptRowIds(owner) : []

  const anchor = owner
    ? candidates.findIndex(row => row.id === owner.id || transcriptRowIds(row).some(id => ownerRowIds.includes(id)))
    : -1

  const from = anchor >= 0 ? anchor : end < 0 ? candidates.findLastIndex(isPrompt) : candidates.length
  const until = candidates.findIndex((row, at) => at > from && isPrompt(row))
  const segment = new Set(candidates.slice(from + 1, until < 0 ? undefined : until))

  return candidates.filter(
    row =>
      row.role === 'assistant' &&
      !isLiveTailRow(row) &&
      (segment.has(row) || toolCallIdsOf(row).some(id => turnToolIds.has(id)))
  )
}

const hasWholeLines = (haystack: string, needle: string) => `\n${haystack}\n`.includes(`\n${needle}\n`)

/**
 * A fold carries sealed text verbatim as a text part, or inside Thinking when
 * the provider stored public commentary in `reasoning` (Codex Responses, #119716).
 */
const foldCarriesText = (fold: ChatMessage, text: string) =>
  fold.parts.some(part =>
    part.type === 'text'
      ? textWithoutReferenceLines(part.text).trim() === text
      : part.type === 'reasoning' && hasWholeLines(part.text, text)
  )

/**
 * History folds a tool-heavy turn into one bubble, while the live stream sealed
 * each interim segment and the final answer as bubbles of their own. A sealed
 * live bubble is that same occurrence when its turn's folds hold every tool
 * call it ran (durable identity) and its text, either verbatim (a sealed middle
 * segment, #119540) or as the final answer, equal or extended (#118670). This
 * holds with a partial or missing completion receipt, where full-bubble
 * equality sees neither.
 */
export function durableFoldCoversLiveResponse(folds: ChatMessage[], live: ChatMessage): boolean {
  const liveToolIds = toolCallIdsOf(live)
  const sealed = live.pending !== true || live.interim === true

  if (!folds.length || (liveToolIds.length && (!sealed || liveToolIds.some(id => !id)))) {
    return false
  }

  const liveTexts = textPartsOf(live)

  const answer = liveToolIds.length
    ? lastFoldedResponseText(live)
    : textWithoutReferenceLines(chatMessageText(live)).trim()

  if (!answer && !liveTexts.length && !liveToolIds.length) {
    return false
  }

  const foldedToolIds = new Set(folds.flatMap(toolCallIdsOf))

  if (!liveToolIds.every(id => foldedToolIds.has(id))) {
    return false
  }

  if (sealed && liveTexts.every(text => folds.some(fold => foldCarriesText(fold, text)))) {
    return true
  }

  return (
    Boolean(answer) &&
    folds.some(fold => {
      const folded = lastFoldedResponseText(fold)

      return folded === answer || isStrictAnswerTextExtension(folded, answer)
    })
  )
}

/**
 * True when `next` is a pure forward extension of the previous *answer* text.
 * Empty previous answer never accepts a dump as an extension — that is how the
 * mid-turn inflight flat dump used to sandwich structured rows (#76444).
 * Re-exported by utils.ts for its existing call sites there.
 */
export function isStrictAnswerTextExtension(next: string, previous: string): boolean {
  const n = next.trim()
  const p = previous.trim()

  if (!p || !n) {
    return false
  }

  return n.startsWith(p)
}

/** Answer text of a row under the fold compare: the text after the last tool
 *  call for a tool turn, the whole text otherwise — reference lines stripped,
 *  separators folded, so a store-side join difference is not a mismatch. */
const foldedAnswerForCompare = (message: ChatMessage): string => {
  const raw = toolCallIdsOf(message).length
    ? lastFoldedResponseText(message)
    : textWithoutReferenceLines(chatMessageText(message))

  return raw.replace(/\s+/g, ' ').trim()
}

/**
 * #131500: is a committed fold the durable twin of the still-live local row?
 *
 * A reconnect can leave a stale live copy in the renderer's in-memory list
 * while the durable refresh delivers the same reply folded under a
 * different (positionally synthesized) id. Ordinal pairing misses it — a
 * hidden directive in the turn shifts the ordinals — and full-text compare
 * misses it too: history folds narration, tools and answer into one bubble,
 * and the two sides join segments with different separators. Both copies
 * then render until a process restart rebuilds state.
 *
 * Matching mirrors assistantTimelineMatch's arms (chat-messages/
 * reconciliation.ts), anchored to the live row's own user turn by
 * committedFoldsOfLocalTurn so a repeated answer in an older turn cannot
 * pass: row id, tool-call id overlap, or normalized final-text equality.
 * The fold must also HOLD the live row's answer — equal (the stale copy's
 * exact content) or further along (the settled final, mirroring the
 * committedMatch drop). A lagging partial never retires the stream: that
 * mid-turn flush is the committedPrefix replacement / fold-carry path.
 */
export function committedTwinCoversLiveResponse(folds: ChatMessage[], live: ChatMessage): boolean {
  const liveRowIds = transcriptRowIds(live)
  const liveToolIds = toolCallIdsOf(live)
  const liveAnswer = foldedAnswerForCompare(live)

  return folds.some(fold => {
    if (liveRowIds.length && transcriptRowIds(fold).some(id => liveRowIds.includes(id))) {
      return true
    }

    if (liveToolIds.length && toolCallIdsOf(fold).some(id => liveToolIds.includes(id))) {
      const foldAnswer = foldedAnswerForCompare(fold)

      return Boolean(foldAnswer) && (foldAnswer === liveAnswer || isStrictAnswerTextExtension(foldAnswer, liveAnswer))
    }

    return Boolean(liveAnswer) && liveAnswer === foldedAnswerForCompare(fold)
  })
}
