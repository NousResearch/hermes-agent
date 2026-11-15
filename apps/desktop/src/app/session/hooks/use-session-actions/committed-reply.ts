import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { isLiveTailReplyId } from '@/lib/spoken-reply'

import { isLiveTailRow } from './optimistic-user-dedup'
import { transcriptRowIds } from './pending-turn-identity'

/**
 * True when `next` is a pure forward extension of the previous *answer* text.
 * Empty previous answer never accepts a dump as an extension — that is how the
 * mid-turn inflight flat dump used to sandwich structured rows (#76444).
 */
export function isStrictAnswerTextExtension(next: string, previous: string): boolean {
  const n = next.trim()
  const p = previous.trim()

  if (!p || !n) {
    return false
  }

  return n.startsWith(p)
}

/** A folded tool-turn bubble's answer: its text after the last tool call. */
export function replyAfterLastTool(message: ChatMessage): string {
  const lastTool = message.parts.findLastIndex(part => part.type === 'tool-call')

  return chatMessageText({ ...message, parts: message.parts.slice(lastTool + 1) })
}

const spokenText = (message: ChatMessage) =>
  message.parts
    .flatMap(part => (part.type === 'text' ? [part.text] : []))
    .join(' ')
    .replace(/\s+/g, ' ')

/**
 * Whether a committed bubble already holds the reply a stale live snapshot was
 * streaming. History folds a tool turn's narration, tools and answer into one
 * bubble while the snapshot carries the turn's whole text, so compare the
 * bubble's text with whitespace collapsed. The bubble must end in a reply: a
 * mid-turn commit ends on a tool call and the turn is still running.
 */
export function committedReplyCovers(message: ChatMessage, snapshot: string): boolean {
  return (
    Boolean(replyAfterLastTool(message).trim()) &&
    isStrictAnswerTextExtension(spokenText(message), snapshot.replace(/\s+/g, ' '))
  )
}

const promptText = (message: ChatMessage | undefined) =>
  message ? textWithoutReferenceLines(chatMessageText(message)).trim() : ''

/**
 * Where a transcript page already commits a still-streaming live reply, or -1.
 *
 * A re-attach reads the persisted page over REST while the turn's events ride
 * the socket, so the page can carry the turn's final row before its
 * `message.complete` reaches the live row. Placing the live row beside that
 * bubble paints the reply twice until the completion lands. The live row keeps
 * the turn: its late deltas and the completion still find it by id, while a
 * dropped live row would be re-seeded by the next delta. So it takes the
 * committed bubble's place. `committedNew` admits only page bubbles that are
 * new since the baseline and not themselves live; the bubble must follow the
 * live row's own prompt, end in a reply, and say exactly what the live row
 * says. A live row still short of the stored text keeps both: its tail may
 * never arrive (a completion lost to a reconnect), and only the committed
 * bubble holds the whole reply.
 */
export function pageReplyCommittingLiveRow(
  page: ChatMessage[],
  committedNew: (message: ChatMessage) => boolean,
  live: ChatMessage,
  liveMessages: ChatMessage[]
): number {
  if (live.role !== 'assistant' || live.pending !== true || live.error || !isLiveTailReplyId(live.id)) {
    return -1
  }

  const ownPrompt = liveMessages.slice(0, liveMessages.indexOf(live)).findLast(message => message.role === 'user')
  const pagePrompt = page.findLastIndex(message => message.role === 'user')

  if (pagePrompt < 0 || !promptText(ownPrompt) || promptText(ownPrompt) !== promptText(page[pagePrompt])) {
    return -1
  }

  const said = spokenText(live).trim()

  return page.findIndex(
    (message, index) =>
      index > pagePrompt &&
      message.role === 'assistant' &&
      committedNew(message) &&
      Boolean(replyAfterLastTool(message).trim()) &&
      spokenText(message).trim() === said
  )
}

/** Text of the response that follows a folded tool round, not the commentary before it. */
export function lastFoldedResponseText(message: ChatMessage): string {
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

const toolCallIdsOf = (message: ChatMessage) =>
  message.parts.flatMap(part => (part.type === 'tool-call' ? [part.toolCallId] : []))

const textPartsOf = (message: ChatMessage) =>
  message.parts.flatMap(part => {
    const text = part.type === 'text' ? textWithoutReferenceLines(part.text).trim() : ''

    return text ? [text] : []
  })

const isPrompt = (message: ChatMessage) =>
  message.role === 'user' && !chatMessageText(message).trimStart().startsWith('[System:')

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

  const sealed =
    live.pending !== true || live.interim === true || folds.some(fold => fold.durableComplete === true)

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
