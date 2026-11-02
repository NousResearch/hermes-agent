import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { isLiveTailReplyId } from '@/lib/spoken-reply'

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
 * live row's own prompt, end in a reply, and say at least what the live row
 * says. A live row still short of the stored text (its tail lost to a
 * reconnect, or still in flight) is the same reply: see livePlaceOnCommittedPage.
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
      committedReplyCovers(message, said)
  )
}

/**
 * The committed bubble a live row replaces, and the row that stands there: the
 * live row, so the turn's late deltas and its completion still find it by id.
 * A live row short of the stored reply takes the stored body, because its tail
 * may never arrive (a completion lost to a reconnect). Keeping both bubbles
 * instead painted the reply twice until the completion landed.
 */
export function livePlaceOnCommittedPage(
  page: ChatMessage[],
  committedNew: (message: ChatMessage) => boolean,
  live: ChatMessage,
  liveMessages: ChatMessage[]
): { index: number; row: ChatMessage } | null {
  const index = pageReplyCommittingLiveRow(page, committedNew, live, liveMessages)

  if (index < 0) {
    return null
  }

  const stored = page[index]

  return { index, row: spokenText(stored).trim() === spokenText(live).trim() ? live : { ...live, parts: stored.parts } }
}
