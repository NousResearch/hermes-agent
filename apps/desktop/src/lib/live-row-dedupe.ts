import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import { chatMessageText } from '@/lib/chat-messages'
import type { ChatMessage } from '@/lib/chat-messages'
import { isLiveTailReplyId } from '@/lib/spoken-reply'

/**
 * A live-turn row — the gateway's text-only `inflight` projection, a
 * still-streaming local bubble, or an interim row sealed inside the running
 * turn — as opposed to a committed transcript row. Mirrors the predicate in
 * use-session-actions/utils.ts, which keeps it private there.
 */
const isLiveTailRow = (message: ChatMessage): boolean =>
  message.pending === true || isLiveTailReplyId(message.id) || message.interim === true

/**
 * #80151: the store's flat projection and the streamed parts join the same
 * segments with different separators (blank lines around folded tool rounds,
 * reference lines), so byte-prefix pairing misses the same turn. Compare the
 * answer text with reference lines stripped and separators folded away.
 *
 * Lives here, not in the actions layer, so `store/session` can share the fold
 * without importing a module that imports the store back.
 */
export const foldAnswerTextForCompare = (text: string): string => textWithoutReferenceLines(text).replace(/\s+/g, '')

/**
 * One reply, one live row.
 *
 * A restored mid-turn session can be projected twice: once by the persisted
 * live turn, whose live row is keyed by the STORED session id
 * (`assistant-stream-<stored>`), and once by the inflight snapshot, keyed by the
 * RUNTIME session id (`assistant-stream-<runtime>`). Both rows carry the same
 * answer, so the reply renders twice - the durable fold never catches it because
 * it compares a live row against committed rows and skips other live rows.
 *
 * Live rows only, and only a text-only row is dropped, so a structured tail
 * (reasoning, tool calls) is never lost to this pass, and the committed-vs-live
 * fold keeps its own logic (#70209, #80151). Returns the SAME array when nothing
 * is dropped: callers rely on reference identity for equivalent transcripts.
 */
export function dropDuplicateLiveAssistantRows(messages: ChatMessage[]): ChatMessage[] {
  // Same turn only: two live rows carrying one answer are a duplicate only when
  // they belong to the same turn. An identical prompt submitted again streams a
  // second live row whose text repeats the first turn's answer, and that row is
  // the new turn's bubble (the last user row separates them).
  const lastUserIndex = messages.findLastIndex(message => message.role === 'user')
  const firstByAnswer = new Map<string, number>()
  const dropped = new Set<number>()

  messages.forEach((message, index) => {
    if (message.role !== 'assistant' || !isLiveTailRow(message) || index <= lastUserIndex) {
      return
    }

    const text = foldAnswerTextForCompare(chatMessageText(message))

    if (!text) {
      return
    }

    const first = firstByAnswer.get(text)

    if (first === undefined) {
      firstByAnswer.set(text, index)

      return
    }

    const textOnly = message.parts.length > 0 && message.parts.every(part => part.type === 'text')

    if (textOnly) {
      dropped.add(index)
    }
  })

  return dropped.size === 0 ? messages : messages.filter((_, index) => !dropped.has(index))
}

/**
 * One reply, one live row.
 *
 * A live row whose answer the committed rows already carry is the reply's second
 * copy. It does not matter whether the row is still flagged as streaming: the app
 * log's proven pair is a committed row (`status={"type":"complete","reason":"stop"}`)
 * beside a live row still carrying `pending=true` / `status={"type":"running"}` for
 * the same answer, minutes after the turn ended - a stale live bubble that nothing
 * clears, which rendered the reply twice.
 *
 * EXACT folded-text match only. A live row genuinely streaming the same answer past
 * the committed text folds differently and survives, so a running turn keeps its
 * live tail; only a row whose whole answer is already represented disappears.
 *
 * Callers must be transcript reconciliation or render paths (store writes, the
 * runtime repository, the restore/activate reconcile). Returns the SAME array when
 * nothing is dropped: callers rely on reference identity for equivalent transcripts.
 */
export function dropLiveRowsRepresentedByCommitted(messages: ChatMessage[]): ChatMessage[] {
  // Same turn only. A committed row BEFORE the latest user row belongs to an
  // earlier turn, and an identical prompt submitted again is a new occurrence
  // whose reply may legitimately repeat the previous answer word for word - that
  // live row is the new turn's bubble and must survive. Only committed rows after
  // the last user row can represent the live row being considered.
  const lastUserIndex = messages.findLastIndex(message => message.role === 'user')
  const committedAnswers = new Set<string>()

  messages.forEach((message, index) => {
    if (message.role !== 'assistant' || isLiveTailRow(message) || index <= lastUserIndex) {
      return
    }

    const text = foldAnswerTextForCompare(chatMessageText(message))

    if (text) {
      committedAnswers.add(text)
    }
  })

  if (committedAnswers.size === 0) {
    return messages
  }

  const dropped = new Set<number>()

  messages.forEach((message, index) => {
    if (message.role !== 'assistant' || !isLiveTailRow(message) || index <= lastUserIndex) {
      return
    }

    const text = foldAnswerTextForCompare(chatMessageText(message))

    if (text && committedAnswers.has(text)) {
      dropped.add(index)
    }
  })

  return dropped.size === 0 ? messages : messages.filter((_, index) => !dropped.has(index))
}
