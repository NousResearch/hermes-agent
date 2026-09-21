/**
 * Quoting a specific room message.
 *
 * Rooms already thread — a reply box per thread — but a thread cannot answer a
 * *particular* line: "that number is wrong", "which file?", "agreed, but …".
 * Chat apps solve that with a quoted reference above the reply, and that is
 * what this is.
 *
 * The quote is a SNAPSHOT (speaker, text, age), not a message id: the room log
 * is trimmed, so a pointer would rot. It is also cut to GROUP_QUOTE_TEXT_CHARS,
 * because the log is synced through profile ui_meta with a byte budget and a
 * quote rides every member's copy of the message.
 */
import type { GroupMessage, GroupQuote } from './types'

/** How much of the quoted message travels with the reply. Long enough to
 *  recognise the line, short enough that a quote cannot dominate the log. */
export const GROUP_QUOTE_TEXT_CHARS = 140

/** Take the quote snapshot for a message, as `from` sees it. */
export function quoteFromMessage(entry: GroupMessage, from: string, at = Date.now()): GroupQuote {
  const text = String(entry.text || '').replace(/\s+/g, ' ').trim()

  return {
    at,
    from,
    text: text.length > GROUP_QUOTE_TEXT_CHARS ? `${text.slice(0, GROUP_QUOTE_TEXT_CHARS - 1).trimEnd()}…` : text
  }
}
