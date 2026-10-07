/**
 * Reply comments — many small notes on a chat reply, one send.
 *
 * OpenChamber-style flow: the user selects text in a chat message, writes a
 * short note, and presses Enter to pin it as a chip above the composer. They
 * can pin several chips and send everything in a single turn.
 *
 * Transport contract (renderer-only, no backend change): at the submit
 * boundary the pending comments are frozen into ordinary Markdown — one
 * blockquote of the quoted passage plus a labeled note line per comment —
 * ahead of the typed draft. The model already reads `> quote` as citation,
 * so no new wire format is introduced.
 */

export interface ReplyComment {
  /** Renderer-lifetime id (uuid). Never sent; only for chip keying. */
  id: string
  /** Normalized quoted passage from the chat reply. */
  quote: string
  /** The user's note on that passage (may be empty). */
  note: string
}

/** Whitespace-collapsed budget for one quoted passage. */
export const REPLY_COMMENT_QUOTE_CHARS = 800
/** Budget for one note. */
export const REPLY_COMMENT_NOTE_CHARS = 500
/** Max pinned comments per composer (fail-soft: extras are refused). */
export const MAX_REPLY_COMMENTS = 10

const ELLIPSIS = '…'

export function normalizeReplyQuote(raw: string): string {
  const collapsed = raw.replace(/\s+/g, ' ').trim()

  if (collapsed.length <= REPLY_COMMENT_QUOTE_CHARS) {
    return collapsed
  }

  return `${collapsed.slice(0, REPLY_COMMENT_QUOTE_CHARS).trimEnd()}${ELLIPSIS}`
}

export function normalizeReplyNote(raw: string): string {
  const collapsed = raw.replace(/\s+/g, ' ').trim()

  if (collapsed.length <= REPLY_COMMENT_NOTE_CHARS) {
    return collapsed
  }

  return `${collapsed.slice(0, REPLY_COMMENT_NOTE_CHARS).trimEnd()}${ELLIPSIS}`
}

const createReplyCommentId = (): string =>
  globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`

/** Null when there is no quoted passage — a comment needs an anchor. */
export function createReplyComment(quote: string, note: string, id = createReplyCommentId()): ReplyComment | null {
  const normalizedQuote = normalizeReplyQuote(quote)

  if (!normalizedQuote) {
    return null
  }

  return { id, note: normalizeReplyNote(note), quote: normalizedQuote }
}

/** One transport block: blockquote of the passage + labeled note line. */
export function formatReplyCommentBlock(comment: ReplyComment): string {
  const quoted = comment.quote
    .split('\n')
    .map(line => `> ${line}`.trimEnd())
    .join('\n')

  return comment.note ? `${quoted}\nNote: ${comment.note}` : quoted
}

export function serializeReplyComments(comments: readonly ReplyComment[]): string {
  return comments.map(formatReplyCommentBlock).join('\n\n')
}

/**
 * Freeze pending comments into the draft at a send boundary, ahead of what
 * the user typed (mirrors the follow-up passage order). Empty list returns
 * the draft unchanged.
 */
export function mergeReplyCommentsIntoDraft(draft: string, comments: readonly ReplyComment[]): string {
  if (comments.length === 0) {
    return draft
  }

  const blocks = serializeReplyComments(comments)

  return draft.trim() ? `${blocks}\n\n${draft}` : blocks
}
