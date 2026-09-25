import type { ChatMessage } from '@/lib/chat-messages'
import { chatMessageText } from '@/lib/chat-messages/parts'

/**
 * Attachment-rewrite tolerance for pasted attachments (#120978, follow-up to
 * #119326).
 *
 * A pasted clipboard attachment has no rowId on the optimistic local row, and
 * the backend rewrites the durable prompt around it: the caption survives but
 * the refs become `[Image attached at: <path>]` / `[screenshot]` markers, and a
 * `<memory-context>…</memory-context>` block can be injected. Exact text+refs
 * equality can never bridge that, so `hydratedIdFor()` returns undefined and
 * the conservative append path paints the local pair below the newest turn.
 *
 * The tolerance is gated on ATTACHMENT EVIDENCE on both sides so a plain
 * repeated prompt — or a second attempt under the same caption — is never
 * swallowed: the stored row must carry rewrite markers AND the local row must
 * carry its own attachment evidence (refs or markers).
 */

// Rewrite markers the backend stamps onto the durable prompt of a pasted
// attachment turn. Kept in one pattern so a new marker shape lands here once.
const ATTACHMENT_REWRITE_MARKER_RE = /\[Image attached at:[^\]]*\]|\[(?:screenshot|image|attachment|file)\]/i

const MEMORY_CONTEXT_RE = /<memory-context>[\s\S]*?(?:<\/memory-context>|$)/gi
const IMAGE_ATTACHED_RE = /\[Image attached at:[^\]]*\]/gi
const SHORT_MARKER_RE = /\[(?:screenshot|image|attachment|file)\]/gi

/** The stored row carries backend rewrite markers for a pasted attachment. */
export const carriesAttachmentRewrite = (value: string): boolean => ATTACHMENT_REWRITE_MARKER_RE.test(value)

/**
 * The caption a rewrite buried: memory-context block, path markers and short
 * markers stripped, whitespace collapsed. Empty when the row was markers only.
 */
export const attachmentTolerantUserText = (value: string): string =>
  value
    .replace(MEMORY_CONTEXT_RE, ' ')
    .replace(IMAGE_ATTACHED_RE, ' ')
    .replace(SHORT_MARKER_RE, ' ')
    .replace(/\s+/g, ' ')
    .trim()

/** Either side showing attachment evidence: real refs or rewrite markers. */
const carriesAttachmentEvidence = (message: ChatMessage): boolean =>
  (message.attachmentRefs ?? []).some(ref => ref.trim().length > 0) ||
  carriesAttachmentRewrite(chatMessageText(message))

/**
 * Same attachment turn, tolerantly: the stored row carries rewrite markers,
 * the local row carries attachment evidence, and the tolerant captions are
 * equal. The marker gate means a plain repeat of the caption never matches.
 */
export const sameAttachmentTurn = (stored: ChatMessage, local: ChatMessage): boolean => {
  if (stored.role !== 'user' || local.role !== 'user') {
    return false
  }

  const storedText = chatMessageText(stored)

  if (!carriesAttachmentRewrite(storedText)) {
    return false
  }

  if (!carriesAttachmentEvidence(local)) {
    return false
  }

  return attachmentTolerantUserText(storedText) === attachmentTolerantUserText(chatMessageText(local))
}
