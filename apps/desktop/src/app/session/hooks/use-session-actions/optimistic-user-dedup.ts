import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import { type ChatMessage, chatMessageText, sameAttachmentTurn, textPart } from '@/lib/chat-messages'
import type { ComposerAttachment } from '@/store/composer'

import type { ClientSessionState } from '../../../types'

import { conflictingTranscriptIdentity } from './pending-turn-identity'

/**
 * The optimistic user row's identity lifecycle, extracted from the submit
 * pipeline and the transcript-refresh reconcile so each stays under its
 * complexity cap (#131848): the bubble factory, the durable-row receipt that
 * closes the id-swap race, and the acknowledged-candidate compare that keeps
 * a mid-send timeline refresh from rendering the submitted turn twice.
 */
export interface OptimisticUserBubble {
  /** Stable row id for the whole submit pipeline (drop/rewrite/receipt all key on it). */
  readonly optimisticId: string
  /** The row to append/rewrite — reflects the CURRENT display text, refs, and submitted text. */
  buildUserMessage: () => ChatMessage
  /**
   * Stamp the exact transport text prompt.submit is about to carry. The bubble
   * paints a DISPLAY projection of it (chip labels for @terminal: selections,
   * resolved refs), so the persisted twin compares equal against THIS, not
   * against the paint (#131848).
   */
  setSubmittedText: (text: string) => void
}

export function createOptimisticUserBubble(input: {
  attachments: ComposerAttachment[]
  /** Live read: ref resolution rewrites the refs array after sync. */
  attachmentRefs: () => string[]
  bubbleText: string
  submittedAt: number
}): OptimisticUserBubble {
  const optimisticId = `user-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  let submittedText: string | undefined

  return {
    optimisticId,
    buildUserMessage: () => ({
      id: optimisticId,
      role: 'user',
      parts: [
        textPart(
          input.bubbleText || (input.attachmentRefs().length ? '' : input.attachments.map(a => a.label).join(', '))
        )
      ],
      timestamp: input.submittedAt,
      attachmentRefs: input.attachmentRefs(),
      ...(submittedText !== undefined ? { submitText: submittedText } : {})
    }),
    setSubmittedText: text => {
      submittedText = text
    }
  }
}

/**
 * Bind the submit receipt's durable row id onto THIS send's optimistic
 * occurrence.
 *
 * The worker may finish before this acknowledgement arrives. Bind only this
 * send's optimistic row; never reset live state or assume the newest user row
 * still belongs to this RPC.
 */
export function bindSubmittedRowReceipt(
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState,
  sessionId: string,
  optimisticId: string,
  rowId: number
): void {
  updateSessionState(sessionId, state => {
    const index = state.messages.findIndex(message => message.id === optimisticId && message.role === 'user')

    if (index < 0 || state.messages[index].rowId === rowId) {
      return state
    }

    return {
      ...state,
      messages: state.messages.map((message, i) => (i === index ? { ...message, rowId } : message))
    }
  })
}

/**
 * Does a newly committed durable user row represent this optimistic row?
 *
 * #122079: the tolerant arms widen the TEXT compare only — they stay inside
 * the identity gate, so a rowId-bearing optimistic row is never swallowed by
 * a committed row it provably is not (a genuine repeat of the same captioned
 * paste). The rowId-less paste from #120978 carries no identity and keeps
 * matching tolerantly.
 *
 * #131848: the optimistic bubble paints a DISPLAY projection of the submitted
 * text (chip labels for @terminal: selections, resolved refs) while the
 * persisted twin stores the transport text, so the plain text compare can
 * miss and a mid-send timeline refresh re-appends the stored copy beside the
 * optimistic row. The submitText arm compares the exact text prompt.submit
 * carried — same identity gate as every other arm.
 */
export function isAcknowledgedOptimisticUser(message: ChatMessage, candidates: ChatMessage[]): boolean {
  return candidates.some(
    candidate =>
      !conflictingTranscriptIdentity(message, candidate) &&
      (textWithoutReferenceLines(chatMessageText(candidate)) ===
        textWithoutReferenceLines(chatMessageText(message)) ||
        sameAttachmentTurn(candidate, message) ||
        (message.submitText !== undefined &&
          message.submitText.trim() !== '' &&
          textWithoutReferenceLines(chatMessageText(candidate)) === textWithoutReferenceLines(message.submitText)))
  )
}
