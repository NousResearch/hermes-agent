import type { ReactNode } from 'react'

/** Optional, message-local UI for existing transcript rows. No chat turn is created. */
export const TRANSCRIPT_MESSAGE_AREA = 'chat.transcript-message'

export type TranscriptMessageProps =
  | { kind: 'slash-result'; sessionId: string; messageId: string; isLast: boolean; command: string; output: string }
  | { kind: 'assistant-footer'; sessionId: string; messageId: string; isLast: boolean }

export interface TranscriptMessageContribution {
  /** Return true to claim this row; false leaves the host's text/controls intact. */
  match: (props: TranscriptMessageProps) => boolean
  /** Rendered inside the real message row; may return null. */
  render: (props: TranscriptMessageProps) => ReactNode
}
