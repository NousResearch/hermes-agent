import type { ReactNode } from 'react'

import type { Contribution } from '@/contrib/types'

export const TRANSCRIPT_MESSAGE_AREA = 'chat.transcript-message'

/** The transcript placement currently offered to a contribution. */
export interface TranscriptMessageProps {
  kind: 'slash-result' | 'assistant-footer'
  messageId: string
  sessionId: string | null
  isLast: boolean
  /** Full slash command, including the leading slash and any arguments. Slash results only. */
  command?: string
  /** Existing plain-text slash result. Slash results only. */
  output?: string
}

export interface TranscriptMessageContribution extends Omit<Contribution, 'area' | 'data' | 'render'> {
  area: typeof TRANSCRIPT_MESSAGE_AREA
  data: {
    match: (props: TranscriptMessageProps) => boolean
  }
  render: (props: TranscriptMessageProps) => ReactNode
}
