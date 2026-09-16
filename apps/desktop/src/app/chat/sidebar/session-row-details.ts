import type { SessionListDensity } from '@/store/session-list-density'
import type { SessionInfo } from '@/types/hermes'

export interface SessionRowDetails {
  metadata: string
  preview: null | string
}

export interface SessionRowFormatters {
  messageCount: (count: number) => string
  toolCallCount: (count: number) => string
}

const modelLabel = (model: null | string) => model?.split('/').pop()?.trim() || null
const oneLine = (value: null | string) => value?.replace(/\s+/g, ' ').trim() || null

// Compact rows can show a two-line, 13px `leading-none` title plus the body's
// 4px vertical padding: 30px. Comfortable and detailed rows add 13.5px text
// lines plus margins, so their estimates round the 45.5px and 63px content
// floors up to 46px and 64px. Keep the virtualizer at those floors until each
// row has been observed; underestimating makes long lists jump when their
// actual heights are measured.
export const sessionRowEstimate = (density: SessionListDensity) =>
  ({ compact: 30, comfortable: 46, detailed: 64 })[density]

export function sessionRowDetails(session: SessionInfo, fmt: SessionRowFormatters): SessionRowDetails {
  const preview = oneLine(session.preview)
  const hasOwnTitle = Boolean(session.title?.trim())

  const metadata = [
    session.git_branch?.trim() || null,
    modelLabel(session.model),
    session.message_count > 0 ? fmt.messageCount(session.message_count) : null,
    session.tool_call_count > 0 ? fmt.toolCallCount(session.tool_call_count) : null
  ]
    .filter(Boolean)
    .join(' · ')

  return {
    metadata,
    preview: hasOwnTitle ? preview : null
  }
}
