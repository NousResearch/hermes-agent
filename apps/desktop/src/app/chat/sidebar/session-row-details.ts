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

/** Shared contract for rendered minimum height and virtual-list placement. */
export const SESSION_ROW_HEIGHT_PX: Record<SessionListDensity, number> = {
  compact: 28,
  comfortable: 48,
  detailed: 64
}

export const sessionRowEstimate = (density: SessionListDensity) => SESSION_ROW_HEIGHT_PX[density]

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
