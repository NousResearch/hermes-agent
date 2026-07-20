import type { SessionInfo } from '@/types/hermes'

export interface SessionRowDetails {
  metadata: string
  preview: null | string
}

const countLabel = (count: number, singular: string, plural = `${singular}s`) =>
  `${count} ${count === 1 ? singular : plural}`

const modelLabel = (model: null | string) => model?.split('/').pop()?.trim() || null
const oneLine = (value: null | string) => value?.replace(/\s+/g, ' ').trim() || null

export const sessionRowEstimate = (density: 'compact' | 'comfortable' | 'detailed') =>
  ({ compact: 28, comfortable: 45, detailed: 63 })[density]

export function sessionRowDetails(session: SessionInfo): SessionRowDetails {
  const preview = oneLine(session.preview)
  const hasOwnTitle = Boolean(session.title?.trim())

  const metadata = [
    session.git_branch?.trim() || null,
    modelLabel(session.model),
    countLabel(session.message_count, 'message'),
    countLabel(session.tool_call_count, 'tool call')
  ]
    .filter(Boolean)
    .join(' · ')

  return {
    metadata,
    preview: hasOwnTitle ? preview : null
  }
}
