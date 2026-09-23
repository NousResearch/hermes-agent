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

export interface SessionRowDetailsOptions {
  /** Display name of the profile that owns the session. Set only where rows
   *  from several profiles share one list (the Show-all view): the row then
   *  answers "whose chat, which model" (#89888). Null or absent means the
   *  surrounding list already says who owns the row — a single-profile scope,
   *  a profile-grouped list, or the default profile. */
  profileName?: null | string
}

const modelLabel = (model: null | string) => model?.split('/').pop()?.trim() || null
const oneLine = (value: null | string | undefined) => value?.replace(/\s+/g, ' ').trim() || null

export const sessionRowEstimate = (density: SessionListDensity) =>
  ({ compact: 28, comfortable: 45, detailed: 63 })[density]

export function sessionRowDetails(
  session: SessionInfo,
  fmt: SessionRowFormatters,
  options: SessionRowDetailsOptions = {}
): SessionRowDetails {
  const preview = oneLine(session.preview)
  const hasOwnTitle = Boolean(session.title?.trim())

  const metadata = [
    session.git_branch?.trim() || null,
    // Beside the model, not at the end: "whose chat, which model" reads in one
    // glance, and the counts keep the tail where they can be scanned together.
    oneLine(options.profileName),
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
