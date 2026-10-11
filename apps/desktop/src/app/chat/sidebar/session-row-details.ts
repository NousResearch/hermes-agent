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

/** Virtual-list placement estimate for the Inbox-style card. A full card
 *  stacks four text lines (header, title, preview, model/size) where the
 *  tallest inline density stacks three, plus the card's own padding — and a
 *  title that wraps to two lines on a narrow sidebar adds one more title
 *  line (#88473). Deliberately at or ABOVE that worst case: an oversized
 *  estimate paints a brief gap that self-measurement closes, while an
 *  undersized one paints rows over their neighbours (and the divider below)
 *  on a cold start, before any measurement can correct it. */
export const SESSION_CARD_ROW_ESTIMATE_PX = 96

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
