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

// Two 13px title lines keep the shared 1.35 leading (including Windows glyph
// clearance), plus 4px body padding. Metadata adds 13.5px + 2px margin;
// detailed preview adds another 13.5px + 4px margin. Round up until the
// virtualizer measures each row so unmeasured titles cannot overlap.
export const sessionRowEstimate = (density: SessionListDensity) =>
  ({ compact: 40, comfortable: 55, detailed: 73 })[density]

/** Virtual-list placement estimate for the Inbox-style card. A full card
 *  stacks four text lines (header, title, preview, model/size) where the
 *  tallest inline density stacks three, plus the card's own padding — and a
 *  title that wraps to two lines on a narrow sidebar adds one more title
 *  line (#88473). Deliberately at or ABOVE that worst case: an oversized
 *  estimate paints a brief gap that self-measurement closes, while an
 *  undersized one paints rows over their neighbours (and the divider below)
 *  on a cold start, before any measurement can correct it. */
export const SESSION_CARD_ROW_ESTIMATE_PX = 110

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
