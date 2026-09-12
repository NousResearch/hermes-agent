import { textPart } from './parts'
import type { ChatMessage } from './types'

interface ReviewSummaryPayload {
  text?: unknown
  review_id?: unknown
  timestamp?: unknown
  row_id?: unknown
}

let liveSequence = 0

/** One projection for the gateway event and the durable REST/resume receipt. */
export function reviewSummaryMessage(payload: ReviewSummaryPayload, fallbackTime: number): ChatMessage | null {
  const text = typeof payload.text === 'string' ? payload.text.trim().replace(/^[^\p{L}\p{N}]+/u, '') : ''

  if (!text) {
    return null
  }

  const rowId =
    typeof payload.row_id === 'number' && Number.isSafeInteger(payload.row_id) && payload.row_id > 0
      ? payload.row_id
      : undefined

  const reviewId =
    typeof payload.review_id === 'string' && payload.review_id.trim()
      ? payload.review_id.trim()
      : rowId !== undefined
        ? `row-${rowId}`
        : null

  const timestamp =
    typeof payload.timestamp === 'number' && Number.isFinite(payload.timestamp) && payload.timestamp > 0
      ? payload.timestamp
      : fallbackTime

  return {
    id: reviewId ? `review-summary:${reviewId}` : `review-summary-live:${++liveSequence}`,
    role: 'system',
    parts: [textPart(`review:${text}`, timestamp)],
    timestamp,
    ...(rowId !== undefined ? { rowId } : {})
  }
}

/** A fetch started before a receipt was committed must not erase its newer live event.
 * Only carry durable rows received DURING this read. Empty/reset history and earlier rows
 * remain authoritative (do not resurrect rewound messages or grow an unbounded cache). */
export function preserveNewerReviewSummaries(
  next: ChatMessage[],
  previous: ChatMessage[],
  beforeRead: ChatMessage[]
): ChatMessage[] {
  if (!next.length || !previous.length) {
    return next
  }

  const watermark = next.reduce((max, message) => Math.max(max, message.rowId ?? 0), 0)

  if (!watermark) {
    return next
  }

  const ids = new Set([...next, ...beforeRead].map(message => message.id))

  const newer = previous.filter(
    message =>
      message.role === 'system' &&
      message.id.startsWith('review-summary:') &&
      (message.rowId ?? 0) > watermark &&
      !ids.has(message.id)
  )

  return newer.length ? [...next, ...newer] : next
}
