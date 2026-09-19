/**
 * Reconciliation helpers shared by transcript hydration (wiring.tsx) and the
 * backfill graft — extracted here so the disappearing-agent-message fixes are
 * unit-testable without React.
 */
import { chatMessageText } from '@/lib/chat-messages/parts'
import type { ChatMessage } from '@/lib/chat-messages'

const normalizedTimelineText = (message: ChatMessage) => chatMessageText(message).replace(/\s+/g, ' ').trim()

/** Timestamps agree within this window: the live stream stamps occurredAt =
 *  Date.now()/1000, the stored row its persisted ts — same turn, sub-second
 *  skew, plus clock/format drift between the two writers. Generous ceiling:
 *  distinct turns in an active bot chat are seconds-to-minutes apart, never
 *  under two seconds. */
const TEMPORAL_MATCH_TOLERANCE_S = 2

const nearTimestamp = (a: number | undefined, b: number | undefined): boolean => {
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    return false
  }

  return Math.abs((a as number) - (b as number)) <= TEMPORAL_MATCH_TOLERANCE_S
}

/** The temporal fallback is only safe when text/tool matching had NOTHING to
 *  work with or genuinely could not disambiguate: a live stream is matched to
 *  its stored row by id/tool/text first, so reaching here means the texts
 *  differ. Two DISTINCT turns can still sit within the tolerance (a fast dry
 *  turn answers in <2s), so require at least one row to carry no comparable
 *  text at all — a textless/empty stream fragment — before trusting time
 *  alone. A turn with real text on both sides is never matched by clock. */
const temporalFallbackAllowed = (stored: ChatMessage, local: ChatMessage): boolean => {
  const hasRealText = (message: ChatMessage) => normalizedTimelineText(message).length > 0

  return !hasRealText(stored) || !hasRealText(local)
}

export const assistantTimelineMatch = (stored: ChatMessage, local: ChatMessage): boolean => {
  if (stored.id === local.id) {
    return true
  }

  const localToolIds = new Set(
    local.parts.filter(part => part.type === 'tool-call').map(part => (part.type === 'tool-call' ? part.toolCallId : ''))
  )

  if (localToolIds.size > 0 && stored.parts.some(part => part.type === 'tool-call' && localToolIds.has(part.toolCallId))) {
    return true
  }

  const storedText = normalizedTimelineText(stored)

  if (Boolean(storedText) && storedText === normalizedTimelineText(local)) {
    return true
  }

  // TEMPORAL FALLBACK (disappearing agent messages): a live stream whose text
  // was truncated, compressed or otherwise normalized differently from the
  // persisted row never text-matches, and the previous behavior let hydration
  // replace the live bubble with the stored row — the message "disappeared"
  // until a later rehydrate brought it back. Same turn when the timestamps
  // agree AND at least one side is textless (a text-bearing turn on both
  // sides is never matched by clock — two distinct turns can sit within the
  // tolerance, so time alone cannot carry the match there). Text/tool
  // matching still wins first.
  return (
    (nearTimestamp(stored.timestamp, local.timestamp) || nearTimestamp(stored.completedAt, local.completedAt)) &&
    temporalFallbackAllowed(stored, local)
  )
}

/**
 * graftRefreshedTailOntoBackfill with one fix: `preserveLocalAssistantErrors`
 * APPENDS preserved local rows (user prompt + failed assistant pair) AFTER the
 * merged tail. On the next graft those rows can trail the refreshed tail, so
 * the anchor lookup fails with anchor === 0 — and the old code returned the
 * bare tail, silently dropping the preserved rows (they reappear on the next
 * successful hydrate: the "messages vanish and come back" symptom). When the
 * previous transcript ends with rows the refreshed tail legitimately lacks,
 * those rows are carried back onto the graft.
 */
export function graftRefreshedTailOntoBackfill(refreshedTail: ChatMessage[], previous: ChatMessage[]): ChatMessage[] {
  if (refreshedTail.length === 0 || previous.length === 0) {
    return refreshedTail
  }

  const first = refreshedTail[0]

  const anchor = previous.findIndex(
    message =>
      (first.rowId !== undefined && message.rowId !== undefined && message.rowId === first.rowId) ||
      message.id === first.id
  )

  if (anchor < 0) {
    return refreshedTail
  }

  if (anchor === 0) {
    // anchor === 0 means previous[0] IS the refreshed tail's first row. Any
    // previous rows AFTER that point are locally-preserved rows (user prompt +
    // failed assistant) hydration does not know about yet — carry them back
    // instead of dropping them.
    return previous.length > refreshedTail.length ? [...refreshedTail, ...previous.slice(refreshedTail.length)] : refreshedTail
  }

  return [...previous.slice(0, anchor), ...refreshedTail]
}
