/**
 * BOUND THE RETAINED TRANSCRIPT — release paged-through history from the store.
 *
 * `transcript-window` bounds what REACHES assistant-ui, but the session store
 * underneath it keeps every message it ever materialized: the tail hydration,
 * every page "Show earlier" fetched, and every turn this window streamed. A
 * long-lived window therefore grows with session content forever — measured as
 * the renderer's dominant footprint on #77311 — and the payload is what costs:
 * `ChatMessage.parts` carries the rendered tool output, file previews, diffs and
 * images for each of those rows.
 *
 * The rows in question are already off the user's screen (older than the
 * window's own start) and already on the backend (they are persisted with a
 * `rowId`), so holding them is pure duplication. This module decides the cut:
 * keep the live window plus one weight page of slack so a single "Show earlier"
 * still pages through memory, and release everything older. Re-hydration is the
 * existing older-page backfill — the caller rewinds the session's
 * `transcript-tail` bookkeeping so `expandWindow` fetches the released rows
 * again instead of treating the transcript as fully materialized
 * (see app/chat/transcript-backfill).
 *
 * Invariants:
 * - Never inside the window: the cut starts at the window's first message.
 * - Never splits an assistant branch group (same reason the window cut does
 *   not: a group without its fork point is re-parented).
 * - Never drops a row without a durable `rowId` — an unpersisted row cannot be
 *   fetched back, so releasing it would lose content for good.
 * - Reports `released: false` and does no work (no weight walk, no row count)
 *   when there is nothing to release, so a re-cut of an untouched transcript
 *   stays cheap.
 */

import type { ChatMessage } from '@/lib/chat-messages'
import { messageStoreWeight } from '@/lib/render-weight'

import { alignToBranchGroup, TRANSCRIPT_WINDOW_BUDGET } from './transcript-window'

/**
 * Weight of already-paged-through history kept behind the live window: one
 * window page. Enough that the first "Show earlier" of a session pages through
 * memory (and the backfill it may then start is prefetched by the same click),
 * while the retained payload stays proportional to the window instead of to the
 * session's length.
 *
 * Distinct from `TRANSCRIPT_WINDOW_SLACK`, which is the re-cut hysteresis.
 */
export const TRANSCRIPT_RETAIN_BUDGET = TRANSCRIPT_WINDOW_BUDGET

export type TranscriptRetention =
  | { released: false }
  | {
      released: true
      /** Transcript to keep in the store; everything before it was released. */
      messages: ChatMessage[]
      /** Rows released this pass. */
      releasedRows: number
      /** Persisted rows still retained, counted from the newest. This is the
       *  `transcript-tail` offset the next older REST page starts at — the
       *  backend pages by persisted rows, so it is counted in the same
       *  currency. */
      retainedPersistedRows: number
    }

const NOTHING_RELEASED: TranscriptRetention = { released: false }

function persistedRowCount(messages: readonly ChatMessage[]): number {
  let count = 0

  for (const message of messages) {
    if (message.rowId !== undefined) {
      count += 1
    }
  }

  return count
}

/**
 * How much of `messages` the store must keep, given the live window's first
 * message. Pure: the caller owns applying it (store write + tail rewind).
 */
export function boundRetainedTranscript(
  messages: readonly ChatMessage[],
  windowAnchorId: null | string
): TranscriptRetention {
  if (windowAnchorId === null || messages.length === 0) {
    return NOTHING_RELEASED
  }

  const anchor = messages.findIndex(message => message.id === windowAnchorId)

  // The window already starts at the oldest row the store holds.
  if (anchor <= 0) {
    return NOTHING_RELEASED
  }

  // Spend the slack walking back from the window, so paging history stays a
  // memory read for one more page.
  let boundary = anchor
  let slackLeft = TRANSCRIPT_RETAIN_BUDGET

  while (boundary > 0 && slackLeft > 0) {
    boundary -= 1
    slackLeft -= messageStoreWeight(messages[boundary].parts)
  }

  boundary = alignToBranchGroup(messages, boundary)

  if (boundary <= 0) {
    return NOTHING_RELEASED
  }

  // A released row must be fetchable again, so every row being released has to
  // carry its durable id. A prefix holding a never-persisted row (an in-memory
  // row the backend never wrote) is left whole rather than released in part:
  // the older-page fetch is a row offset from the newest, and a prefix that
  // cannot be re-fetched in full would leave a hole in the middle.
  for (let i = 0; i < boundary; i += 1) {
    if (messages[i].rowId === undefined) {
      return NOTHING_RELEASED
    }
  }

  const retained = messages.slice(boundary)
  const retainedPersistedRows = persistedRowCount(retained)

  // Offset bookkeeping is measured from persisted rows. Without one in the
  // retained slice there is no anchor to re-fetch against — keep everything.
  if (retainedPersistedRows === 0) {
    return NOTHING_RELEASED
  }

  return { messages: retained, releasedRows: boundary, released: true, retainedPersistedRows }
}
