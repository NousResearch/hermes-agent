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
      /** Messages released this pass. */
      releasedRows: number
      /** Backend rows those messages covered — the `transcript-tail` older-page
       *  offset is counted in backend rows, so this is what the caller rewinds
       *  it by. The hydration fold merges a turn's tool rows into one message,
       *  so this is usually larger than `releasedRows`. */
      releasedServerRows: number
    }

const NOTHING_RELEASED: TranscriptRetention = { released: false }

/**
 * Backend rows the slice covers. The hydration fold merges a turn's tool rows
 * into the assistant message they belong to, so a message is not one backend
 * row; the older-page offset is counted in backend rows, so the released rows
 * have to be converted before the offset can be rewound.
 */
function serverRowCount(messages: readonly ChatMessage[], end: number): number {
  let rows = 0

  for (let i = 0; i < end; i += 1) {
    rows += messages[i].serverRowSpan ?? 1
  }

  return rows
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

  // Nothing in flight may be released: a `pending` row has no backend row yet,
  // so it cannot be fetched back and dropping it would lose content outright.
  for (let i = 0; i < boundary; i += 1) {
    if (messages[i].pending) {
      return NOTHING_RELEASED
    }
  }

  const retained = messages.slice(boundary)

  return {
    messages: retained,
    releasedRows: boundary,
    releasedServerRows: serverRowCount(messages, boundary),
    released: true
  }
}
