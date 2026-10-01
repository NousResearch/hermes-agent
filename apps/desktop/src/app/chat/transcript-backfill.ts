/**
 * ON-DEMAND OLDER-PAGE BACKFILL for the transcript window.
 *
 * Tail hydration (`getLatestSessionMessages`) loads only the newest page of a
 * session. "Show earlier" first pages the DOM budget, then the in-memory store
 * window — and when the whole in-memory transcript is materialized but the
 * REST hydration was truncated (`transcript-tail` bookkeeping), this module
 * fetches the next older page and merges it into the session store.
 *
 * Offsets follow the backend's `order: 'latest'` semantics: measured back
 * from the NEWEST persisted row. Rows persisted after hydration shift that
 * origin, so a fetched page can overlap rows we already hold and even extend
 * past the cached tail. Shared durable rows anchor the merge on either side;
 * the offset still advances by the fetched count, which self-corrects the
 * drift on the next page.
 */

import { getOlderSessionMessages, getSessionMessages, type ProfileScope } from '@/hermes'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import {
  recordTranscriptBackfillPage,
  tailStateFromPage,
  type TranscriptProfileScope,
  transcriptTailState
} from '@/store/transcript-tail'
import type { SessionMessage, SessionMessagesResponse } from '@/types/hermes'

/**
 * Compaction projection can stamp the session's opening USER row
 * `display_kind: hidden` (#96875): the durable store holds the greeting but
 * hydration drops the row, so paging to the very top of a compacted session
 * renders only the first assistant reply. When an older page carries the
 * opening user turn, clear the hidden stamp (keeping the content as the
 * display projection) so `toChatMessages` keeps it on screen. Only the
 * FIRST user row with content is unhidden — later hidden rows stay hidden
 * (model scaffolding, muted turns), and an opening row with no content
 * was never a greeting.
 */
export function unhideOpeningUserRows(messages: SessionMessage[]): SessionMessage[] {
  const openingIndex = messages.findIndex(message => message.role === 'user')

  if (openingIndex < 0) {
    return messages
  }

  const opening = messages[openingIndex]

  if (opening.display_kind !== 'hidden') {
    return messages
  }

  const content = opening.display_content ?? opening.content

  if (content == null || content === '') {
    return messages
  }

  return messages.map((message, index) => {
    if (index !== openingIndex) {
      return message
    }

    const { display_kind: _hidden, ...rest } = message

    return {
      ...rest,
      display_content: typeof content === 'string' ? content : String(content)
    }
  })
}

/** Older rows likely exist beyond what the in-memory store holds. */
export function transcriptBackfillAvailable(
  storedSessionId: null | string | undefined,
  profile?: TranscriptProfileScope
): boolean {
  return Boolean(transcriptTailState(storedSessionId, profile)?.possiblyTruncated)
}

/**
 * Merge a fetched page into the in-memory transcript, deduplicating rows
 * the store already holds (offset drift makes overlap normal — see module doc).
 * A page with no shared row is presumed older; overlapping pages use their
 * shared rows to place fresh messages before, within, or after the cached tail.
 * Preserves reference identity when nothing changes: handing React a fresh
 * array of the same messages re-renders the runtime for nothing.
 */
export function mergeOlderTranscriptPage(existing: ChatMessage[], olderPage: ChatMessage[]): ChatMessage[] {
  // Backfill only makes sense under an already-hydrated tail. An empty store
  // here means the session was swapped or wiped mid-fetch; prepending would
  // paint the older page as the whole conversation.
  if (existing.length === 0 || olderPage.length === 0) {
    return existing
  }

  const existingRowIndices = new Map<number, number>()
  const existingIdIndices = new Map<string, number>()

  existing.forEach((message, index) => {
    if (message.rowId !== undefined) {
      existingRowIndices.set(message.rowId, index)
    }

    existingIdIndices.set(message.id, index)
  })

  // The offset counts backwards from the newest durable row. While a long
  // turn persists, an "older" page can overlap the cached tail AND extend
  // beyond its end. Position fresh rows by the shared anchors, not by the
  // page's requested direction.
  const insertions = new Map<number, ChatMessage[]>()
  let pending: ChatMessage[] = []
  let lastAnchor = -1

  for (const message of olderPage) {
    const anchor =
      (message.rowId !== undefined ? existingRowIndices.get(message.rowId) : undefined) ??
      existingIdIndices.get(message.id)

    if (anchor === undefined) {
      pending.push(message)

      continue
    }

    if (pending.length) {
      insertions.set(anchor, [...(insertions.get(anchor) ?? []), ...pending])
      pending = []
    }

    lastAnchor = anchor
  }

  if (pending.length) {
    const position = lastAnchor < 0 ? 0 : lastAnchor + 1
    insertions.set(position, [...(insertions.get(position) ?? []), ...pending])
  }

  if (insertions.size === 0) {
    return existing
  }

  const merged: ChatMessage[] = []

  for (let index = 0; index <= existing.length; index++) {
    const additions = insertions.get(index)

    if (additions) {
      merged.push(...additions)
    }

    if (index < existing.length) {
      merged.push(existing[index])
    }
  }

  return merged
}

/**
 * Re-anchor a refreshed TAIL onto a transcript that has backfilled older
 * pages. Background refreshes and post-turn rehydrates re-read only the
 * newest page; replacing the store with that page outright would silently
 * drop everything "Show earlier" already loaded. Find where the refreshed
 * tail begins inside the previous transcript and keep the older prefix when
 * that prefix is actually earlier. An anchor on the first on-screen row is a
 * real match: the page replaces the window from there. A page that already
 * contains every on-screen row replaces the window. When the page overlaps
 * the screen but that splice would put an older stored id after a newer one,
 * merge by stored id. The fresh page wins where both sides share an id, and
 * a row with no stored id stays at the end. A page that shares no stored id
 * is the transcript now on screen — a compaction rewrite or a different
 * session — and replaces the window.
 */
function pageCoversWindow(previous: ChatMessage[], refreshedIds: Set<string>, refreshedRowIds: Set<number>): boolean {
  return previous.every(
    message => (message.rowId !== undefined && refreshedRowIds.has(message.rowId)) || refreshedIds.has(message.id)
  )
}

function durableRowIds(messages: ChatMessage[]): Set<number> {
  return new Set(messages.flatMap(message => (message.rowId === undefined ? [] : [message.rowId])))
}

/** A text-only refresh can omit the live tool bubble after the turn settles. */
function retainCompletedTurnTools(messages: ChatMessage[], previous: ChatMessage[]): ChatMessage[] {
  const previousFinalIndex = previous.findLastIndex(
    message => message.role === 'assistant' && message.rowId !== undefined && Boolean(chatMessageText(message).trim())
  )

  if (previousFinalIndex < 0) {
    return messages
  }

  const previousFinal = previous[previousFinalIndex]

  const finalIndex = messages.findIndex(
    message => message.role === 'assistant' && message.rowId === previousFinal.rowId
  )

  if (finalIndex < 0 || chatMessageText(messages[finalIndex]).trim() !== chatMessageText(previousFinal).trim()) {
    return messages
  }

  const previousUserIndex = previous.findLastIndex(
    (message, index) => index < previousFinalIndex && message.role === 'user'
  )

  const userIndex = messages.findLastIndex((message, index) => index < finalIndex && message.role === 'user')

  if (
    previousUserIndex < 0 ||
    userIndex < 0 ||
    previous[previousUserIndex].rowId === undefined ||
    previous[previousUserIndex].rowId !== messages[userIndex].rowId
  ) {
    return messages
  }

  const existingIds = new Set(
    messages
      .slice(userIndex + 1, finalIndex + 1)
      .flatMap(message => message.parts.flatMap(part => (part.type === 'tool-call' ? [part.toolCallId] : [])))
  )

  const missing = previous.slice(previousUserIndex + 1, previousFinalIndex + 1).flatMap(message =>
    message.parts.filter(part => {
      if (part.type !== 'tool-call' || (part.result === undefined && part.completedAt === undefined)) {
        return false
      }

      if (existingIds.has(part.toolCallId)) {
        return false
      }

      existingIds.add(part.toolCallId)

      return true
    })
  )

  if (!missing.length) {
    return messages
  }

  return messages.map((message, index) =>
    index === finalIndex ? { ...message, parts: [...missing, ...message.parts] } : message
  )
}

function sharesDurableRow(first: ChatMessage[], second: ChatMessage[]): boolean {
  const rowIds = durableRowIds(first)

  return second.some(message => message.rowId !== undefined && rowIds.has(message.rowId))
}

interface StoredRowSlot {
  message: ChatMessage
  /** Rows without a stored id (e.g. a page-local tool fold) that precede this row. */
  leading: ChatMessage[]
}

/**
 * Stored-id merge for a page that overlaps the window but does not anchor in
 * front of it. A row with no stored id travels with the next stored row after
 * it, so a page-local fold stays in front of the row it preceded. Rows with no
 * stored id after the last stored row stay at the end (page first, then live).
 */
function mergeOverlappingTail(previous: ChatMessage[], refreshedTail: ChatMessage[]): ChatMessage[] {
  // Compaction, rewind, or a different session arrives as new stored ids.
  // This function is the path that puts that page on screen.
  if (!sharesDurableRow(previous, refreshedTail)) {
    return refreshedTail
  }

  const refreshedIds = new Set(refreshedTail.map(message => message.id))
  const byRowId = new Map<number, StoredRowSlot>()

  const place = (messages: ChatMessage[], fresh: boolean): ChatMessage[] => {
    let pending: ChatMessage[] = []

    for (const message of messages) {
      if (message.rowId === undefined) {
        // The fresh page's copy of an unstored row wins over the window's.
        if (fresh || !refreshedIds.has(message.id)) {
          pending.push(message)
        }

        continue
      }

      const existing = byRowId.get(message.rowId)

      if (!fresh && existing) {
        pending = []

        continue
      }

      // The fresh page replaces the row; keep the window's leading rows when
      // the page brought none of its own for it.
      const leading = fresh && existing && pending.length === 0 ? existing.leading : pending
      byRowId.set(message.rowId, { message, leading })
      pending = []
    }

    return pending
  }

  const previousTrailing = place(previous, false)
  const refreshedTrailing = place(refreshedTail, true)

  const stored = [...byRowId.entries()]
    .sort((left, right) => left[0] - right[0])
    .flatMap(([, { leading, message }]) => [...leading, message])

  return [...stored, ...refreshedTrailing, ...previousTrailing]
}

export function graftRefreshedTailOntoBackfill(
  refreshedTail: ChatMessage[],
  previous: ChatMessage[],
  options: { pageNewerThanRendered?: boolean } = {}
): ChatMessage[] {
  if (refreshedTail.length === 0 || previous.length === 0) {
    return refreshedTail
  }

  // The first rendered message can be a page-local tool fold whose id is not
  // durable. Anchor on the first shared persisted row anywhere in the page.
  const refreshedRowIds = durableRowIds(refreshedTail)

  const firstDurable = refreshedTail.find(message => message.rowId !== undefined)

  const anchor = firstDurable === undefined ? -1 : previous.findIndex(message => message.rowId === firstDurable.rowId)

  const anchorRowId = firstDurable?.rowId

  // A hit on the first row, or a hit after a prefix whose stored ids are all
  // earlier, is the backfill anchor. A hit further down on a row that was
  // glued on late is not: the prefix is newer than the match.
  const prefixIsEarlier =
    anchor > 0 &&
    anchorRowId !== undefined &&
    previous.slice(0, anchor).every(message => message.rowId === undefined || message.rowId < anchorRowId)

  const grafted = (() => {
    if (anchor === 0) {
      return retainCompletedTurnTools(refreshedTail, previous)
    }

    if (prefixIsEarlier) {
      // A page-local tool fold can sit in front of the anchor on BOTH sides: the
      // window's copy was hydrated from the same page, and the refreshed page
      // re-emits that row with the same id. Keeping both copies makes the graft
      // non-idempotent — the refreshed window comes back one row longer than the
      // local window on every read, so `messagesIfTranscriptBehind` reports
      // "behind" forever: the send is refused before `prompt.submit` runs and a
      // duplicate accumulates per retry. Drop only the prefix copies the page
      // already carries. Every durable prefix row keeps travelling in front of
      // the refreshed tail, and so does an unstored row the page has no copy of
      // (it can only have come from an older page).
      const refreshedIds = new Set(refreshedTail.map(message => message.id))

      const prefix = previous
        .slice(0, anchor)
        .filter(message => message.rowId !== undefined || !refreshedIds.has(message.id))

      return retainCompletedTurnTools(prefix.length ? [...prefix, ...refreshedTail] : refreshedTail, previous)
    }

    const refreshedIds = new Set(refreshedTail.map(message => message.id))

    // The page already contains everything on screen, including a live row the
    // tail really did cover. Take the page. This is what keeps a finished reply
    // through a long tool turn.
    if (pageCoversWindow(previous, refreshedIds, refreshedRowIds)) {
      return retainCompletedTurnTools(refreshedTail, previous)
    }

    return retainCompletedTurnTools(mergeOverlappingTail(previous, refreshedTail), previous)
  })()

  // A page read before the rendered transcript's generation that omits its
  // newer durable rows is stale (#119819): retain those rows instead of
  // letting the graft drop them. After a rewind the page is the authority —
  // see retainNewerRowsOverStalePage.
  return retainNewerRowsOverStalePage(grafted, previous, options.pageNewerThanRendered)
}

const REFRESH_OVERLAP_PAGE_LIMIT = 4

/**
 * Reader for the pages older than a refreshed newest page. Paging follows the
 * transcript-tail rules: it starts at the page's own offset and stops once a
 * page comes back short, or without pagination metadata (a legacy backend
 * already returned everything).
 */
export function olderPageReader(
  storedSessionId: string,
  scope: ProfileScope,
  page: null | Pick<SessionMessagesResponse, 'messages' | 'pagination'> | undefined
): () => Promise<ChatMessage[]> {
  let state = page ? tailStateFromPage(page) : undefined

  return async () => {
    if (!state?.possiblyTruncated) {
      return []
    }

    const older = await getOlderSessionMessages(storedSessionId, scope, state.nextOffset)
    state = tailStateFromPage(older)

    return toChatMessages(older.messages)
  }
}

/**
 * A refresh begins at the newest persisted row. A tool-heavy turn can fill
 * that page entirely, putting its first durable row after the rendered
 * transcript. Read a small, bounded number of older pages until one shares a
 * durable row, so graftRefreshedTailOntoBackfill can retain the live prefix.
 * Stored ids only grow, so once a page reaches below the oldest rendered id
 * no older page can overlap.
 */
export async function extendRefreshPageToOverlap(
  refreshedTail: ChatMessage[],
  previous: ChatMessage[],
  readOlderPage: () => Promise<ChatMessage[]>
): Promise<ChatMessage[]> {
  if (!refreshedTail.length || !previous.length) {
    return refreshedTail
  }

  const previousRowIds = durableRowIds(previous)

  // Streamed or optimistic rows carry no stored id: nothing can overlap.
  if (previousRowIds.size === 0) {
    return refreshedTail
  }

  const sharesPrevious = (messages: ChatMessage[]) =>
    messages.some(message => message.rowId !== undefined && previousRowIds.has(message.rowId))

  if (sharesPrevious(refreshedTail)) {
    return refreshedTail
  }

  const oldestPrevious = Math.min(...previousRowIds)
  let extended = refreshedTail

  for (let page = 0; page < REFRESH_OVERLAP_PAGE_LIMIT; page += 1) {
    let older: ChatMessage[]

    try {
      older = await readOlderPage()
    } catch {
      // A refresh failure must retain today's newest-page behavior.
      return refreshedTail
    }

    if (!older.length) {
      return refreshedTail
    }

    extended = [...older, ...extended]

    if (sharesPrevious(older)) {
      return extended
    }

    if (older.some(message => message.rowId !== undefined && message.rowId < oldestPrevious)) {
      return refreshedTail
    }
  }

  return refreshedTail
}

/**
 * A stale or partial REST response can start at an older page and omit the
 * durable tail already rendered (an active tile, a slow-poll background
 * refresh). Row ids are monotonic in stored history, so when the page's
 * high-water mark sits strictly below the rendered transcript's, retain the
 * rendered durable rows the page never covered instead of letting the graft
 * clobber them.
 *
 * Not every omission is staleness. The comparison is against the page's
 * COVERED row set — the backend rows the folded messages stand for — not
 * against the merged messages' leading rowId. The hydration fold merges a
 * tool-interleaved turn's rows (interim assistant, tool calls, tool
 * results, final answer) into ONE message keyed on the turn's FIRST row id,
 * so a page that legitimately folds rendered rows into fewer durable
 * messages reports a lower leading id while still covering them all: it
 * supersedes those rows (same-generation fold), it is not a stale page.
 * A rendered row only survives retention when its row id is strictly
 * GREATER than every backend row the page covered — the true high-water
 * mark, derivable by expanding each message's `serverRowSpan` from its
 * leading rowId.
 *
 * The generation gate: a rewind (`/undo`, `/retry`, truncation) soft-deletes
 * rows and DROPS the high-water mark with no signal a row-count comparison
 * can distinguish from a stale page — the rendered tail is legitimately
 * "newer than" the refreshed page because the user removed it. The caller
 * therefore only treats a below-water page as stale when the transcript it
 * would clobber was read at the SAME rewind generation as the page (no undo
 * happened in between); after a rewind the page is the authority and the
 * removed rows must stay gone.
 */
export function retainNewerRowsOverStalePage(
  merged: ChatMessage[],
  previous: ChatMessage[],
  pageNewerThanRendered?: boolean
): ChatMessage[] {
  if (pageNewerThanRendered) {
    return merged
  }

  /** Highest backend row the page/graft covers — folded rows count fully. */
  const coveredRowHighWater = (messages: ChatMessage[]): number =>
    messages.reduce(
      (newest, message) =>
        message.rowId === undefined ? newest : Math.max(newest, message.rowId + (message.serverRowSpan ?? 1) - 1),
      0
    )

  const newestMergedRowId = coveredRowHighWater(merged)
  const newestPreviousRowId = coveredRowHighWater(previous)

  if (!newestMergedRowId || newestMergedRowId >= newestPreviousRowId) {
    return merged
  }

  const mergedRowIds = new Set(
    merged.map(message => message.rowId).filter((rowId): rowId is number => rowId !== undefined)
  )

  // Rows the page superseded (its fold covers them) are not "newer rendered
  // rows the page never covered" — a fold is not a stale page.
  const mergedCoveredRows = new Set<number>()

  for (const message of merged) {
    if (message.rowId !== undefined) {
      for (let row = message.rowId; row < message.rowId + (message.serverRowSpan ?? 1); row += 1) {
        mergedCoveredRows.add(row)
      }
    }
  }

  const missingNewerRows = previous.filter(
    message =>
      message.rowId !== undefined &&
      message.rowId > newestMergedRowId &&
      !mergedRowIds.has(message.rowId) &&
      !mergedCoveredRows.has(message.rowId)
  )

  return missingNewerRows.length ? [...merged, ...missingNewerRows] : merged
}

export interface BackfillRequest {
  /** Durable stored session id — the tail bookkeeping key. */
  storedSessionId: string
  /** Owner scope captured when the tail was hydrated. */
  profile?: TranscriptProfileScope
  /** Stale-response guard: called after the fetch resolves; when it reports
   *  false (the user switched sessions mid-flight) the page is discarded and
   *  the bookkeeping is left untouched, mirroring the isCurrentResume()
   *  pattern in use-session-actions. */
  isCurrent: () => boolean
  /** Apply the converted older page to the session's message store. The
   *  callback owns WHERE the messages live (session-state cache vs the global
   *  draft atom) and must merge via `mergeOlderTranscriptPage`. */
  applyOlderPage: (olderPage: ChatMessage[]) => void
}

// One fetch per stored session at a time. Keyed by stored id (not runtime id)
// so a mid-fetch runtime rebind cannot double-fetch the same page.
const inflightByStoredSessionId = new Map<string, Promise<boolean>>()

/** Test-only: drop in-flight guards between cases. */
export function _resetTranscriptBackfillForTests(): void {
  inflightByStoredSessionId.clear()
}

/**
 * Fetch the next older page for a session and prepend it via
 * `applyOlderPage`. Resolves true when a page was applied. Concurrent calls
 * for the same session share one fetch.
 */
export function backfillOlderTranscriptPage(request: BackfillRequest): Promise<boolean> {
  const { profile, storedSessionId } = request
  const inflightKey = JSON.stringify([profile || null, storedSessionId])
  const inflight = inflightByStoredSessionId.get(inflightKey)

  if (inflight) {
    return inflight
  }

  const run = (async () => {
    const tail = transcriptTailState(storedSessionId, profile)

    if (!tail?.possiblyTruncated) {
      return false
    }

    let page

    try {
      page = await getOlderSessionMessages(storedSessionId, tail.profile, tail.nextOffset)
    } catch {
      // Non-fatal: the action stays available and the next click retries.
      return false
    }

    // A route can stay put while rewind or revalidation replaces its tail.
    // This page belongs to the exact tail generation we fetched against, not
    // merely the same stored id. Never graft it onto a newer display history.
    if (!request.isCurrent() || transcriptTailState(storedSessionId, profile) !== tail) {
      return false
    }

    // A response without pagination metadata is a legacy backend that ignored
    // the paging query and returned the FULL transcript one-shot. The merge
    // below prepends whatever prefix the store is missing, and the recorded
    // state marks the session fully loaded so the REST action retires.
    recordTranscriptBackfillPage(storedSessionId, page, profile)
    const olderRows = unhideOpeningUserRows(page.messages)
    request.applyOlderPage(toChatMessages(olderRows))

    // #96875: paging can reach the top while the opening USER turn is still
    // missing — the durable row is compaction-projected to display_kind=hidden
    // (dropped at hydration) or sits before the last reachable `latest` page.
    // Once the tail bookkeeping reports the session fully loaded and the page
    // that landed carries no user turn, fetch the oldest display rows once and
    // prepend the opening user turn. Best-effort: a failure keeps the older
    // page that already landed.
    const openingTurnLoaded = olderRows.some(message => message.role === 'user' && message.display_kind !== 'hidden')

    if (!openingTurnLoaded && !transcriptTailState(storedSessionId, profile)?.possiblyTruncated) {
      try {
        const origin = await getSessionMessages(storedSessionId, tail.profile, {
          includeCompacted: true,
          limit: 20,
          offset: 0,
          order: 'oldest'
        })

        if (request.isCurrent()) {
          request.applyOlderPage(toChatMessages(unhideOpeningUserRows(origin.messages)))
        }
      } catch {
        // Origin fetch is best-effort; the already-applied older page stays.
      }
    }

    return true
  })().finally(() => {
    inflightByStoredSessionId.delete(inflightKey)
  })

  inflightByStoredSessionId.set(inflightKey, run)

  return run
}
