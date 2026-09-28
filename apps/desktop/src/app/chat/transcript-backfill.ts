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

import { getOlderSessionMessages, type ProfileScope } from '@/hermes'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import {
  recordTranscriptBackfillPage,
  tailStateFromPage,
  type TranscriptProfileScope,
  transcriptTailState
} from '@/store/transcript-tail'
import type { SessionMessagesResponse } from '@/types/hermes'

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

export function graftRefreshedTailOntoBackfill(refreshedTail: ChatMessage[], previous: ChatMessage[]): ChatMessage[] {
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

  if (anchor === 0) {
    return refreshedTail
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

    return prefix.length ? [...prefix, ...refreshedTail] : refreshedTail
  }

  const refreshedIds = new Set(refreshedTail.map(message => message.id))

  // The page already contains everything on screen, including a live row the
  // tail really did cover. Take the page. This is what keeps a finished reply
  // through a long tool turn.
  if (pageCoversWindow(previous, refreshedIds, refreshedRowIds)) {
    return refreshedTail
  }

  return mergeOverlappingTail(previous, refreshedTail)
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
    request.applyOlderPage(toChatMessages(page.messages))

    return true
  })().finally(() => {
    inflightByStoredSessionId.delete(inflightKey)
  })

  inflightByStoredSessionId.set(inflightKey, run)

  return run
}
