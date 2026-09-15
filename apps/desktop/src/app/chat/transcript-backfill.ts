/**
 * ON-DEMAND OLDER-PAGE BACKFILL for the transcript window.
 *
 * Tail hydration (`getLatestSessionMessages`) loads only the newest page of a
 * session. "Show earlier" first pages the DOM budget, then the in-memory store
 * window — and when the whole in-memory transcript is materialized but the
 * REST hydration was truncated (`transcript-tail` bookkeeping), this module
 * fetches the next older page and prepends it to the session store.
 *
 * Offsets follow the backend's `order: 'latest'` semantics: measured back
 * from the NEWEST persisted row. Rows persisted after hydration shift that
 * origin, so a fetched page can overlap rows we already hold — the prepend
 * dedupes by durable row id (falling back to the rendered message id) and
 * the offset still advances by the fetched count, which self-corrects the
 * drift on the next page.
 */

import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import { getOlderSessionMessages } from '@/hermes'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import { recordTranscriptBackfillPage, type TranscriptProfileScope, transcriptTailState } from '@/store/transcript-tail'

/** Older rows likely exist beyond what the in-memory store holds. */
export function transcriptBackfillAvailable(
  storedSessionId: null | string | undefined,
  profile?: TranscriptProfileScope
): boolean {
  return Boolean(transcriptTailState(storedSessionId, profile)?.possiblyTruncated)
}

/**
 * Prepend an older page onto the in-memory transcript, deduplicating rows the
 * store already holds (offset drift makes overlap normal — see module doc).
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

  const existingRowIds = new Set<number>()
  const existingIds = new Set<string>()

  for (const message of existing) {
    if (message.rowId !== undefined) {
      existingRowIds.add(message.rowId)
    }

    existingIds.add(message.id)
  }

  const fresh = olderPage.filter(
    message => !(message.rowId !== undefined && existingRowIds.has(message.rowId)) && !existingIds.has(message.id)
  )

  if (fresh.length === 0) {
    return existing
  }

  return [...fresh, ...existing]
}

/**
 * The text a refreshed row is paired with its previous twin on: answer text,
 * reasoning, and a tool-call signature. Reasoning is included because a row can
 * settle with reasoning and no answer — an aborted turn — and the tool signature
 * because a tool-only row has no text at all and is itself a live row whose
 * disclosure must survive the refresh (`chatMessageText` covers answer parts only).
 */
function textForPairing(message: ChatMessage): string {
  const text = message.parts
    .map(part => {
      if (part.type === 'text' || part.type === 'reasoning') {
        return part.text
      }

      // The tool name is enough to identify a tool-only row against its own stored
      // twin, because the live turn is matched newest first: two turns that call the
      // same tool cannot trade identities when only the newer is still live.
      return part.type === 'tool-call' ? `[tool:${part.toolName}]` : ''
    })
    .join('')

  return textWithoutReferenceLines(text)
}

/**
 * The previous row a refreshed row represents, newest first, or `-1` for none.
 *
 * Only a row whose id is about to change can hand an identity on, so a candidate
 * that is already persisted under a DIFFERENT row id is a different message that
 * happens to read the same — two turns that both open "continue" — and claiming it
 * would leave two rows sharing one identity. A candidate under the same id is the
 * same row already committed: it keys on that id either way, so there is nothing
 * to carry. Newest first, so a repeated phrase pairs with the live turn rather
 * than an older one.
 */
function lastIndexOfPair(
  previous: ChatMessage[],
  message: ChatMessage,
  text: string,
  claimed: Set<number>
): number {
  for (let index = previous.length - 1; index >= 0; index -= 1) {
    if (claimed.has(index)) {
      continue
    }

    const candidate = previous[index]

    if (candidate.role !== message.role || candidate.id === message.id) {
      continue
    }

    if (candidate.rowId !== undefined) {
      // Persisted, so it settles the question on its own: the same row id is the
      // same row, and a different one is a different message that merely reads alike.
      if (candidate.rowId === message.rowId) {
        return index
      }

      continue
    }

    // Unpersisted, so it pairs on what it reads as. A streamed row is a prefix of
    // its stored reply, never the reverse — allowing the reverse would let a short
    // prompt ("ok", "continue") claim a longer older row that merely starts with it.
    const candidateText = textForPairing(candidate)

    if (candidateText !== '' && (candidateText === text || text.startsWith(candidateText))) {
      return index
    }
  }

  return -1
}

/**
 * Carry each refreshed row's render identity (`ChatMessage.rowKey`) across the
 * refresh.
 *
 * A row's id is not stable: a turn's rows are born under an optimistic `user-*`
 * / `assistant-stream-*` id and only acquire their committed one when the stored
 * transcript is re-read. The row element is keyed on the identity it was born
 * with, so a refresh that hands the list a fresh object carrying the committed id
 * re-keys the row and React remounts its subtree — the thinking preview the
 * reader is watching snaps shut to its header (duration lost), a preview they
 * closed reopens, and every other row-local disclosure in the turn resets. This
 * refresh is the path that rewrite travels: `hydrateFromStoredSession` reads the
 * stored tail, converts it with `toChatMessages`, and grafts it here.
 *
 * Pair a refreshed row with the previous row it represents — same role, and the
 * same text once directive lines are dropped, or one a prefix of the other,
 * since a streamed row is a prefix of its stored reply — and let it inherit the
 * identity that row was rendering under. Paired by content rather than by slot:
 * the refreshed tail is a single page while `previous` may hold a longer
 * backfilled prefix, so the arrays are rarely index-aligned.
 *
 * Returns `refreshedTail` untouched when nothing needed carrying, so the common
 * case keeps its array and object identity.
 */
function carryRowKeysOntoRefreshedTail(refreshedTail: ChatMessage[], previous: ChatMessage[]): ChatMessage[] {
  if (previous.length === 0) {
    return refreshedTail
  }

  const claimed = new Set<number>()
  let changed = false

  const carried = refreshedTail.map(message => {
    if (message.rowKey !== undefined) {
      return message
    }

    const index = lastIndexOfPair(previous, message, textForPairing(message), claimed)

    if (index < 0) {
      return message
    }

    claimed.add(index)

    const source = previous[index]

    changed = true

    return { ...message, rowKey: source.rowKey ?? source.id }
  })

  return changed ? carried : refreshedTail
}

/**
 * Re-anchor a refreshed TAIL onto a transcript that has backfilled older
 * pages. Background refreshes and post-turn rehydrates re-read only the
 * newest page; replacing the store with that page outright would silently
 * drop everything "Show earlier" already loaded. Find where the refreshed
 * tail begins inside the previous transcript and keep the older prefix.
 * When no anchor is found (compaction rewrite, different session), the
 * refreshed tail is authoritative — same behavior as before backfill existed.
 *
 * The refreshed rows keep the render identity they were born with, so adopting
 * the stored tail does not re-key (and remount) the rows it replaces — see
 * `carryRowKeysOntoRefreshedTail`.
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

  if (anchor <= 0) {
    return carryRowKeysOntoRefreshedTail(refreshedTail, previous)
  }

  // Only the rows this refresh replaces can hand an identity on. Matching against
  // the retained prefix would let a refreshed row take an identity that is still on
  // screen, and two rows sharing one key is worse than a remount.
  return [
    ...previous.slice(0, anchor),
    ...carryRowKeysOntoRefreshedTail(refreshedTail, previous.slice(anchor))
  ]
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

    // Session switched while the page was in flight: discard it entirely.
    // The bookkeeping stays untouched so a later re-visit (which re-records
    // the tail on hydration anyway) starts from consistent state.
    if (!request.isCurrent()) {
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
