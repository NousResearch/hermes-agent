/**
 * REST TAIL-HYDRATION BOOKKEEPING — keyed by STORED session id.
 *
 * `getLatestSessionMessages` loads a small newest-first page instead of a
 * fixed 500-row transcript. When that page comes back full (returned ===
 * limit), older rows likely exist on the backend; this store records that
 * fact plus the offset the next older page starts at, so the transcript
 * window's "Show earlier" action knows to backfill over REST once the
 * in-memory store is fully materialized (see app/chat/transcript-backfill).
 *
 * Offsets use the backend's `order: 'latest'` semantics: measured back from
 * the NEWEST row, with each page returned in chronological order — so the
 * page immediately older than N already-loaded tail rows starts at offset N.
 */

import { atom } from 'nanostores'

import type { SessionMessagesResponse } from '@/types/hermes'

export interface TranscriptTailState {
  /** Offset (back from the newest row) where the next older page starts. */
  nextOffset: number
  /** The last hydration page was exactly the page limit, so older rows
   *  likely exist beyond what the in-memory store holds. */
  possiblyTruncated: boolean
  /** The request route captured at hydration time, replayed verbatim by a
   *  later backfill so it reaches the backend that served the tail. The
   *  resolved OWNER is the map key, not this field: an ambient read must stay
   *  ambient even once the server has named its profile. */
  profile?: TranscriptProfileScope
}

export type TranscriptProfileScope =
  | null
  | string
  | {
      connectionId?: null | string
      profile?: null | string
    }

export const $transcriptTailBySessionId = atom<Record<string, TranscriptTailState>>({})
const TRANSCRIPT_TAIL_LIMIT = 256
let transcriptTailOrder: string[] = []

type TailPage = Pick<SessionMessagesResponse, 'messages' | 'pagination'>

function normalizedScope(profile?: TranscriptProfileScope): { connectionId: string; profile: string } | null {
  // A bare string is the legacy "named profile" spelling: it always names a
  // profile, so empty means the default one.
  if (typeof profile === 'string') {
    return { connectionId: '', profile: profile.trim() || 'default' }
  }

  if (!profile) {
    return null
  }

  return {
    connectionId: String(profile.connectionId || '').trim(),
    // An omitted profile targets the serving process, not necessarily default.
    profile: String(profile.profile || '').trim()
  }
}

function transcriptTailKey(storedSessionId: string, profile?: TranscriptProfileScope): string {
  const scope = normalizedScope(profile)

  return scope ? JSON.stringify([scope.connectionId, scope.profile, storedSessionId]) : storedSessionId
}

function matchingTailEntries(storedSessionId: string): Array<[string, TranscriptTailState]> {
  return Object.entries($transcriptTailBySessionId.get()).filter(([key]) => {
    if (key === storedSessionId) {
      return true
    }

    try {
      const parsed = JSON.parse(key)

      return Array.isArray(parsed) && parsed.length === 3 && parsed[2] === storedSessionId
    } catch {
      return false
    }
  })
}

/**
 * Resolve the single tail entry an address refers to, returning its key too.
 *
 * `undefined` when the address is ambiguous (two scopes recorded for one stored
 * session, no profile to disambiguate) or absent. Callers that page or rewind
 * must then leave the transcript alone: a route that cannot be addressed
 * exactly is a route that cannot be fetched from.
 */
function resolveTailEntry(
  storedSessionId: string,
  profile?: TranscriptProfileScope
): [key: string, state: TranscriptTailState] | undefined {
  const current = $transcriptTailBySessionId.get()

  if (profile !== undefined) {
    const key = transcriptTailKey(storedSessionId, profile)
    const state = current[key]

    return state ? [key, state] : undefined
  }

  const matches = matchingTailEntries(storedSessionId)

  return matches.length === 1 ? matches[0] : undefined
}

function tailStateFromPage(page: TailPage, profile?: TranscriptProfileScope): TranscriptTailState {
  const pagination = page.pagination

  // No pagination metadata is a legacy backend that ignored the paging query
  // and returned the full transcript: nothing is truncated.
  if (!pagination || pagination.limit <= 0) {
    return { nextOffset: page.messages.length, possiblyTruncated: false, profile }
  }

  return {
    nextOffset: pagination.offset + page.messages.length,
    possiblyTruncated: page.messages.length >= pagination.limit,
    profile
  }
}

function sameTailState(a: TranscriptTailState, b: TranscriptTailState): boolean {
  return (
    a.nextOffset === b.nextOffset &&
    a.possiblyTruncated === b.possiblyTruncated &&
    JSON.stringify(a.profile ?? null) === JSON.stringify(b.profile ?? null)
  )
}

function setTranscriptTailEntry(key: string, state: TranscriptTailState): void {
  const current = $transcriptTailBySessionId.get()

  // Heartbeats re-record identical entries; a needless .set() re-renders the
  // whole chat tree through the tail atom (#113842).
  if (key in current && sameTailState(current[key], state)) {
    // Still a use of the entry: bump it to the MRU end so the constantly
    // re-read active session is not the next eviction candidate.
    transcriptTailOrder = [...transcriptTailOrder.filter(candidate => candidate !== key), key]

    return
  }

  const existing = new Set(Object.keys(current))
  transcriptTailOrder = transcriptTailOrder.filter(candidate => candidate !== key && existing.has(candidate))
  transcriptTailOrder.push(key)

  const next = { ...current, [key]: state }

  while (transcriptTailOrder.length > TRANSCRIPT_TAIL_LIMIT) {
    const oldest = transcriptTailOrder.shift()

    if (oldest !== undefined) {
      delete next[oldest]
    }
  }

  $transcriptTailBySessionId.set(next)
}

/** Record the outcome of a tail hydration (`getLatestSessionMessages`).
 *  `route` is what the request was sent with; `owner` is the resolved backend
 *  the entry is keyed under (defaults to the route when they coincide). */
export function recordTranscriptTail(
  storedSessionId: string,
  page: TailPage,
  route?: TranscriptProfileScope,
  owner: TranscriptProfileScope | undefined = route
): void {
  if (!storedSessionId) {
    return
  }

  const key = transcriptTailKey(storedSessionId, owner)

  // This runs before the active refresh decides whether the page is
  // authoritative (use-background-sync). A transient zero-row read must not
  // turn a known-truncated tail into "nothing earlier", or "Show earlier"
  // disarms while the rows are still on the backend.
  if (page.messages.length === 0 && $transcriptTailBySessionId.get()[key]?.possiblyTruncated) {
    return
  }

  setTranscriptTailEntry(key, tailStateFromPage(page, route))
}

/** Advance the bookkeeping after one older backfill page landed. */
export function recordTranscriptBackfillPage(
  storedSessionId: string,
  page: TailPage,
  profile?: TranscriptProfileScope
): void {
  const entry = resolveTailEntry(storedSessionId, profile)

  if (!entry) {
    return
  }

  setTranscriptTailEntry(entry[0], tailStateFromPage(page, entry[1].profile))
}

/**
 * Re-arm the older-page fetch after in-store history was released.
 *
 * `boundRetainedTranscript` (app/chat/transcript-retention) drops rows that are
 * older than the live window once they are persisted — but they stay reachable
 * only while the transcript still reports older rows as fetchable. Rewind the
 * entry's offset by the released rows so the next "Show earlier" fetches them
 * back instead of treating the in-memory store as the whole transcript.
 *
 * The offset is decremented RELATIVE to what the backend reported, never
 * recomputed from the store's own row count, and it is decremented by BACKEND
 * rows: the hydration fold merges a turn's tool rows into the assistant message
 * they belong to (`ChatMessage.serverRowSpan` carries how many), and the backend
 * pages display history (an `include_compacted` read is grouped by display
 * order; inactive rows are not counted). Subtracting from the backend's own
 * number can only overlap a page that is already in memory — which the merge
 * dedupes — where an over-counted absolute offset would skip rows the reader
 * can then never reach.
 *
 * Returns false when the session has no entry: without one there is no route
 * recorded to fetch a page from, so the caller must keep its rows rather than
 * release history nothing can bring back.
 */
export function rewindTranscriptTail(
  storedSessionId: string,
  releasedServerRows: number,
  profile?: TranscriptProfileScope
): boolean {
  if (!storedSessionId || releasedServerRows <= 0) {
    return false
  }

  const entry = resolveTailEntry(storedSessionId, profile)

  if (!entry) {
    return false
  }

  setTranscriptTailEntry(entry[0], {
    nextOffset: Math.max(0, entry[1].nextOffset - releasedServerRows),
    possiblyTruncated: true,
    profile: entry[1].profile
  })

  return true
}

export function transcriptTailState(
  storedSessionId: null | string | undefined,
  profile?: TranscriptProfileScope
): TranscriptTailState | undefined {
  if (!storedSessionId) {
    return undefined
  }

  return resolveTailEntry(storedSessionId, profile)?.[1]
}

/** Drops the LRU order as well as the atom. */
export function clearTranscriptTailPaging(): void {
  transcriptTailOrder = []
  $transcriptTailBySessionId.set({})
}

export function clearTranscriptTail(storedSessionId: string, profile?: TranscriptProfileScope): void {
  const current = $transcriptTailBySessionId.get()

  const keys =
    profile === undefined
      ? matchingTailEntries(storedSessionId).map(([key]) => key)
      : [transcriptTailKey(storedSessionId, profile)]

  if (keys.length === 0) {
    return
  }

  const next = { ...current }

  for (const key of keys) {
    delete next[key]
  }

  const removed = new Set(keys)
  transcriptTailOrder = transcriptTailOrder.filter(key => !removed.has(key))

  $transcriptTailBySessionId.set(next)
}
