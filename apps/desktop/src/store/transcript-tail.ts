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
  /** Owning profile captured at hydration time, so a later backfill routes
   *  its REST read to the same backend that served the tail. */
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
  if (typeof profile === 'string') {
    return { connectionId: '', profile: profile.trim() || 'default' }
  }

  if (!profile) {
    return null
  }

  return {
    connectionId: String(profile.connectionId || '').trim(),
    profile: String(profile.profile || '').trim() || 'default'
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

function setTranscriptTailEntry(key: string, state: TranscriptTailState): void {
  const current = $transcriptTailBySessionId.get()
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

/** Record the outcome of a tail hydration (`getLatestSessionMessages`). */
export function recordTranscriptTail(storedSessionId: string, page: TailPage, profile?: TranscriptProfileScope): void {
  if (!storedSessionId) {
    return
  }

  const key = transcriptTailKey(storedSessionId, profile)
  setTranscriptTailEntry(key, tailStateFromPage(page, profile))
}

/** Advance the bookkeeping after one older backfill page landed. */
export function recordTranscriptBackfillPage(
  storedSessionId: string,
  page: TailPage,
  profile?: TranscriptProfileScope
): void {
  const current = $transcriptTailBySessionId.get()

  const selected: Array<[string, TranscriptTailState | undefined]> =
    profile === undefined
      ? matchingTailEntries(storedSessionId)
      : [[transcriptTailKey(storedSessionId, profile), current[transcriptTailKey(storedSessionId, profile)]]]

  if (selected.length !== 1) {
    return
  }

  const [key, previous] = selected[0]

  if (!previous) {
    return
  }

  setTranscriptTailEntry(key, tailStateFromPage(page, previous.profile))
}

export function transcriptTailState(
  storedSessionId: null | string | undefined,
  profile?: TranscriptProfileScope
): TranscriptTailState | undefined {
  if (!storedSessionId) {
    return undefined
  }

  if (profile !== undefined) {
    return $transcriptTailBySessionId.get()[transcriptTailKey(storedSessionId, profile)]
  }

  const matches = matchingTailEntries(storedSessionId)

  return matches.length === 1 ? matches[0][1] : undefined
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
