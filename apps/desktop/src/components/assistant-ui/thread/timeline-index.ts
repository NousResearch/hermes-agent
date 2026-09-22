import { capabilityScoped, hermesApi, type ProfileScope } from '@/api/client'

import type { TimelineEntry } from './timeline-data'

interface TimelinePage {
  entries: Array<{ row_id: number; preview: string; timestamp?: number }>
  pagination: { next_cursor: number | null; has_more: boolean }
}

export interface TimelineIndex {
  entries: TimelineEntry[]
  complete: boolean
  cursor?: number
  expires: number
}

const cache = new Map<string, TimelineIndex>()
const requests = new Map<string, Promise<TimelineIndex>>()
const MAX_CACHED_SESSIONS = 12
const TTL = 60_000

/** One lookup may advance a partially loaded index by this many pages. */
const MAX_INDEX_PAGES_PER_LOOKUP = 3

export const timelineIndexKey = (id: string, scope: ProfileScope) => JSON.stringify([id, scope])
export const cachedTimelineIndex = (key: string) => cache.get(key)

/**
 * One bounded metadata page per request; never fetch tool or assistant bodies.
 * A complete index is final only up to the turns that existed when it was
 * read: `beyondRowId` names a prompt the caller has seen (the live tail's
 * newest, a jump anchor), and a complete index that does not reach it pages
 * forward from its own cursor instead of answering from the cache.
 */
export function fetchTimelineIndex(id: string, scope: ProfileScope, beyondRowId?: number): Promise<TimelineIndex> {
  const key = timelineIndexKey(id, scope)
  const cached = cache.get(key)
  const previous = cached?.complete && cached.expires <= Date.now() ? undefined : cached

  if (
    cached?.complete &&
    cached.expires > Date.now() &&
    (beyondRowId === undefined || marksReach(cached.entries, beyondRowId))
  ) {
    return Promise.resolve(cached)
  }

  const inflight = requests.get(key)

  if (inflight) {
    return inflight
  }

  const route = {
    ...capabilityScoped(scope),
    ...(typeof scope === 'object' && scope?.connectionId === 'local' ? { connectionId: 'local' } : {})
  }

  const query = new URLSearchParams({ limit: '500' })

  if (route.profile) {
    query.set('profile', route.profile)
  }

  if (previous?.cursor !== undefined) {
    query.set('after_row_id', String(previous.cursor))
  }

  const request = hermesApi<TimelinePage>({
    ...route,
    path: `/api/sessions/${encodeURIComponent(id)}/timeline?${query}`,
    passive: true
  })
    .then(page => {
      const merged = new Map(previous?.entries.map(entry => [entry.rowId, entry]))

      for (const entry of page.entries) {
        merged.set(entry.row_id, { id: `history:${entry.row_id}`, rowId: entry.row_id, preview: entry.preview })
      }

      const value = {
        entries: [...merged.values()],
        complete: !page.pagination.has_more,
        cursor: page.pagination.next_cursor ?? previous?.cursor,
        expires: Date.now() + TTL
      }

      cache.delete(key)
      cache.set(key, value)

      while (cache.size > MAX_CACHED_SESSIONS) {
        cache.delete(cache.keys().next().value!)
      }

      return value
    })
    .finally(() => requests.delete(key))

  requests.set(key, request)

  return request
}

/** Marks are chronological, so the greatest id below the anchor is its predecessor. */
function promptBefore(entries: readonly TimelineEntry[], rowId: number): number | null {
  let previous: number | null = null

  for (const entry of entries) {
    if (entry.rowId !== undefined && entry.rowId < rowId) {
      previous = entry.rowId
    }
  }

  return previous
}

/** Whether the loaded marks reach the anchor, i.e. `promptBefore` is its neighbour. */
const marksReach = (entries: readonly TimelineEntry[], rowId: number) =>
  entries.some(entry => entry.rowId !== undefined && entry.rowId >= rowId)

/**
 * The prompt mark immediately before `rowId` on the shared timeline range — the
 * same marks the rail draws, so "Show earlier" and the rail page one range
 * instead of each inventing its own reachability. Pages load oldest-first, so
 * an anchor past the loaded marks advances the index (cached, coalesced with
 * the rail's own loadMore) rather than guessing across the gap. Resolves null
 * only when nothing precedes the anchor, or when the index cannot name it.
 */
export async function previousPromptRowId(
  id: string,
  scope: ProfileScope,
  rowId: number | undefined
): Promise<number | null> {
  if (rowId === undefined || !Number.isSafeInteger(rowId) || rowId <= 0) {
    return null
  }

  let previous: number | null = null
  let known = -1

  for (let page = 0; page < MAX_INDEX_PAGES_PER_LOOKUP; page++) {
    const index = await fetchTimelineIndex(id, scope, rowId)

    previous = promptBefore(index.entries, rowId)

    // A complete index that gained nothing cannot name the anchor (an unpersisted
    // or non-prompt row); stop rather than re-read the tail page.
    if (previous === null || marksReach(index.entries, rowId) || (index.complete && index.entries.length === known)) {
      return previous
    }

    known = index.entries.length
  }

  return previous
}
