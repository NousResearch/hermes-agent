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

export const timelineIndexKey = (id: string, scope: ProfileScope) => JSON.stringify([id, scope])
export const cachedTimelineIndex = (key: string) => cache.get(key)

/** One bounded metadata page per request; never fetch tool or assistant bodies. */
export function fetchTimelineIndex(id: string, scope: ProfileScope): Promise<TimelineIndex> {
  const key = timelineIndexKey(id, scope)
  const cached = cache.get(key)
  const previous = cached?.complete && cached.expires <= Date.now() ? undefined : cached

  if (cached?.complete && cached.expires > Date.now()) {
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
