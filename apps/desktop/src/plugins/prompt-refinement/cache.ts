/**
 * Reuse for refinements: the same draft refined again with the same depth,
 * style, guidance, chat excerpt and model route returns the earlier answer
 * instantly instead of paying for another model call (switching modes,
 * reopening a review, Approval-First then Automatic on the same words).
 *
 *  - The key is everything the answer depends on (`cacheKey`): the whole
 *    request, the connection and profile that answer it, and with "This
 *    chat's model" the chat and its model. An answer is never reused for
 *    another profile, connection, chat model or chat excerpt.
 *  - Entries expire after `CACHE_TTL_MS`, so a model setting changed where
 *    the plugin cannot see it (the auxiliary slot) takes effect soon.
 *  - Only successful answers are kept; "Try again" (attempt > 1) always asks
 *    the model afresh. Memory only: nothing is written to disk.
 */

import type { RefinementModelSource, RefineRequest, RefineResult } from './types'

export const CACHE_LIMIT = 20
export const CACHE_TTL_MS = 10 * 60_000

/** Where a refinement is answered, as far as the renderer can tell. */
export interface CacheRoute {
  /** Connection and profile that own the chat (and so the auxiliary slot). */
  connectionId: string
  profile: string
  modelSource: RefinementModelSource
  /** The chat whose model is borrowed ("This chat's model"). */
  sessionId: null | string
  /** That chat's model as the app shows it. */
  chatModel: string
}

export function cacheKey(request: RefineRequest, route: CacheRoute): string {
  const borrow = route.modelSource === 'chat'

  return JSON.stringify([
    request,
    route.connectionId,
    route.profile,
    route.modelSource,
    borrow ? route.sessionId : null,
    borrow ? route.chatModel : null
  ])
}

export interface RefinementCache {
  run: (key: string, attempt: number, compute: () => Promise<RefineResult>) => Promise<RefineResult>
  clear: () => void
}

interface Entry {
  result: RefineResult
  storedAt: number
}

export function createRefinementCache(
  limit = CACHE_LIMIT,
  ttlMs = CACHE_TTL_MS,
  now: () => number = Date.now
): RefinementCache {
  const entries = new Map<string, Entry>()

  const fresh = (key: string): RefineResult | undefined => {
    const entry = entries.get(key)

    if (entry && now() - entry.storedAt >= ttlMs) {
      entries.delete(key)

      return undefined
    }

    return entry?.result
  }

  return {
    async run(key, attempt, compute) {
      const hit = attempt <= 1 ? fresh(key) : undefined

      if (hit) {
        // Refresh recency (least recently used goes first); the age is kept.
        const entry = entries.get(key) as Entry
        entries.delete(key)
        entries.set(key, entry)

        return hit
      }

      const result = await compute()

      if (result.ok) {
        entries.delete(key)
        entries.set(key, { result, storedAt: now() })

        while (entries.size > limit) {
          entries.delete(entries.keys().next().value as string)
        }
      }

      return result
    },
    clear: () => entries.clear()
  }
}
