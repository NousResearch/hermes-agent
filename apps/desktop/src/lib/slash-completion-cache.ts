import { atom } from 'nanostores'

import { queryClient } from '@/lib/query-client'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

// Root for every cached `/` completion response — the bare-slash catalog and
// each typed query. Not in PROFILE_INDEPENDENT_QUERY_ROOTS, so a profile or
// gateway switch drops it with the rest of the profile-scoped cache.
const SLASH_COMPLETIONS_KEY = 'slash-completions'

// The command catalog and its completions are a scan of the command registry
// plus the skills on disk: they change when a skill is added, removed, or
// toggled — not while the user types. Both gateway calls are expensive on the
// backend (a full skills-dir scan per request), so hold the answer for an hour
// and let the mutation sites invalidate it when the truth actually moves.
const SLASH_COMPLETIONS_TTL_MS = 60 * 60_000

/** Serve a `/` completion response from cache, fetching only when stale. */
export function cachedSlashCompletion<T>(key: string, fetcher: () => Promise<T>): Promise<T> {
  return queryClient.fetchQuery({
    queryKey: [SLASH_COMPLETIONS_KEY, key],
    queryFn: fetcher,
    gcTime: SLASH_COMPLETIONS_TTL_MS,
    staleTime: SLASH_COMPLETIONS_TTL_MS,
    // A completion is only worth having while the popover is open. Retrying a
    // failed lookup with backoff would spend seconds answering a keystroke the
    // user has already typed past; the caller falls back to an empty list and
    // the next keystroke asks again.
    retry: false
  })
}

/** True when `cachedSlashCompletion(key)` will resolve without a round trip. */
export function hasCachedSlashCompletion(key: string): boolean {
  const state = queryClient.getQueryState([SLASH_COMPLETIONS_KEY, key])

  return state?.data !== undefined && Date.now() - state.dataUpdatedAt < SLASH_COMPLETIONS_TTL_MS
}

/**
 * Read a cached completion response without fetching. For data that improves a
 * response but must not cost a round trip to get — the catalog's per-skill
 * usage map, which refines the ordering of a typed query but is not worth
 * delaying that query for.
 */
export function peekCachedSlashCompletion<T>(key: string): T | undefined {
  return hasCachedSlashCompletion(key) ? queryClient.getQueryData<T>([SLASH_COMPLETIONS_KEY, key]) : undefined
}

// `@` path completions are a directory listing, which unlike the command
// catalog CAN change under the user (a build writes files, a branch switch
// rewrites a tree). They get the same de-duplication but a short TTL: long
// enough that walking back up a path you just walked down is instant, short
// enough that the listing never looks stale.
const PATH_COMPLETIONS_KEY = 'path-completions'
const PATH_COMPLETIONS_TTL_MS = 15_000

/** Serve an `@` path completion from cache, fetching only when stale. */
export function cachedPathCompletion<T>(key: string, fetcher: () => Promise<T>): Promise<T> {
  return queryClient.fetchQuery({
    queryKey: [PATH_COMPLETIONS_KEY, key],
    queryFn: fetcher,
    gcTime: PATH_COMPLETIONS_TTL_MS,
    staleTime: PATH_COMPLETIONS_TTL_MS,
    retry: false
  })
}

/** True when `cachedPathCompletion(key)` will resolve without a round trip. */
export function hasCachedPathCompletion(key: string): boolean {
  const state = queryClient.getQueryState([PATH_COMPLETIONS_KEY, key])

  return state?.data !== undefined && Date.now() - state.dataUpdatedAt < PATH_COMPLETIONS_TTL_MS
}

/**
 * Bumped on every invalidation. The composer's completion adapter de-dupes by
 * query, so an unchanged `/` would never re-ask on its own — it watches this
 * instead to know the answer it's holding is no longer current.
 */
export const $slashCompletionsEpoch = atom(0)

/**
 * Drop cached `/` completions. Called from every site that changes which
 * skills exist or are enabled — install/uninstall/update from the hub, a
 * skill toggle or delete in Capabilities — so the composer's list matches
 * the backend without waiting out the TTL.
 */
export function invalidateSlashCompletions(): void {
  void queryClient.invalidateQueries({ queryKey: [SLASH_COMPLETIONS_KEY] })
  $slashCompletionsEpoch.set($slashCompletionsEpoch.get() + 1)
}

// Each profile has its own skills directory, so a cached catalog is only valid
// for the profile that produced it. Dropped at the source rather than in the
// composer so it holds whether or not a chat is mounted at switch time.
let cachedProfile: null | string = null

$activeGatewayProfile.subscribe(value => {
  const key = normalizeProfileKey(value)

  if (cachedProfile !== null && cachedProfile !== key) {
    invalidateSlashCompletions()
  }

  cachedProfile = key
})
