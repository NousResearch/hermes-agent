import { useQuery } from '@tanstack/react-query'

import { type ProfileScope, profileScopeKey } from '@/api/client'
import { peekConfigReadOrigin } from '@/api/config'
import { getHermesConfigRecord } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import type { HermesConfigRecord } from '@/types/hermes'

// One shared cache for the whole profile config record (`GET /api/config`).
// Every settings surface (MCP, model, config) reads and writes through this key
// so a save in one shows in the others, and revisiting a tab paints the cache
// instead of blanking on a fresh fetch.
//
// Distinct from session/hooks/use-hermes-config.ts, which is side-effecting —
// it pushes personality/cwd/voice/… into the session stores for live chat.
export const HERMES_CONFIG_KEY = ['hermes-config-record'] as const
type HermesConfigCacheEntry = {
  record: HermesConfigRecord
  scope: { connectionId?: string; profile?: string } | undefined
}
type HermesConfigCacheWrite =
  | HermesConfigRecord
  | undefined
  | ((previous: HermesConfigRecord | undefined) => HermesConfigRecord | undefined)

// Per-scope cache key. The base key (no suffix) is the app-wide active
// profile, unchanged for every caller that passes nothing. An explicit scope —
// the Capabilities scope selector configuring ANOTHER profile, possibly on
// another registered gateway — gets its own suffixed key so switching the
// selector refetches and never paints stale cross-profile config (the
// AGENTS.md scope-in-key rule). profileScopeKey folds a remote pin's
// connection id into the suffix, so two gateways' same-named profiles never
// share a cache row.
export const hermesConfigKey = (profile?: ProfileScope) =>
  profile == null ? HERMES_CONFIG_KEY : ([...HERMES_CONFIG_KEY, profileScopeKey(profile)] as const)

// staleTime 0 → serve cache instantly, background-revalidate on every mount.
// `profile` scopes both the query key and the fetch; omitting it preserves the
// exact app-wide behavior (base key, `profileScoped(undefined)` fallback).
export const useHermesConfigRecord = (profile?: ProfileScope) => {
  const query = useQuery<HermesConfigCacheEntry>({
    queryKey: hermesConfigKey(profile),
    // null/undefined both mean "no override" → fetch with undefined so
    // capabilityScoped falls back to the app-wide active profile (passing null
    // would wrongly target the primary backend).
    queryFn: async () => {
      const record = await getHermesConfigRecord(profile ?? undefined)

      // Query observers share a row. Keep provenance in that row, rather than
      // in a hook instance, so a second observer cannot display A's record
      // while using B's ambient route for its write.
      return { record, scope: peekConfigReadOrigin(record) }
    },
    staleTime: 0
  })

  return {
    ...query,
    data: query.data?.record,
    refetch: async (...args: Parameters<typeof query.refetch>) => {
      const result = await query.refetch(...args)

      return { ...result, data: result.data?.record }
    },
    // Leave this undefined until a read completes so `_apiProfile` remains
    // the normal fallback for writers that have no cached record yet.
    writeScope: query.data?.scope
  }
}

// setHermesConfigCache writes the app-wide (base-key) record. Pass a profile to
// write the suffixed per-profile cache instead — keeps the selector's optimistic
// write-through landing on the same key its query reads.
const configCacheWriter = (profile?: ProfileScope) => (next: HermesConfigCacheWrite): void => {
  queryClient.setQueryData<HermesConfigCacheEntry>(hermesConfigKey(profile), previous => {
    const record = typeof next === 'function' ? next(previous?.record) : next

    return record === undefined ? undefined : { record, scope: previous?.scope }
  })
}

export const setHermesConfigCache = configCacheWriter()
export const hermesConfigCacheWriter = (profile?: ProfileScope) => configCacheWriter(profile)

export const invalidateHermesConfig = (profile?: ProfileScope) =>
  queryClient.invalidateQueries({ queryKey: hermesConfigKey(profile) })
