import { useQuery } from '@tanstack/react-query'

import { getApiRequestProfile, getHermesConfigRecord, type ProfileScope, profileScopeKey } from '@/hermes'
import { queryClient, writeCache } from '@/lib/query-client'
import type { HermesConfigRecord } from '@/types/hermes'

// One shared cache for the whole profile config record (`GET /api/config`).
// Every settings surface (MCP, model, config) reads and writes through this key
// so a save in one shows in the others, and revisiting a tab paints the cache
// instead of blanking on a fresh fetch.
//
// Distinct from session/hooks/use-hermes-config.ts, which is side-effecting —
// it pushes personality/cwd/voice/… into the session stores for live chat.
export const HERMES_CONFIG_KEY = ['hermes-config-record'] as const

// Per-scope cache key. An omitted scope resolves to the app-wide active
// profile so ambient consumers share the same row as Settings. An explicit scope —
// the Capabilities scope selector configuring ANOTHER profile, possibly on
// another registered gateway — gets its own suffixed key so switching the
// selector refetches and never paints stale cross-profile config (the
// AGENTS.md scope-in-key rule). profileScopeKey folds a remote pin's
// connection id into the suffix, so two gateways' same-named profiles never
// share a cache row.
export const hermesConfigKey = (profile?: ProfileScope) => {
  const resolvedProfile = profile ?? getApiRequestProfile() ?? undefined

  return resolvedProfile == null
    ? HERMES_CONFIG_KEY
    : ([...HERMES_CONFIG_KEY, profileScopeKey(resolvedProfile)] as const)
}

// staleTime 0 → serve cache instantly, background-revalidate on every mount.
// `profile` scopes both the query key and the fetch; omitting it preserves
// ambient request routing while keying the response by its concrete profile.
export const useHermesConfigRecord = (profile?: ProfileScope) =>
  useQuery({
    queryKey: hermesConfigKey(profile),
    // null/undefined both mean "no override" → fetch with undefined so
    // capabilityScoped falls back to the app-wide active profile (passing null
    // would wrongly target the primary backend).
    queryFn: () => getHermesConfigRecord(profile ?? undefined),
    staleTime: 0
  })

// Ambient writes resolve their concrete profile at call time, matching
// useHermesConfigRecord() even after an app-wide profile switch.
export const setHermesConfigCache = (
  next: HermesConfigRecord | undefined | ((prev: HermesConfigRecord | undefined) => HermesConfigRecord | undefined)
): void => writeCache<HermesConfigRecord>(hermesConfigKey())(next)
export const hermesConfigCacheWriter = (profile?: ProfileScope) =>
  writeCache<HermesConfigRecord>(hermesConfigKey(profile))

export const invalidateHermesConfig = (profile?: ProfileScope) =>
  queryClient.invalidateQueries({ queryKey: hermesConfigKey(profile) })
