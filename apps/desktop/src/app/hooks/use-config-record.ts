import { useQuery } from '@tanstack/react-query'

import { getHermesConfigRecord } from '@/hermes'
import { queryClient, writeCache } from '@/lib/query-client'
import { normalizeProfileKey } from '@/store/profile'
import type { HermesConfigRecord } from '@/types/hermes'

// One shared cache for the whole profile config record (`GET /api/config`).
// Every settings surface (MCP, model, config) reads and writes through this key
// so a save in one shows in the others, and revisiting a tab paints the cache
// instead of blanking on a fresh fetch.
//
// Distinct from session/hooks/use-hermes-config.ts, which is side-effecting —
// it pushes personality/cwd/voice/… into the session stores for live chat.
export const HERMES_CONFIG_KEY = ['hermes-config-record'] as const

// Per-profile cache key. The base key (no profile suffix) is the app-wide
// active profile, unchanged for every caller that passes nothing. An explicit
// profile — the Capabilities profile-scope selector configuring ANOTHER
// profile — gets its own suffixed key so switching the selector refetches and
// never paints stale cross-profile config (the AGENTS.md scope-in-key rule).
export const hermesConfigKey = (profile?: null | string) =>
  profile == null ? HERMES_CONFIG_KEY : ([...HERMES_CONFIG_KEY, normalizeProfileKey(profile)] as const)

// staleTime 0 → serve cache instantly, background-revalidate on every mount.
// `profile` scopes both the query key and the fetch; omitting it preserves the
// exact app-wide behavior (base key, `profileScoped(undefined)` fallback).
export const useHermesConfigRecord = (profile?: null | string) =>
  useQuery({
    queryKey: hermesConfigKey(profile),
    // null/undefined both mean "no override" → fetch with undefined so
    // profileScoped falls back to the app-wide active profile (passing null
    // would wrongly target the primary backend).
    queryFn: () => getHermesConfigRecord(profile ?? undefined),
    staleTime: 0
  })

// setHermesConfigCache writes the app-wide (base-key) record. Pass a profile to
// write the suffixed per-profile cache instead — keeps the selector's optimistic
// write-through landing on the same key its query reads.
export const setHermesConfigCache = writeCache<HermesConfigRecord>(HERMES_CONFIG_KEY)
export const hermesConfigCacheWriter = (profile?: null | string) =>
  writeCache<HermesConfigRecord>(hermesConfigKey(profile))

export const invalidateHermesConfig = (profile?: null | string) =>
  queryClient.invalidateQueries({ queryKey: hermesConfigKey(profile) })
