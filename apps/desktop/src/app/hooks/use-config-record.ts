import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'

import { getHermesConfigRecord, type ProfileScope, profileScopeKey } from '@/hermes'
import { queryClient, writeCache } from '@/lib/query-client'
import { BACKEND_BOOT_WAIT_TIMEOUT_MS, withTimeout } from '@/lib/with-timeout'
import { $activeGatewayProfile } from '@/store/profile'
import type { HermesConfigRecord } from '@/types/hermes'

// One shared cache for the whole profile config record (`GET /api/config`).
// Every settings surface (MCP, model, config) reads and writes through this key
// so a save in one shows in the others, and revisiting a tab paints the cache
// instead of blanking on a fresh fetch.
//
// Distinct from session/hooks/use-hermes-config.ts, which is side-effecting —
// it pushes personality/cwd/voice/… into the session stores for live chat.
export const HERMES_CONFIG_KEY = ['hermes-config-record'] as const

const CONFIG_LOAD_TIMEOUT_MESSAGE = 'Timed out loading Hermes config'

// Per-scope cache key. Always suffixes the resolved profile so follow-active
// and an explicit selector never share a row or an in-flight fetch (AGENTS.md
// scope-in-key). Follow-active callers pass nothing; the hook / writer
// substitutes $activeGatewayProfile. profileScopeKey folds a remote pin's
// connection id into the suffix, so two gateways' same-named profiles never
// share a cache row.
export const hermesConfigKey = (profile?: ProfileScope) =>
  [...HERMES_CONFIG_KEY, profileScopeKey(profile)] as const

function activeProfileScope(): string {
  return profileScopeKey($activeGatewayProfile.get())
}

// staleTime 0 → serve cache instantly, background-revalidate on every mount.
// `profile` scopes both the query key and the fetch; omitting it follows the
// app-wide active profile (same as capabilityScoped(undefined)).
export const useHermesConfigRecord = (profile?: ProfileScope) => {
  const active = useStore($activeGatewayProfile)
  const keyProfile = profile ?? active

  return useQuery({
    queryKey: hermesConfigKey(keyProfile),
    // null/undefined both mean "no override" → fetch with undefined so
    // capabilityScoped falls back to the app-wide active profile (passing null
    // would wrongly target the primary backend).
    queryFn: () =>
      withTimeout(
        getHermesConfigRecord(profile ?? undefined),
        BACKEND_BOOT_WAIT_TIMEOUT_MS,
        CONFIG_LOAD_TIMEOUT_MESSAGE
      ),
    staleTime: 0
  })
}

// Writes the follow-active (current gateway profile) record. Pass a profile to
// hermesConfigCacheWriter to write a suffixed per-profile cache instead.
export const setHermesConfigCache = (
  next:
    | HermesConfigRecord
    | undefined
    | ((prev: HermesConfigRecord | undefined) => HermesConfigRecord | undefined)
): void => writeCache<HermesConfigRecord>(hermesConfigKey(activeProfileScope()))(next)

export const hermesConfigCacheWriter = (profile?: ProfileScope) =>
  writeCache<HermesConfigRecord>(hermesConfigKey(profile ?? activeProfileScope()))

export const invalidateHermesConfig = (profile?: ProfileScope) =>
  queryClient.invalidateQueries({ queryKey: hermesConfigKey(profile ?? activeProfileScope()) })
