import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'

// profileScopeKey comes from its home module, not the '@/hermes' barrel: the
// ambient key calls it on every render, and settings tests that mock the
// barrel would otherwise have to re-export it.
import { profileScopeKey } from '@/api/client'
import { getHermesConfigRecord, type ProfileScope } from '@/hermes'
import { queryClient, writeCache } from '@/lib/query-client'
import { $activeConnectionId } from '@/store/connections'
import { $activeGatewayRoute } from '@/store/gateway'
import type { HermesConfigRecord } from '@/types/hermes'

// One shared cache for the whole profile config record (`GET /api/config`).
// Every settings surface (MCP, model, config) reads and writes through this key
// so a save in one shows in the others, and revisiting a tab paints the cache
// instead of blanking on a fresh fetch.
//
// Distinct from session/hooks/use-hermes-config.ts, which is side-effecting —
// it pushes personality/cwd/voice/… into the session stores for live chat.
export const HERMES_CONFIG_KEY = ['hermes-config-record'] as const

/** Source of the AMBIENT record: the active registry connection (null → the
 *  legacy primary route) and the bare profile the active gateway serves. */
export interface AmbientConfigScope {
  connectionId: null | string
  profileKey: string
}

export const ambientConfigScope = (): AmbientConfigScope => ({
  connectionId: $activeConnectionId.get(),
  profileKey: $activeGatewayRoute.get()
})

// Scope suffix of the ambient record. The key must carry the source (AGENTS.md
// scope-in-key rule): with a bare root key, switching gateways served the
// previous machine's record from cache, and the next settings save PUT that
// whole record — deep-merged on the remote — onto the other machine's
// config.yaml. Same `connectionId::profile` shape as an explicit pin, so a pin
// on the active source shares its row.
export const ambientConfigScopeKey = (ambient: AmbientConfigScope = ambientConfigScope()): string =>
  profileScopeKey({ connectionId: ambient.connectionId ?? 'primary', profile: ambient.profileKey })

// Per-scope cache key. No profile → the ambient record of the active gateway
// and profile (resolved on every call, never cached). An explicit scope — the
// Capabilities scope selector configuring ANOTHER profile, possibly on another
// registered gateway — gets its own suffix so switching the selector refetches
// and never paints stale cross-profile config. profileScopeKey folds a remote
// pin's connection id into the suffix, so two gateways' same-named profiles
// never share a cache row.
export const hermesConfigKey = (profile?: ProfileScope, ambient?: AmbientConfigScope) =>
  [...HERMES_CONFIG_KEY, profile == null ? ambientConfigScopeKey(ambient) : profileScopeKey(profile)] as const

// staleTime 0 → serve cache instantly, background-revalidate on every mount.
// `profile` scopes both the query key and the fetch; omitting it targets the
// app-wide active profile (`profileScoped(undefined)` fallback) on the active
// gateway.
export const useHermesConfigRecord = (profile?: ProfileScope) => {
  // Reactive reads, not store getters: under the React Compiler a value with
  // no reactive inputs is computed once per component instance, so a
  // getter-based key would freeze on the first gateway and keep serving its
  // record after a switch.
  const connectionId = useStore($activeConnectionId)
  const profileKey = useStore($activeGatewayRoute)

  return useQuery({
    queryKey: hermesConfigKey(profile, { connectionId, profileKey }),
    // null/undefined both mean "no override" → fetch with undefined so
    // capabilityScoped falls back to the app-wide active profile (passing null
    // would wrongly target the primary backend).
    queryFn: () => getHermesConfigRecord(profile ?? undefined),
    staleTime: 0
  })
}

type ConfigCacheUpdate = HermesConfigRecord | undefined | ((prev: HermesConfigRecord | undefined) => HermesConfigRecord | undefined)

// Cache writer for optimistic write-through. The key is resolved at WRITE
// time, not when the writer is created, so a writer memoized by a long-lived
// settings panel keeps landing on the row of whichever gateway is active when
// the save happens — the same row its query reads.
export const hermesConfigCacheWriter =
  (profile?: ProfileScope) =>
  (next: ConfigCacheUpdate): void =>
    writeCache<HermesConfigRecord>(hermesConfigKey(profile))(next)

// setHermesConfigCache writes the ambient (active gateway + profile) record.
export const setHermesConfigCache = hermesConfigCacheWriter()

export const invalidateHermesConfig = (profile?: ProfileScope) =>
  queryClient.invalidateQueries({ queryKey: hermesConfigKey(profile) })
