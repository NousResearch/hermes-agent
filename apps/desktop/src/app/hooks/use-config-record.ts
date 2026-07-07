import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'

import { getHermesConfigRecord } from '@/hermes'
import { queryClient, writeCache } from '@/lib/query-client'
import { $activeGatewayProfile, activeProfileQueryKey, normalizeProfileKey } from '@/store/profile'
import type { HermesConfigRecord } from '@/types/hermes'

// One shared cache for the whole profile config record (`GET /api/config`).
// Every settings surface (MCP, model, config) reads and writes through this key
// so a save in one shows in the others, and revisiting a tab paints the cache
// instead of blanking on a fresh fetch.
//
// The key carries the ACTIVE GATEWAY PROFILE segment: the record is
// profile-scoped state served by whichever backend the gateway is on, so
// per-profile keys guarantee profile A's config can never paint under profile
// B, and hopping back to A serves A's warm cache instead of refetching.
//
// Distinct from session/hooks/use-hermes-config.ts, which is side-effecting —
// it pushes personality/cwd/voice/… into the session stores for live chat.
export const HERMES_CONFIG_KEY = ['hermes-config-record'] as const

export const hermesConfigKey = (profile: null | string | undefined) =>
  [...HERMES_CONFIG_KEY, normalizeProfileKey(profile)] as const

// staleTime 0 → serve cache instantly, background-revalidate on every mount.
export const useHermesConfigRecord = () => {
  const profileKey = normalizeProfileKey(useStore($activeGatewayProfile))

  return useQuery({ queryKey: hermesConfigKey(profileKey), queryFn: getHermesConfigRecord, staleTime: 0 })
}

// Imperative cache write/invalidate target the profile the gateway is on NOW —
// saves always go through the active backend, so that's the record they touched.
export const setHermesConfigCache = (
  next: HermesConfigRecord | undefined | ((prev: HermesConfigRecord | undefined) => HermesConfigRecord | undefined)
): void => writeCache<HermesConfigRecord>(hermesConfigKey(activeProfileQueryKey()))(next)

export const invalidateHermesConfig = () =>
  queryClient.invalidateQueries({ queryKey: hermesConfigKey(activeProfileQueryKey()) })
