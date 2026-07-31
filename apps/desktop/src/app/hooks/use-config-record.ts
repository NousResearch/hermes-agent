import { useQuery } from '@tanstack/react-query'

import { getHermesConfigRecord } from '@/hermes'
import { HERMES_CONFIG_QUERY_KEY, queryClient, writeCache } from '@/lib/query-client'
import type { HermesConfigRecord } from '@/types/hermes'

// One shared cache for the whole profile config record (`GET /api/config`).
// Every settings surface (MCP, model, config) reads and writes through this key
// so a save in one shows in the others, and revisiting a tab paints the cache
// instead of blanking on a fresh fetch.
//
// The key itself lives in lib/query-client.ts: on a profile/gateway switch the
// store-level boundary (invalidateProfileScopedQueries) hard-resets this record
// so no consumer can seed a draft from the previous profile's data.
//
// Distinct from session/hooks/use-hermes-config.ts, which is side-effecting —
// it pushes personality/cwd/voice/… into the session stores for live chat.
export const HERMES_CONFIG_KEY = HERMES_CONFIG_QUERY_KEY

// staleTime 0 → serve cache instantly, background-revalidate on every mount.
export const useHermesConfigRecord = () =>
  useQuery({ queryKey: HERMES_CONFIG_KEY, queryFn: getHermesConfigRecord, staleTime: 0 })

export const setHermesConfigCache = writeCache<HermesConfigRecord>(HERMES_CONFIG_KEY)

export const invalidateHermesConfig = () => queryClient.invalidateQueries({ queryKey: HERMES_CONFIG_KEY })
