import type { ModelOptionProvider, ModelOptionsResult } from '@hermes/shared'
import { hashKey, type QueryClient, type QueryKey, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import type { HermesGateway } from '@/hermes'
import { modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'

// 1 initial open fetch + up to 5 interval ticks at 2s (~10s total): bounded per open
// so a failing backend cannot poll forever, while never exhausting across opens in a
// long-lived session.
export const NOUS_PENDING_MAX_POLLS = 6

interface PollAnchor {
  baseline: number
  key: string
  wasOpen: boolean
}

export interface UseModelOptionsQueryOptions {
  enabled?: boolean
  explicitOnly?: boolean
  gateway?: HermesGateway
  ownerConnectionId?: null | string
  profile?: null | string
  request?: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  sessionId?: null | string
}

function isNousProvider(provider: ModelOptionProvider): boolean {
  return provider.slug === 'nous' && !provider.free_tier_row
}

export function hasNousProvider(data: ModelOptionsResult | undefined): boolean {
  return Boolean(data?.providers?.some(isNousProvider))
}

// Nous only: re-read on every open (cached list shows instantly, the re-read swaps in when
// it lands). The catalog / entitlement caches are short-lived (5m / 3m), and window-plan
// gauges can update after the initial boot fetch, so the 60s default staleTime would hide
// fresh Nous state until "Refresh models". Every other provider keeps the 60s cache.
export function nousRefetchOnMount(data: ModelOptionsResult | undefined): 'always' | true {
  return hasNousProvider(data) ? 'always' : true
}

// A cache-only open on a fresh or TTL-expired backend ships rows marked `pricing_pending`
// or `free_tier_pending` while prewarm runs in the background; re-read a few times per
// open so prices and window-plan labels appear without "Refresh models".
export function nousPricingPoll(
  data: ModelOptionsResult | undefined,
  attemptsSinceOpen: number
): false | number {
  return attemptsSinceOpen < NOUS_PENDING_MAX_POLLS &&
    data?.providers?.some(p => isNousProvider(p) && Boolean(p.pricing_pending || p.free_tier_pending))
    ? 2_000
    : false
}

function readTotalAttempts(client: QueryClient, queryKey: QueryKey): number {
  const state = client.getQueryState(queryKey)

  return (state?.dataUpdateCount ?? 0) + (state?.errorUpdateCount ?? 0)
}

/** Captures the query's cumulative attempt count (`dataUpdateCount + errorUpdateCount`) when
 *  a picker mounts, changes queryKey, or transitions from closed to open. Uses synchronous
 *  derived-state reset (avoiding a `useEffect` microtask race against `refetchOnMount`). */
export function useNousPendingPollAnchor(queryKey: QueryKey, open = true): number {
  const client = useQueryClient()
  const key = hashKey(queryKey)

  const [anchor, setAnchor] = useState<PollAnchor>(() => ({
    baseline: readTotalAttempts(client, queryKey),
    key,
    wasOpen: open
  }))

  if (anchor.key !== key || anchor.wasOpen !== open) {
    const next: PollAnchor = {
      baseline: open ? readTotalAttempts(client, queryKey) : anchor.baseline,
      key,
      wasOpen: open
    }

    setAnchor(next)

    return next.baseline
  }

  return anchor.baseline
}

/** Shared `model.options` query for interactive pickers (`ModelCatalogMenu`,
 *  `ModelPickerDialog`): keeps non-Nous providers on the 60s cache while re-reading
 *  Nous on open and polling pending pricing/entitlement replies on a per-open budget. */
export function useModelOptionsQuery({
  enabled = true,
  explicitOnly,
  gateway,
  ownerConnectionId,
  profile,
  request,
  sessionId
}: UseModelOptionsQueryOptions) {
  const queryKey = modelOptionsQueryKey(profile, sessionId, ownerConnectionId)
  const baseline = useNousPendingPollAnchor(queryKey, enabled)

  return useQuery({
    enabled,
    // Gateway-first even with no session: a connected (possibly remote)
    // gateway owns the model catalog, including virtual providers the local
    // REST fallback can't know about (#53817).
    queryFn: (): Promise<ModelOptionsResult> =>
      requestModelOptions({ explicitOnly, gateway, profile, request, sessionId }),
    queryKey,
    refetchInterval: query =>
      enabled
        ? nousPricingPoll(
            query.state.data,
            Math.max(0, query.state.dataUpdateCount + query.state.errorUpdateCount - baseline)
          )
        : false,
    refetchOnMount: query => nousRefetchOnMount(query.state.data),
    staleTime: query => (hasNousProvider(query.state.data) ? 0 : 60_000)
  })
}
