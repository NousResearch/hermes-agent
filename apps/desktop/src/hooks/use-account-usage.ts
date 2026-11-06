import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useMemo, useSyncExternalStore } from 'react'

import type { HermesGateway } from '@/hermes'
import { isMissingRpcMethod } from '@/lib/gateway-rpc'
import { useStoresSelector } from '@/lib/use-session-slice'
import { $gateway, gatewayRouteState } from '@/store/gateway'
import { $gatewayActivationEpoch } from '@/store/gateway-activation-epoch'
import { $primaryGatewayRoute } from '@/store/gateway-primary-route'
import { $cronSessions, $gatewayState, $messagingSessions, $sessions, $unlistedSessionOwnerRows } from '@/store/session'
import { requestForSessionProfile, type SessionOwnerScope } from '@/store/session-request-router'
import { $sessionStates, $sessionTiles, knownOwnerForSession } from '@/store/session-states'
import type { AccountUsageResponse, AccountUsageSnapshot } from '@/types/account-usage'

export const ACCOUNT_USAGE_REFRESH_MS = 3 * 60_000
export const ACCOUNT_USAGE_BACKOFF_MS = 15 * 60_000
export const ACCOUNT_USAGE_REQUEST_TIMEOUT_MS = 45_000

export interface AccountUsageOptions {
  enabled: boolean
  model: string
  provider: string
  requestGateway: <T = unknown>(
    method: string,
    params?: Record<string, unknown>,
    timeoutMs?: number,
    signal?: AbortSignal
  ) => Promise<T>
  sessionId: null | string
}

interface UsageReading {
  failures: number
  nextPollAt: number
  snapshot: AccountUsageSnapshot | null
  status: AccountUsageResponse['status']
}

interface TransportIdentity {
  generation: number
  id: number
  observed: boolean
}

const transports = new WeakMap<HermesGateway, TransportIdentity>()
let nextTransportId = 0

const OWNER_STORES = [
  $sessionStates,
  $sessionTiles,
  $sessions,
  $cronSessions,
  $messagingSessions,
  $unlistedSessionOwnerRows
]

const ROUTE_STORES = [
  $gateway,
  $gatewayState,
  $gatewayActivationEpoch,
  $primaryGatewayRoute,
  $sessionStates,
  $sessionTiles
]

const SUPPORTED_PROVIDERS = new Set(['openrouter', 'openai-codex', 'anthropic'])

/** A profile-only owner follows the primary profile door, never the foreground connection. */
function ownerIdentity(owner: SessionOwnerScope): string {
  if (typeof owner === 'string') {
    return JSON.stringify(owner.trim() || null)
  }

  const connectionId = owner?.connectionId?.trim()

  if (!owner || !connectionId) {
    return 'null'
  }

  return JSON.stringify({
    connectionId,
    profile: owner.profile.trim() || 'default',
    targetProfile: owner.targetProfile?.trim() || undefined
  })
}

function transportIdentity(gateway: HermesGateway): TransportIdentity {
  let identity = transports.get(gateway)

  if (!identity) {
    identity = { generation: 0, id: ++nextTransportId, observed: false }
    transports.set(gateway, identity)
  }

  return identity
}

function ownerTransport(owner: SessionOwnerScope) {
  if (!owner) {
    return null
  }

  return gatewayRouteState(
    typeof owner === 'string' ? null : owner.connectionId,
    typeof owner === 'string' ? owner : owner.profile
  )
}

function transportSnapshot(owner: SessionOwnerScope): string {
  const route = ownerTransport(owner)
  const gateway = route?.gateway

  if (!gateway) {
    return ''
  }

  const identity = transportIdentity(gateway)

  return JSON.stringify([
    route.key,
    typeof owner === 'string' ? $primaryGatewayRoute.get() : null,
    identity.id,
    identity.generation
  ])
}

/** Track the owning socket through reconnects without foregrounding or dialing it. */
function useOwnerTransport(owner: SessionOwnerScope): string {
  const subscribe = useCallback(
    (notify: () => void) => {
      if (!owner) {
        return () => {}
      }

      let watched: HermesGateway | null = null

      let offSocket = () => {}

      const update = () => {
        const gateway = ownerTransport(owner)?.gateway

        if (gateway && watched !== gateway) {
          offSocket()
          watched = gateway
          const identity = transportIdentity(gateway)

          if (!identity.observed) {
            identity.observed = true
            // * A weak, socket-owned listener keeps cached capability verdicts invalid across unmounts.
            gateway.onState(() => {
              identity.generation += 1
            })
          }

          // * Keep observing a closed socket so its next open can wake a background tile.
          offSocket = gateway.onState(update)
        }

        notify()
      }

      const stops = ROUTE_STORES.map(store => store.listen(update))
      update()

      return () => {
        offSocket()
        stops.forEach(stop => stop())
      }
    },
    [owner]
  )

  return useSyncExternalStore(
    subscribe,
    () => transportSnapshot(owner),
    () => ''
  )
}

function pollingAllowed(reading: UsageReading | undefined): boolean {
  return reading?.status !== 'unsupported' && (!reading || Date.now() >= reading.nextPollAt)
}

/** Query state carries the failure streak so switching away or remounting cannot reset backoff. */
export function useAccountUsage(options: AccountUsageOptions) {
  const queryClient = useQueryClient()
  const provider = options.provider.trim().toLowerCase()
  const supportedProvider = SUPPORTED_PROVIDERS.has(provider)
  const ownerKey = useStoresSelector(OWNER_STORES, () => ownerIdentity(knownOwnerForSession(options.sessionId)))
  const owner: SessionOwnerScope = useMemo(() => JSON.parse(ownerKey) as SessionOwnerScope, [ownerKey])
  const transport = useOwnerTransport(options.enabled && supportedProvider ? owner : null)
  const enabled = options.enabled && supportedProvider && Boolean(options.sessionId && owner && transport)
  const queryKey = ['account-usage', ownerKey, transport, options.sessionId, provider, options.model] as const

  const query = useQuery<UsageReading>({
    enabled: state => enabled && state.state.data?.status !== 'unsupported',
    gcTime: ACCOUNT_USAGE_BACKOFF_MS,
    queryFn: async ({ signal }) => {
      const failures = (queryClient.getQueryData<UsageReading>(queryKey)?.failures ?? 0) + 1

      const unavailable = (): UsageReading => ({
        failures,
        nextPollAt: Date.now() + (failures >= 3 ? ACCOUNT_USAGE_BACKOFF_MS : ACCOUNT_USAGE_REFRESH_MS),
        snapshot: null,
        status: 'unavailable'
      })

      if (!owner || !options.sessionId || transportSnapshot(owner) !== transport) {
        return unavailable()
      }

      try {
        const response = await requestForSessionProfile<AccountUsageResponse>(
          owner,
          options.requestGateway,
          'session.account_usage',
          { session_id: options.sessionId, profile: typeof owner === 'string' ? owner : owner.profile },
          ACCOUNT_USAGE_REQUEST_TIMEOUT_MS,
          signal
        )

        if (signal.aborted) {
          throw new DOMException('Account usage request cancelled', 'AbortError')
        }

        if (
          transportSnapshot(owner) !== transport ||
          ownerIdentity(knownOwnerForSession(options.sessionId)) !== ownerKey
        ) {
          return unavailable()
        }

        if (response?.status === 'unsupported') {
          return { failures: 0, nextPollAt: 0, snapshot: null, status: 'unsupported' }
        }

        if (response?.status !== 'ok' || response.snapshot?.provider !== provider) {
          return unavailable()
        }

        return {
          failures: 0,
          nextPollAt: Date.now() + ACCOUNT_USAGE_REFRESH_MS,
          snapshot: response.snapshot,
          status: 'ok'
        }
      } catch (error) {
        if (signal.aborted) {
          throw error
        }

        return isMissingRpcMethod(error)
          ? { failures: 0, nextPollAt: 0, snapshot: null, status: 'unsupported' }
          : unavailable()
      }
    },
    queryKey,
    refetchInterval: state => {
      if (state.state.fetchStatus === 'fetching' || state.state.data?.status === 'unsupported') {
        return false
      }

      return (state.state.data?.failures ?? 0) >= 3 ? ACCOUNT_USAGE_BACKOFF_MS : ACCOUNT_USAGE_REFRESH_MS
    },
    refetchIntervalInBackground: false,
    refetchOnMount: state => pollingAllowed(state.state.data),
    refetchOnReconnect: false,
    refetchOnWindowFocus: state => pollingAllowed(state.state.data),
    retry: false,
    staleTime: state => ((state.state.data?.failures ?? 0) >= 3 ? ACCOUNT_USAGE_BACKOFF_MS : ACCOUNT_USAGE_REFRESH_MS)
  })

  return {
    available: enabled && query.data?.status !== 'unsupported',
    error: query.isError || query.data?.status === 'unavailable',
    loading: query.isFetching,
    refresh: query.refetch,
    snapshot: enabled && !query.isError && query.data?.status === 'ok' ? query.data.snapshot : null,
    supportedProvider
  }
}
