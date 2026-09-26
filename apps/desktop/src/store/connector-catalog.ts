/**
 * The live connector catalog for a chat session, read once per session and cached.
 *
 * The onboarding picker used to be a hardcoded list, and it drifted from the
 * deployed catalog: it offered apps the gateway does not carry and spelled
 * others with hyphens the gateway does not use. The build chat then had to
 * tell the user the pick could not be connected. This hook asks the gateway
 * what is there, through the same session-owned RPC the connector cards use,
 * so the picker can only offer what can be connected.
 *
 * `available: false` (toolset off, signed out), a failed request, and a
 * request that takes longer than 15 s all resolve to `unavailable`; the caller
 * decides what to show. There is no fallback list here, because a fallback is
 * how the drift started. Missing session ids resolve to `unavailable` too,
 * and the probe runs once they arrive, so the card never waits on a request
 * that was never sent.
 */
import { useQuery } from '@tanstack/react-query'

import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import { translateNow } from '@/i18n'
import type { ConnectorRow } from '@/lib/connector-tools'
import { isMissingRpcMethod, isOutOfSyncRpcParams } from '@/lib/gateway-rpc'
import { queryClient } from '@/lib/query-client'
import { requestGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { assertSessionOwnerResolved } from '@/store/session-owner-resolution'
import { isSessionOwnerRoute } from '@/store/session-request-router'

export type ConnectorCatalog =
  { status: 'loading' } | { status: 'ready'; rows: ConnectorRow[] } | { status: 'unavailable' }

/** The picks card remounts on every transcript rebuild (each hidden submit, each turn end), so the read is
 *  held in the query cache per session: a remount paints the rows it already has instead of flashing back to
 *  the skeleton for another round trip. */
async function readConnectorCatalog(storedId: string, runtimeId: string): Promise<ConnectorCatalog> {
  try {
    const scope = await resolveSessionOwner(storedId)
    assertSessionOwnerResolved(scope, { method: 'connectors.list', sessionId: storedId })
    const connectionId = isSessionOwnerRoute(scope) ? scope.connectionId : null
    const profile = isSessionOwnerRoute(scope) ? scope.profile : scope || $activeGatewayProfile.get()

    const response = await requestGatewayForAgent<{ available: boolean; connectors: ConnectorRow[] }>(
      connectionId,
      profile,
      'connectors.list',
      { owner: { session_id: runtimeId, type: 'session' } },
      15000
    )

    return response.available ? { rows: response.connectors, status: 'ready' } : { status: 'unavailable' }
  } catch (error) {
    if (isMissingRpcMethod(error) || isOutOfSyncRpcParams(error instanceof Error ? error : String(error))) {
      notifyError(error, translateNow('connectors.unavailable'), { id: 'connectors-rpc-out-of-sync' })
    }

    return { status: 'unavailable' }
  }
}

const catalogKey = (storedId: null | string, runtimeId: null | string) =>
  ['onboarding', 'connectors.list', storedId, runtimeId] as const

/** Start the read when the guide session opens: cold, it takes several seconds, and the card that needs it is
 *  two turns away. */
export function prefetchConnectorCatalog(storedId: string, runtimeId: string): void {
  void queryClient.prefetchQuery({
    queryFn: () => readConnectorCatalog(storedId, runtimeId),
    queryKey: catalogKey(storedId, runtimeId),
    staleTime: Infinity
  })
}

export function useConnectorCatalog(storedId: null | string, runtimeId: null | string): ConnectorCatalog {
  const query = useQuery({
    enabled: Boolean(storedId && runtimeId),
    queryFn: () => readConnectorCatalog(storedId!, runtimeId!),
    queryKey: catalogKey(storedId, runtimeId),
    staleTime: Infinity
  })

  if (!storedId || !runtimeId) {
    return { status: 'unavailable' }
  }

  return query.data ?? { status: 'loading' }
}
