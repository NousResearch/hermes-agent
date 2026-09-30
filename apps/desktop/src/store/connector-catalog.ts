/**
 * The live connector catalog for a chat session, read once per session and cached.
 *
 * The onboarding picker used to be a hardcoded list, and it drifted from the
 * deployed catalog: it offered apps the gateway does not carry and spelled
 * others with hyphens the gateway does not use. The build chat then had to
 * tell the user the pick could not be connected. This hook asks the gateway
 * what is there, through the account-owned RPC the Connectors page uses, on
 * the backend that owns the session, so the picker can only offer what can be
 * connected. The session-owned list answers for the session's own tools, and
 * the setup profile carries only its beat tools, so it always said no.
 *
 * `available: false` (toolset off, signed out), a failed request, and a
 * request that takes longer than 15 s all resolve to `unavailable`; the caller
 * decides what to show. There is no fallback list here, because a fallback is
 * how the drift started. Missing session ids resolve to `unavailable` too,
 * and the probe runs once they arrive, so the card never waits on a request
 * that was never sent.
 */
import type { ConnectorRow } from '@hermes/shared'
import { useQuery } from '@tanstack/react-query'

import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import { translateNow } from '@/i18n'
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
async function readConnectorCatalog(storedId: string): Promise<ConnectorCatalog> {
  try {
    const scope = await resolveSessionOwner(storedId)
    assertSessionOwnerResolved(scope, { method: 'connectors.list', sessionId: storedId })
    const connectionId = isSessionOwnerRoute(scope) ? scope.connectionId : null
    const profile = isSessionOwnerRoute(scope) ? scope.profile : scope || $activeGatewayProfile.get()

    const response = await requestGatewayForAgent<{ available: boolean; connectors: ConnectorRow[] }>(
      connectionId,
      profile,
      'connectors.list',
      { owner: { type: 'account' } },
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

const catalogKey = (storedId: null | string) => ['onboarding', 'connectors.list', storedId] as const

/** Start the read when the guide session opens: cold, it takes several seconds, and the card that needs it is
 *  two turns away. */
export function prefetchConnectorCatalog(storedId: string): void {
  void queryClient.prefetchQuery({
    queryFn: () => readConnectorCatalog(storedId),
    queryKey: catalogKey(storedId),
    staleTime: Infinity
  })
}

export function useConnectorCatalog(storedId: null | string): ConnectorCatalog {
  const query = useQuery({
    enabled: Boolean(storedId),
    queryFn: () => readConnectorCatalog(storedId!),
    queryKey: catalogKey(storedId),
    staleTime: Infinity
  })

  if (!storedId) {
    return { status: 'unavailable' }
  }

  return query.data ?? { status: 'loading' }
}
