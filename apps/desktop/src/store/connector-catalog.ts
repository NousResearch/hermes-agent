/**
 * The live connector catalog for a chat session, read once per mount.
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
import { useEffect, useState } from 'react'

import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import { translateNow } from '@/i18n'
import type { ConnectorRow } from '@/lib/connector-tools'
import { isMissingRpcMethod, isOutOfSyncRpcParams } from '@/lib/gateway-rpc'
import { requestGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { assertSessionOwnerResolved } from '@/store/session-owner-resolution'
import { isSessionOwnerRoute } from '@/store/session-request-router'

export type ConnectorCatalog =
  { status: 'loading' } | { status: 'ready'; rows: ConnectorRow[] } | { status: 'unavailable' }

export function useConnectorCatalog(storedId: null | string, runtimeId: null | string): ConnectorCatalog {
  const [catalog, setCatalog] = useState<ConnectorCatalog>(() =>
    storedId && runtimeId ? { status: 'loading' } : { status: 'unavailable' }
  )

  useEffect(() => {
    if (!storedId || !runtimeId) {
      setCatalog({ status: 'unavailable' })

      return
    }

    setCatalog({ status: 'loading' })
    let cancelled = false
    const ambientProfile = $activeGatewayProfile.get()

    void resolveSessionOwner(storedId)
      .then(scope => {
        assertSessionOwnerResolved(scope, { method: 'connectors.list', sessionId: storedId })

        const connectionId = isSessionOwnerRoute(scope) ? scope.connectionId : null
        const profile = isSessionOwnerRoute(scope) ? scope.profile : scope || ambientProfile

        return requestGatewayForAgent<{ available: boolean; connectors: ConnectorRow[] }>(
          connectionId,
          profile,
          'connectors.list',
          { owner: { session_id: runtimeId, type: 'session' } },
          15000
        )
      })
      .then(response => {
        if (!cancelled) {
          setCatalog(response.available ? { rows: response.connectors, status: 'ready' } : { status: 'unavailable' })
        }
      })
      .catch(error => {
        if (!cancelled) {
          setCatalog({ status: 'unavailable' })

          if (isMissingRpcMethod(error) || isOutOfSyncRpcParams(error)) {
            notifyError(error, translateNow('connectors.unavailable'), { id: 'connectors-rpc-out-of-sync' })
          }
        }
      })

    return () => {
      cancelled = true
    }
  }, [storedId, runtimeId])

  return catalog
}
