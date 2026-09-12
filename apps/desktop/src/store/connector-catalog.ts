/**
 * The live connector catalog for a chat session, read once per mount.
 *
 * The onboarding picker used to be a hardcoded list, and it drifted from the
 * deployed catalog: it offered apps the gateway does not carry and spelled
 * others with hyphens the gateway does not use. The pick was then a promise
 * the build chat had to walk back. This hook asks the gateway what is
 * actually there, through the same session-owned RPC the connector cards
 * use, so the picker can only ever offer what can be connected.
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
import type { ConnectorRow } from '@/lib/connector-tools'
import { requestGatewayForAgent } from '@/store/gateway'
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
          { session_id: runtimeId },
          15000
        )
      })
      .then(response => {
        if (!cancelled) {
          setCatalog(response.available ? { rows: response.connectors, status: 'ready' } : { status: 'unavailable' })
        }
      })
      .catch(() => {
        if (!cancelled) {
          setCatalog({ status: 'unavailable' })
        }
      })

    return () => {
      cancelled = true
    }
  }, [storedId, runtimeId])

  return catalog
}
