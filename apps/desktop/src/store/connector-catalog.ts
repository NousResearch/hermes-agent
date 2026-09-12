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
 * `available: false` (toolset off, signed out) and a failed request both
 * resolve to `null` rows — the caller decides what to show; there is no
 * fallback list here, because a fallback is how the drift started.
 */
import { useEffect, useState } from 'react'

import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import type { ConnectorRow } from '@/lib/connector-tools'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { assertSessionOwnerResolved } from '@/store/session-owner-resolution'
import { isSessionOwnerRoute } from '@/store/session-request-router'

export type ConnectorCatalog = { status: 'loading' } | { status: 'ready'; rows: ConnectorRow[] } | { status: 'unavailable' }

export function useConnectorCatalog(storedId: null | string, runtimeId: null | string): ConnectorCatalog {
  const [catalog, setCatalog] = useState<ConnectorCatalog>({ status: 'loading' })

  useEffect(() => {
    if (!storedId || !runtimeId) {
      return
    }

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
          { session_id: runtimeId }
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
