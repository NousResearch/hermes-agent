import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import type { GatewayRequester } from '@/app/contrib/types'
import {
  $busyInputConfig,
  type BusyInputMode,
  busyInputOwnerKey,
  normalizeBusyInputMode
} from '@/store/busy-input-mode'
import { serverOwnsComposerQueue } from '@/store/composer-queue'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState, knownSessionOwner, ownerLookupSessionRows } from '@/store/session'
import { requestForSessionProfile } from '@/store/session-request-router'
import { sessionTileOwnerRoute } from '@/store/session-states'

export function useBusyInputMode({
  sessionId,
  storedSessionId,
  requestGateway
}: {
  sessionId: string | null
  storedSessionId: string | null
  requestGateway: GatewayRequester
}): BusyInputMode | null {
  const config = useStore($busyInputConfig)
  const connection = useStore($connection)
  const profile = useStore($activeGatewayProfile)
  const gatewayState = useStore($gatewayState)

  const route = storedSessionId
    ? (sessionTileOwnerRoute(storedSessionId) ?? knownSessionOwner(ownerLookupSessionRows(), storedSessionId))
    : undefined

  const connectionId = route && typeof route === 'object' ? route.connectionId : connection?.connectionId
  const ownerProfile = route && typeof route === 'object' ? route.profile : route || profile
  const targetProfile = route && typeof route === 'object' ? route.targetProfile : undefined
  const owner = busyInputOwnerKey(connectionId, targetProfile || ownerProfile)

  const [loaded, setLoaded] = useState<{ owner: string; connection: typeof connection; sessionId: string; mode: BusyInputMode } | null>(
    null
  )

  const canonical = serverOwnsComposerQueue(sessionId ?? storedSessionId)
  const configured = !canonical && config?.owner === owner && config.connection === connection ? config.mode : null

  useEffect(() => {
    if (configured !== null || !sessionId || gatewayState !== 'open') {
      return
    }

    let cancelled = false
    const ownerRoute = connectionId ? { connectionId, profile: ownerProfile, targetProfile } : ownerProfile
    void requestForSessionProfile<{ value?: unknown }>(ownerRoute, requestGateway, 'config.get', {
      key: 'busy',
      session_id: sessionId
    })
      .then(result => {
        if (!cancelled && result && Object.hasOwn(result, 'value')) {
          setLoaded({ owner, connection, sessionId, mode: normalizeBusyInputMode(result.value) })
        }
      })
      .catch(() => undefined)

    return () => {
      cancelled = true
    }
  }, [
    configured,
    connection,
    connectionId,
    gatewayState,
    owner,
    ownerProfile,
    requestGateway,
    sessionId,
    targetProfile
  ])

  if (gatewayState !== 'open') {
    return null
  }

  return configured ?? (loaded?.owner === owner && loaded.connection === connection && loaded.sessionId === sessionId ? loaded.mode : null)
}
