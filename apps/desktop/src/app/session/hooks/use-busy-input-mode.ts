import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import type { GatewayRequester } from '@/app/contrib/types'
import {
  $busyInputConfig,
  type BusyInputMode,
  busyInputOwnerKey,
  normalizeBusyInputMode
} from '@/store/busy-input-mode'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState, knownSessionOwner, ownerLookupSessionRows } from '@/store/session'
import { sessionTileOwnerRoute } from '@/store/session-states'

// Main sessions reuse the existing full-config snapshot. A session on another
// backend initializes its own value through the established session router.
// Neither path does network I/O on Send; no value is persisted to disk.
export function useBusyInputMode({
  sessionId,
  storedSessionId,
  requestGateway
}: {
  sessionId: string | null
  storedSessionId: string | null
  requestGateway: GatewayRequester
}): (targetId: string) => BusyInputMode | null {
  const config = useStore($busyInputConfig)
  const connection = useStore($connection)
  const profile = useStore($activeGatewayProfile)
  const gatewayState = useStore($gatewayState)

  const route = storedSessionId
    ? (sessionTileOwnerRoute(storedSessionId) ?? knownSessionOwner(ownerLookupSessionRows(), storedSessionId))
    : undefined

  const owner = busyInputOwnerKey(
    route && typeof route === 'object' ? route.connectionId : connection?.connectionId,
    route && typeof route === 'object' ? route.targetProfile || route.profile : route || profile
  )

  const [loaded, setLoaded] = useState<{ owner: string; connection: typeof connection; mode: BusyInputMode } | null>(
    null
  )

  const configured = config?.owner === owner && config.connection === connection ? config.mode : null

  useEffect(() => {
    if (configured !== null || !sessionId || gatewayState !== 'open') {
      return
    }

    let cancelled = false
    void requestGateway<{ value?: unknown }>('config.get', { key: 'busy', session_id: sessionId })
      .then(result => {
        if (!cancelled && result && Object.prototype.hasOwnProperty.call(result, 'value')) {
          setLoaded({ owner, connection, mode: normalizeBusyInputMode(result.value) })
        }
      })
      .catch(() => undefined)

    return () => {
      cancelled = true
    }
  }, [configured, config, connection, gatewayState, owner, requestGateway, sessionId])

  return targetId => {
    if (
      targetId !== sessionId ||
      gatewayState !== 'open' ||
      $connection.get() !== connection ||
      $activeGatewayProfile.get() !== profile
    ) {
      return null
    }

    return configured ?? (loaded?.owner === owner && loaded.connection === connection ? loaded.mode : null)
  }
}
