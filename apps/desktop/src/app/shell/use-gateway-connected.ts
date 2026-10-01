import { useEffect, useState } from 'react'

import type { HermesGateway } from '@/hermes'

/**
 * React bridge for the gateway's imperative connection-state emitter.
 *
 * Model catalog queries must not run while the socket is down: a rejected
 * `model.options` request is otherwise cached by React Query and the picker can
 * keep rendering the old "gateway is not connected" error after reconnect.
 * Toggling the query's `enabled` flag back to true on `open` gives React Query
 * a clean retry boundary.
 */
export function useGatewayConnected(gateway?: HermesGateway): boolean {
  const [connected, setConnected] = useState(() => !gateway || gateway.connectionState === 'open')

  useEffect(() => {
    if (!gateway) {
      setConnected(true)

      return undefined
    }

    return gateway.onState(state => setConnected(state === 'open'))
  }, [gateway])

  return connected
}
