import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { map } from 'nanostores'

import type { GatewaySessionWire } from '@/api/client'

/** The session wire each connection's gateway negotiated on its last `gateway.ready`. A connection
 *  is one backend, so every socket to it (the primary and its profile secondaries) agrees. */
export const $connectionWires = map<Record<string, GatewaySessionWire>>({})

const wireKey = (connectionId: null | string | undefined) => connectionId?.trim() || LOCAL_CONNECTION_ID

/** Published from each socket's `gateway.ready`, after the client latched the frame's wire. */
export function recordConnectionWire(connectionId: null | string | undefined, wire: GatewaySessionWire | null): void {
  const key = wireKey(connectionId)

  if (wire && $connectionWires.get()[key] !== wire) { $connectionWires.setKey(key, wire) }
}

/** Null until a socket to that connection has become ready. */
export function connectionWire(connectionId: null | string | undefined): GatewaySessionWire | null {
  return $connectionWires.get()[wireKey(connectionId)] ?? null
}
