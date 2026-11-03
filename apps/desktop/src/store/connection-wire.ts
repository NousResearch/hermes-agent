import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { map } from 'nanostores'

import type { GatewaySessionWire } from '@/api/client'
import type { HermesConnection } from '@/global'
import type { PrimaryGatewayRoute } from '@/store/gateway-primary-route'

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

/** The connection the primary socket serves. An established route without a registry id is the
 *  local source, never the foreground connection: a foregrounded secondary keeps `$connection`
 *  while the primary reconnects in the background. Only a route not yet recorded (both fields
 *  null, between a gateway swap and its descriptor) falls back to the window's connection. */
export function primaryWireConnectionId(route: PrimaryGatewayRoute, windowConnectionId: null | string | undefined): null | string | undefined {
  return route.mode || route.connectionId ? route.connectionId : windowConnectionId
}

/** Published from a secondary socket's `gateway.ready` under the connection it serves: its registry
 *  id, else the one its profile's descriptor resolved to (a profile's remote override resolves to
 *  that remote, not the local source). A resolved remote without a registry id has no key of its
 *  own, so it publishes nothing rather than overwrite the local source's wire. */
export function recordSecondaryWire(connectionId: null | string, resolved: null | Pick<HermesConnection, 'connectionId' | 'mode'>, wire: GatewaySessionWire | null): void {
  const key = connectionId || resolved?.connectionId || (resolved?.mode === 'local' ? LOCAL_CONNECTION_ID : null)

  if (key) { recordConnectionWire(key, wire) }
}
