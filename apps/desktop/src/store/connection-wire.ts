import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { map } from 'nanostores'

import type { GatewaySessionWire } from '@/api/client'
import type { HermesConnection } from '@/global'
import type { PrimaryGatewayRoute } from '@/store/gateway-primary-route'

/** The session wire each connection's gateway negotiated on its last `gateway.ready`. A connection
 *  is one backend, so every socket to it (the primary and its profile secondaries) agrees. */
export const $connectionWires = map<Record<string, GatewaySessionWire>>({})

// A remote reached without a registry id (`HERMES_DESKTOP_REMOTE_URL`) is not the local source,
// which may run beside it as a secondary. It gets a key no registry id can take.
export const UNNAMED_REMOTE_WIRE_KEY = '\u0000unnamed-remote'

/** A connection as a socket or owner names it: its registry id, else what its mode says it is. */
export interface WireSource {
  connectionId?: null | string
  mode?: null | string
}

function wireKey(source: null | string | undefined | WireSource): string {
  const id = typeof source === 'object' && source ? source.connectionId : source
  const named = id?.trim()

  if (named) { return named }

  return typeof source === 'object' && source?.mode === 'remote' ? UNNAMED_REMOTE_WIRE_KEY : LOCAL_CONNECTION_ID
}

/** Published from each socket's `gateway.ready`, after the client latched the frame's wire. */
export function recordConnectionWire(source: null | string | undefined | WireSource, wire: GatewaySessionWire | null): void {
  const key = wireKey(source)

  if (wire && $connectionWires.get()[key] !== wire) { $connectionWires.setKey(key, wire) }
}

/** Null until a socket to that connection has become ready. */
export function connectionWire(source: null | string | undefined | WireSource): GatewaySessionWire | null {
  return $connectionWires.get()[wireKey(source)] ?? null
}

/** The connection the primary socket serves. An established route speaks for itself, even without
 *  a registry id (the local source, or an unnamed remote), never for the foreground connection: a
 *  foregrounded secondary keeps `$connection` while the primary reconnects in the background. Only
 *  a route not yet recorded (both fields null, between a gateway swap and its descriptor) falls
 *  back to the window's connection. */
export function primaryWireSource(route: PrimaryGatewayRoute, windowConnection: null | undefined | WireSource): null | undefined | WireSource {
  return route.mode || route.connectionId ? route : windowConnection
}

/** Published from a secondary socket's `gateway.ready` under the connection it serves: its registry
 *  id, else the one its profile's descriptor resolved to (a profile's remote override resolves to
 *  that remote, not the local source). A resolved remote without a registry id has no key of its
 *  own, so it publishes nothing rather than overwrite the local source's wire. */
export function recordSecondaryWire(connectionId: null | string, resolved: null | Pick<HermesConnection, 'connectionId' | 'mode'>, wire: GatewaySessionWire | null): void {
  const key = connectionId || resolved?.connectionId || (resolved?.mode === 'local' ? LOCAL_CONNECTION_ID : null)

  if (key) { recordConnectionWire(key, wire) }
}
