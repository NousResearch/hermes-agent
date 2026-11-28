import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { map } from 'nanostores'

import type { GatewaySessionWire } from '@/api/client'
import type { HermesConnection } from '@/global'
import type { PrimaryGatewayRoute } from '@/store/gateway-primary-route'

/** Negotiated wires for connections and exact profile routes: a profile override may
 *  reach a different backend from the connection's primary socket. */
export const $connectionWires = map<Record<string, GatewaySessionWire>>({})

// A remote reached without a registry id (`HERMES_DESKTOP_REMOTE_URL`) is not the local source,
// which may run beside it as a secondary. It gets a key no registry id can take.
export const UNNAMED_REMOTE_WIRE_KEY = '\u0000unnamed-remote'

/** A connection as a socket or owner names it: its registry id, else what its mode says it is. */
export interface WireSource {
  connectionId?: null | string
  mode?: null | string
  profile?: null | string
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
  const key = wireKey(source)
  const profile = typeof source === 'object' && source ? source.profile : null
  const wires = $connectionWires.get()

  return (profile ? wires[JSON.stringify([key, profile])] : null) ?? wires[key] ?? null
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
export function recordSecondaryWire(connectionId: null | string, resolved: null | Pick<HermesConnection, 'connectionId' | 'mode'>, wire: GatewaySessionWire | null, profile?: string): void {
  const key = connectionId || resolved?.connectionId || (resolved?.mode === 'local' ? LOCAL_CONNECTION_ID : null)

  // This socket may serve a profile override rather than the connection's primary backend.
  if (key && profile && wire) { $connectionWires.setKey(JSON.stringify([key, profile]), wire) }

  // A profile's remote override (`connection.json` → `profiles.<name>.mode: remote`) is dialed
  // under the local scope, so both ids still say `local`. It is a different backend: recording
  // its wire there would flip the local source to `legacy` when the remote predates session
  // authority, sending local image attaches down the legacy `image.attach` RPC.
  if (resolved?.mode === 'remote' && key === LOCAL_CONNECTION_ID) { return }

  if (key) { recordConnectionWire(key, wire) }
}
