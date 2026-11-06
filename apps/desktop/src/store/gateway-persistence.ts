import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import type { DesktopConnectionsRegistry, DesktopRegistryConnection } from '@/global'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { gatewayRouteState, openGatewayForAgent } from '@/store/gateway'
import { keepRoute } from '@/store/gateway-kept-routes'
import { $primaryGatewayRoute, type PrimaryGatewayRoute } from '@/store/gateway-primary-route'

// Keep explicitly registered remote gateways connected for the window lifetime.
// This device is an implicit entry: a remote-primary Desktop must not create
// a local gateway until the user actually selects This device.
const HOME_PROFILE = 'default'
const REDIAL_INTERVAL_MS = 30_000

/** The window's primary socket already holds this gateway open; a second socket
 *  to the same backend would duplicate its event stream. */
function servedByPrimary(connection: DesktopRegistryConnection, primary: PrimaryGatewayRoute): boolean {
  if (primary.connectionId) {
    return primary.connectionId === connection.id
  }

  return connection.id === LOCAL_CONNECTION_ID && primary.mode !== 'remote'
}

/** The non-local registry connections that need their own kept socket. */
function persistentGatewayIds(registry: DesktopConnectionsRegistry | null, primary: PrimaryGatewayRoute): string[] {
  return (registry?.connections ?? [])
    .filter(connection => connection.id !== LOCAL_CONNECTION_ID && !servedByPrimary(connection, primary))
    .map(connection => connection.id)
}

/** A published route names the gateway the primary socket serves. Until one is
 *  published (boot, or a primary teardown), the primary's own gateway looks
 *  unserved and would get a duplicate socket. */
function primaryPublished(primary: PrimaryGatewayRoute): boolean {
  return Boolean(primary.connectionId || primary.mode)
}

/** Start keeping non-local registered gateways connected. Returns the teardown. */
export function startPersistentGateways(): () => void {
  const pins = new Map<string, () => void>()

  const reconcile = () => {
    const primary = $primaryGatewayRoute.get()

    if (!primaryPublished(primary)) {
      return
    }

    const wanted = new Set(persistentGatewayIds($connectionsRegistry.get(), primary))

    for (const [connectionId, release] of pins) {
      if (!wanted.has(connectionId)) {
        pins.delete(connectionId)
        release()
      }
    }

    for (const connectionId of wanted) {
      if (!pins.has(connectionId)) {
        pins.set(connectionId, keepRoute(connectionId, HOME_PROFILE))
      }

      // The kept route itself: another profile's open socket on the same
      // gateway does not carry this one.
      if (!gatewayRouteState(connectionId, HOME_PROFILE).open) {
        void openGatewayForAgent(connectionId, HOME_PROFILE).catch(() => undefined)
      }
    }
  }

  const offRegistry = $connectionsRegistry.subscribe(reconcile)
  const offPrimary = $primaryGatewayRoute.listen(reconcile)
  const timer = setInterval(reconcile, REDIAL_INTERVAL_MS)

  return () => {
    clearInterval(timer)
    offRegistry()
    offPrimary()

    for (const release of pins.values()) {
      release()
    }

    pins.clear()
  }
}

let stopKeeping: (() => void) | null = null

/** The window-lifetime keeper: starts once, later calls are no-ops. */
export function keepGatewaysConnected(): void {
  stopKeeping ??= startPersistentGateways()
}

import.meta.hot?.dispose(() => {
  stopKeeping?.()
  stopKeeping = null
})
