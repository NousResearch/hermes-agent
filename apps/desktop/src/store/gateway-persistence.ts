import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import type { DesktopConnectionsRegistry, DesktopRegistryConnection } from '@/global'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { liveSecondaryConnectionIds, openGatewayForAgent, retainGatewayForRelay } from '@/store/gateway'
import { $primaryGatewayRoute, type PrimaryGatewayRoute } from '@/store/gateway-primary-route'

// Every registered gateway stays connected for as long as the window lives: its
// home socket is dialed at startup, pinned against the idle pruner, and redialed
// when it drops. Picking a gateway or opening one of its sessions then costs no
// dial, and the unified Sessions list (Electron splices every pooled backend's
// rows) shows every gateway at once instead of only the one in front.
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

/** The registry connections that need their own pinned socket. */
export function persistentGatewayIds(
  registry: DesktopConnectionsRegistry | null,
  primary: PrimaryGatewayRoute
): string[] {
  return (registry?.connections ?? []).filter(connection => !servedByPrimary(connection, primary)).map(c => c.id)
}

const connectedListeners = new Set<() => void>()

/** Fires each time a kept gateway (re)connects, so lists drawn before it was
 *  reachable re-pull its rows. Returns the unsubscribe. */
export function onKeptGatewayConnected(handler: () => void): () => void {
  connectedListeners.add(handler)

  return () => connectedListeners.delete(handler)
}

function dial(connectionId: string): void {
  // Background priority: a gateway that is down or needs sign-in fails fast and
  // stays throttled; the next tick (or a focus/online nudge) tries again.
  void openGatewayForAgent(connectionId, HOME_PROFILE).then(
    () => connectedListeners.forEach(handler => handler()),
    () => undefined
  )
}

/** Start keeping every registered gateway connected. Returns the teardown. */
export function startPersistentGateways(): () => void {
  const pins = new Map<string, () => void>()

  const reconcile = () => {
    const wanted = new Set(persistentGatewayIds($connectionsRegistry.get(), $primaryGatewayRoute.get()))

    for (const [connectionId, release] of pins) {
      if (!wanted.has(connectionId)) {
        pins.delete(connectionId)
        release()
      }
    }

    const live = liveSecondaryConnectionIds()

    for (const connectionId of wanted) {
      if (!pins.has(connectionId)) {
        pins.set(connectionId, retainGatewayForRelay(connectionId, HOME_PROFILE, { includeLocal: true }))
      }

      if (!live.has(connectionId)) {
        dial(connectionId)
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
