import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import type { DesktopConnectionsRegistry, DesktopRegistryConnection } from '@/global'
import { localBootstrapNeeded } from '@/lib/local-backend-probe'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { gatewayRouteState, openGatewayForAgent } from '@/store/gateway'
import { keepRoute } from '@/store/gateway-kept-routes'
import { $primaryGatewayRoute, type PrimaryGatewayRoute } from '@/store/gateway-primary-route'

// Every registered gateway stays connected for as long as the window lives: its
// home socket is dialed at startup, kept against the idle pruner, and redialed
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

/** The registry connections that need their own kept socket. */
function persistentGatewayIds(registry: DesktopConnectionsRegistry | null, primary: PrimaryGatewayRoute): string[] {
  return (registry?.connections ?? []).filter(connection => !servedByPrimary(connection, primary)).map(c => c.id)
}

/** Dialing This device where Hermes is not installed would start the installer
 *  unseen; only a confirmed click may do that. */
async function dial(connectionId: string, localInstalled: () => Promise<boolean>): Promise<void> {
  if (connectionId === LOCAL_CONNECTION_ID && !(await localInstalled())) {
    return
  }

  // A gateway that is down or needs sign-in fails this dial; the next tick
  // (every REDIAL_INTERVAL_MS) tries again.
  await openGatewayForAgent(connectionId, HOME_PROFILE)
}

/** Start keeping every registered gateway connected. Returns the teardown. */
export function startPersistentGateways(): () => void {
  const pins = new Map<string, () => void>()

  // This device's install verdict. Each probe re-resolves the backend in the
  // main process, so not installed is re-checked only when the registry or the
  // primary route changes; installed holds for the window.
  let localProbe: Promise<boolean> | null = null
  let localSeenInstalled = false

  const localInstalled = () => {
    if (!localProbe) {
      const probe: Promise<boolean> = localBootstrapNeeded().then(
        needed => {
          // A dropped probe answers for its own callers only: its late verdict
          // must not overwrite the one its replacement already recorded.
          if (localProbe === probe) {
            localSeenInstalled = !needed
          }

          return !needed
        },
        (error: unknown) => {
          // A late failure of a dropped probe must not clear its replacement.
          if (localProbe === probe) {
            localProbe = null
          }

          throw error
        }
      )

      localProbe = probe
    }

    return localProbe
  }

  const reconcile = () => {
    const primary = $primaryGatewayRoute.get()

    // Until the primary publishes its route (boot, or a primary teardown), the
    // primary's own gateway looks unserved and would get a duplicate socket.
    if (!primary.connectionId && !primary.mode) {
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
        void dial(connectionId, localInstalled).catch(() => undefined)
      }
    }
  }

  const recheckLocalAndReconcile = () => {
    if (!localSeenInstalled) {
      localProbe = null
    }

    reconcile()
  }

  // Every primary reconnect republishes the same route, which says nothing new
  // about This device; only a different route re-probes it.
  let primaryRoute = $primaryGatewayRoute.get()

  const onPrimaryRoute = (route: PrimaryGatewayRoute) => {
    const changed = route.connectionId !== primaryRoute.connectionId || route.mode !== primaryRoute.mode
    primaryRoute = route

    if (changed) {
      recheckLocalAndReconcile()
    } else {
      reconcile()
    }
  }

  const offRegistry = $connectionsRegistry.subscribe(recheckLocalAndReconcile)
  const offPrimary = $primaryGatewayRoute.listen(onPrimaryRoute)
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
