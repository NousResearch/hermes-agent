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
// A missing This device install is re-probed this often, so one installed
// while the window stays open still joins the kept gateways.
const LOCAL_NOT_INSTALLED_RECHECK_MS = 10 * 60_000

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

/** A published route names the gateway the primary socket serves. Until one is
 *  published (boot, or a primary teardown), the primary's own gateway looks
 *  unserved and would get a duplicate socket. */
function primaryPublished(primary: PrimaryGatewayRoute): boolean {
  return Boolean(primary.connectionId || primary.mode)
}

/** Dialing This device where Unitalk is not installed would start the installer
 *  unseen; only a confirmed click may do that. */
async function dial(
  connectionId: string,
  localInstalled: () => Promise<boolean>,
  stillWanted: () => boolean
): Promise<void> {
  if (connectionId === LOCAL_CONNECTION_ID && !((await localInstalled()) && stillWanted())) {
    return
  }

  // A gateway that is down or needs sign-in fails this dial; the next tick
  // (every REDIAL_INTERVAL_MS) tries again.
  await openGatewayForAgent(connectionId, HOME_PROFILE)
}

/** Start keeping every registered gateway connected. Returns the teardown. */
export function startPersistentGateways(): () => void {
  const pins = new Map<string, () => void>()

  // This device's install verdict. "Installed" is re-checked before every
  // dial (an install can go away while the window lives, and dialing it then
  // would start the installer), while "not installed" is kept until the
  // registry or the primary route changes or LOCAL_NOT_INSTALLED_RECHECK_MS
  // passes, so a missing install is not re-probed every tick. Concurrent dials
  // share one probe in flight.
  let localProbe: Promise<boolean> | null = null
  let localProbeExpiresAt = Infinity
  // Bumped whenever a registry or primary-route change drops the probe: a dial
  // that awaited a dropped probe must not act on its answer.
  let localGeneration = 0

  const localInstalled = () => {
    if (localProbe && Date.now() >= localProbeExpiresAt) {
      localProbe = null
    }

    if (!localProbe) {
      const probe: Promise<boolean> = localBootstrapNeeded().then(
        needed => {
          // A dropped probe's answer is for its own callers only.
          if (localProbe === probe) {
            if (needed) {
              localProbeExpiresAt = Date.now() + LOCAL_NOT_INSTALLED_RECHECK_MS
            } else {
              localProbe = null
            }
          }

          return !needed
        },
        (error: unknown) => {
          if (localProbe === probe) {
            localProbe = null
          }

          throw error
        }
      )

      localProbe = probe
      localProbeExpiresAt = Infinity
    }

    return localProbe
  }

  // The probe takes time: by its answer the keeper may have stopped, released
  // this route, lost its published primary, or dropped the probe for a newer one.
  const stillWanted = (connectionId: string, generation: number) =>
    generation === localGeneration && pins.has(connectionId) && primaryPublished($primaryGatewayRoute.get())

  const reconcile = () => {
    const primary = $primaryGatewayRoute.get()

    if (!primaryPublished(primary)) {
      return
    }

    const wanted = new Set(persistentGatewayIds($connectionsRegistry.get(), primary))
    const generation = localGeneration

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
        void dial(connectionId, localInstalled, () => stillWanted(connectionId, generation)).catch(() => undefined)
      }
    }
  }

  const recheckLocalAndReconcile = () => {
    localProbe = null
    localGeneration += 1
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
