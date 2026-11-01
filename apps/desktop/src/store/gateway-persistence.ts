import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import type { DesktopConnectionsRegistry, DesktopRegistryConnection } from '@/global'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { liveSecondaryConnectionIds, openGatewayForAgent } from '@/store/gateway'
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
export function persistentGatewayIds(
  registry: DesktopConnectionsRegistry | null,
  primary: PrimaryGatewayRoute
): string[] {
  return (registry?.connections ?? []).filter(connection => !servedByPrimary(connection, primary)).map(c => c.id)
}

/** Dialing This device where Hermes is not installed would start the installer
 *  unseen; only a confirmed click may do that. A missing probe fails closed. */
async function localInstalled(): Promise<boolean> {
  const probe = window.hermesDesktop?.probeLocalBackend

  return probe ? (await probe()).bootstrapNeeded !== true : false
}

async function dial(connectionId: string): Promise<void> {
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
        pins.set(connectionId, keepRoute(connectionId, HOME_PROFILE))
      }

      if (!live.has(connectionId)) {
        void dial(connectionId).catch(() => undefined)
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
