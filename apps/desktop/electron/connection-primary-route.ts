/**
 * The v1 `connection.json` side of a registry primary switch.
 *
 * Boot resolves `connection.json`'s app-global route BEFORE the v2 registry primary
 * (`resolveDesktopRemoteRoute`), so the two files must be kept in step or the stale one wins:
 * promoting This device while a global remote route is still on disk leaves every launch
 * dialing the gateway the user just promoted away from — the primary transport stays remote,
 * `globalRemoteActive()` stays true, and each profile that does not exist locally is refused
 * on the forced-local branch (`Profile "<name>" no longer exists.`).
 *
 * Removal already resets that route (`forgetRemovedGlobalRoute`), and Settings → Gateway Apply
 * writes v1 and imports it into the registry (`registerGlobalRoute`). A promotion of This
 * device never wrote v1 at all, and `reconcileRegistryDrift` deliberately leaves the
 * "route registered, primary names another source" case alone, so this module owns the
 * missing direction: registry → `connection.json`.
 */
import { modeIsRemoteLike } from './connection-config'
import { applyConnectionConfigAtomically } from './connection-config-apply'
import { type ConnectionRegistry, setPrimaryConnection } from './connection-registry'
import { forgetRemovedGlobalRoute, withoutGlobalRoute } from './connection-registry-remove'

export interface PrimaryRouteIo {
  readConfig: () => Record<string, any>
  writeConfig: (config: Record<string, any>) => void
  log: (message: string) => void
}

export interface PrimarySwitchIo extends PrimaryRouteIo {
  readRegistry: () => ConnectionRegistry
  writeRegistry: (registry: ConnectionRegistry) => void
}

/**
 * `connection.json` after promoting a LOCAL primary, or null when it needs no change.
 *
 * Only an app-global route is cleared. Per-profile overrides are separate routing decisions
 * and survive (`withoutGlobalRoute` keeps them).
 */
export function configForLocalPrimary(
  registry: ConnectionRegistry,
  config: Record<string, any>
): null | Record<string, any> {
  const primary = registry.connections.find(connection => connection.id === registry.primary)

  if (primary?.kind !== 'local') {
    return null
  }

  const routed = config?.mode === 'ssh' || modeIsRemoteLike(config?.mode)

  return routed ? withoutGlobalRoute(config) : null
}

/**
 * Promote one saved gateway, committing `connection.json` and the registry as one
 * recoverable step.
 *
 * A promotion that only moves the registry keeps the single write it needs. A promotion
 * that must also retire the app-global route goes through the shared Apply boundary, so a
 * failed second write restores both previous files instead of leaving the registry on the
 * new primary while the boot resolver still reads the old one — which is the very state
 * this module exists to prevent.
 */
export async function switchPrimary(id: string, io: PrimarySwitchIo): Promise<ConnectionRegistry> {
  const previousRegistry = io.readRegistry()
  const previousConfig = io.readConfig()
  const registry = setPrimaryConnection(previousRegistry, id)
  const nextConfig = configForLocalPrimary(registry, previousConfig)

  if (!nextConfig) {
    io.writeRegistry(registry)

    return registry
  }

  await applyConnectionConfigAtomically({
    apply: async () => undefined,
    nextConfig,
    nextRegistry: registry,
    previousConfig,
    previousRegistry,
    writeConfig: io.writeConfig,
    writeRegistry: io.writeRegistry
  })

  io.log('[connections] This device is the primary; connection.json route reset to local')

  return registry
}

/**
 * Boot sweep: reset a route whose gateway left the saved list. Strict read, so an unreadable
 * store never reads as "every gateway was removed"; a failure is logged, never thrown.
 */
export function resetRemovedGlobalRoute(registry: ConnectionRegistry, io: PrimaryRouteIo): void {
  try {
    forgetRemovedGlobalRoute(registry, io)
  } catch (error) {
    io.log(`[connections] could not reset a removed gateway's route: ${(error as Error).message}`)
  }
}
