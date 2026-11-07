import { applyConnectionConfigAtomically } from './connection-config-apply'
import { type ConnectionRegistry, reconcileRegistryDrift, removeConnection } from './connection-registry'

interface ConnectionRemovalWriters {
  writeConfig: (config: Record<string, any>) => void
  writeRegistry: (registry: ConnectionRegistry) => void
}

/** connection.json with its app-global route reset to local (per-profile overrides kept). */
export const withoutGlobalRoute = (config: Record<string, any>) => ({ ...config, mode: 'local', remote: {} })

/**
 * True when connection.json's app-global route names a gateway the saved list no
 * longer holds. The list is the authority once it exists: such a route was removed
 * (here or by the Webapp), so it must be dropped, never re-added. A Settings save
 * registers its route at save time, so a live route is never in this state.
 */
export function globalRouteRemoved(registry: ConnectionRegistry, config: Record<string, any>): boolean {
  return reconcileRegistryDrift(registry, config).registry.connections.length > registry.connections.length
}

interface GlobalRouteIo {
  readConfig: () => Record<string, any>
  writeConfig: (config: Record<string, any>) => void
  log?: (message: string) => void
}

/**
 * Reset connection.json's app-global route when the saved list no longer holds it:
 * removed by the Webapp, or by a build that left the route behind. Without this the
 * removed gateway keeps winning the primary boot route over the list's primary.
 */
export function forgetRemovedGlobalRoute(registry: ConnectionRegistry, io: GlobalRouteIo): boolean {
  const config = io.readConfig()

  if (!globalRouteRemoved(registry, config)) {
    return false
  }

  io.writeConfig(withoutGlobalRoute(config))
  io.log?.('[connections] connection.json named a gateway no longer in the saved list; its route was reset to local')

  return true
}

/** Persist the saved-connection removal before tearing down its live resources. */
export async function removeSavedConnection(
  id: string,
  registry: ConnectionRegistry,
  config: Record<string, any>,
  writers: ConnectionRemovalWriters
): Promise<ConnectionRegistry> {
  const nextRegistry = removeConnection(registry, id)

  // Use the reader's identity policy: clear only a global legacy route that
  // this deletion would cause it to re-import. An unrelated removal, a route
  // still represented by another entry, and pre-existing drift stay untouched.
  if (!globalRouteRemoved(registry, config) && globalRouteRemoved(nextRegistry, config)) {
    await applyConnectionConfigAtomically({
      previousConfig: config,
      previousRegistry: registry,
      nextConfig: withoutGlobalRoute(config),
      nextRegistry,
      ...writers,
      apply: async () => undefined
    })
  } else {
    writers.writeRegistry(nextRegistry)
  }

  return nextRegistry
}
