import { applyConnectionConfigAtomically } from './connection-config-apply'
import { type ConnectionRegistry, reconcileRegistryDrift, removeConnection } from './connection-registry'

interface ConnectionRemovalWriters {
  writeConfig: (config: Record<string, any>) => void
  writeRegistry: (registry: ConnectionRegistry) => void
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
  const before = reconcileRegistryDrift(registry, config).registry
  const after = reconcileRegistryDrift(nextRegistry, config).registry

  if (
    before.connections.length === registry.connections.length &&
    after.connections.length > nextRegistry.connections.length
  ) {
    await applyConnectionConfigAtomically({
      previousConfig: config,
      previousRegistry: registry,
      nextConfig: { ...config, mode: 'local', remote: {} },
      nextRegistry,
      ...writers,
      apply: async () => undefined
    })
  } else {
    writers.writeRegistry(nextRegistry)
  }

  return nextRegistry
}
