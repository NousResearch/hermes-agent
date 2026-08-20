import { backendScopeKey, type ConnectionRegistry } from './connection-registry'

/**
 * Return the live SSH pool scope for the registry connection currently used by
 * the Sessions workspace. Non-SSH sources deliberately return null so callers
 * can preserve the legacy v1 routing fallback.
 */
export function activeRegistrySshScope(
  registry: ConnectionRegistry,
  profile: null | string | undefined
): null | string {
  const connection = registry.connections.find(candidate => candidate.id === registry.lastUsed)

  if (!connection || connection.kind !== 'ssh') {
    return null
  }

  return backendScopeKey(connection.id, profile)
}
