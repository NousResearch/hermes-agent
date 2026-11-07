import { modeIsRemoteLike, normalizeRemoteBaseUrl, normalizeSshConfig } from './connection-config'
import { applyConnectionConfigAtomically } from './connection-config-apply'
import { type ConnectionRegistry, reconcileRegistryDrift, removeConnection } from './connection-registry'

interface ConnectionRemovalWriters {
  writeConfig: (config: Record<string, any>) => void
  writeRegistry: (registry: ConnectionRegistry) => void
}

/** connection.json with its app-global route reset to local (per-profile overrides kept). */
export const withoutGlobalRoute = (config: Record<string, any>) => ({ ...config, mode: 'local', remote: {} })

/** user@host:port::remoteProfile of an SSH route, or null when it names no host. */
function sshRouteKey(route: Record<string, any>): null | string {
  const ssh = normalizeSshConfig({ ...route, mode: 'ssh' })

  return ssh
    ? `${(ssh.user || '').trim().toLowerCase()}@${ssh.host.trim().toLowerCase()}:${ssh.port || 22}::${ssh.remoteProfile || ''}`
    : null
}

const urlKey = (url: unknown): string => {
  try {
    return normalizeRemoteBaseUrl(url)
  } catch {
    return ''
  }
}

/**
 * True when connection.json's app-global route names a gateway the saved list no
 * longer holds (same URL; same SSH user, host, port and remote profile). The list
 * is the authority once it exists: such a route was removed (here or by the
 * Webapp), so it must be dropped, never re-added.
 */
export function globalRouteRemoved(registry: ConnectionRegistry, config: Record<string, any>): boolean {
  const remote = config?.remote && typeof config.remote === 'object' ? config.remote : {}

  if (config?.mode === 'ssh') {
    const key = sshRouteKey(remote)

    return Boolean(key) && !registry.connections.some(c => c.kind === 'ssh' && sshRouteKey(c) === key)
  }

  const url = modeIsRemoteLike(config?.mode) ? urlKey(remote.url) : ''

  return (
    Boolean(url) &&
    !registry.connections.some(c => (c.kind === 'remote' || c.kind === 'cloud') && urlKey(c.url) === url)
  )
}

/** The list with connection.json's app-global route registered (Settings save / apply). */
export const registerGlobalRoute = (registry: ConnectionRegistry, config: Record<string, any>) =>
  reconcileRegistryDrift(registry, config).registry

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
