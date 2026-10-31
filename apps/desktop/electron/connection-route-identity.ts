/**
 * Canonical registry route identity for Desktop.
 *
 * Identity is frozen before dialing, while the complete auth/transport
 * envelope still exists. Compatibility callers may reuse the same contract,
 * but must never reconstruct a stronger identity from post-dial metadata.
 */

import { normalizeRemoteBaseUrl, normalizeRemoteHeaders, normalizeSshConfig, normAuthMode } from './connection-config'
import type { ConnectionRegistry, RegistryConnection } from './connection-registry'

interface SshRouteConfig {
  host: string
  keyPath?: string
  mode: 'ssh'
  port?: number
  remoteHermesPath?: string
  remoteProfile?: string
  user?: string
}

export type StoredRoute =
  | {
      authMode?: unknown
      headers?: Record<string, unknown>
      kind: 'cloud' | 'remote'
      org?: unknown
      token?: unknown
      url?: unknown
    }
  | ({ kind: 'ssh' } & Partial<SshRouteConfig>)

function stableValue(value: unknown): string {
  if (!value || typeof value !== 'object') {
    return JSON.stringify(value ?? null)
  }

  if (Array.isArray(value)) {
    return `[${value.map(stableValue).join(',')}]`
  }

  return `{${Object.entries(value)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, item]) => `${JSON.stringify(key)}:${stableValue(item)}`)
    .join(',')}}`
}

function canonicalHeaders(headers: unknown): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(normalizeRemoteHeaders(headers))
      .map(([name, value]): [string, unknown] => [name.toLowerCase(), value])
      .sort(([left], [right]) => left.localeCompare(right))
  )
}

function routeIdentity(route: StoredRoute): null | string {
  if (route.kind === 'ssh') {
    const ssh = normalizeSshConfig({ ...route, mode: 'ssh' })

    if (!ssh) {
      return null
    }

    return stableValue({
      host: ssh.host.trim().toLowerCase(),
      keyPath: ssh.keyPath || '',
      kind: 'ssh',
      port: ssh.port || 22,
      remoteHermesPath: ssh.remoteHermesPath || '',
      remoteProfile: ssh.remoteProfile || '',
      user: (ssh.user || '').trim().toLowerCase()
    })
  }

  try {
    const authMode = normAuthMode(route.authMode)

    return stableValue({
      authMode,
      headers: canonicalHeaders(route.headers),
      kind: route.kind,
      org: route.kind === 'cloud' ? String(route.org || '').trim() : '',
      token: authMode === 'token' ? (route.token ?? null) : null,
      url: normalizeRemoteBaseUrl(route.url)
    })
  } catch {
    return null
  }
}

function registryRoute(connection: RegistryConnection): null | StoredRoute {
  if (connection.kind === 'local') {
    return null
  }

  return connection as StoredRoute
}

/**
 * Match the complete pre-dial route identity used by #88922.
 *
 * `primary` accepts only the configured primary when its full envelope is
 * equal. `unique` accepts exactly one full-envelope match. Zero and multiple
 * matches deliberately remain unresolved.
 */
export function matchingConnectionId(
  registry: ConnectionRegistry,
  route: StoredRoute,
  strategy: 'primary' | 'unique'
): undefined | string {
  const identity = routeIdentity(route)

  if (!identity) {
    return undefined
  }

  if (strategy === 'primary') {
    const primary = registry.connections.find(connection => connection.id === registry.primary)
    const candidate = primary && registryRoute(primary)

    return candidate && routeIdentity(candidate) === identity ? primary.id : undefined
  }

  const matches = registry.connections.filter(connection => {
    const candidate = registryRoute(connection)

    return candidate ? routeIdentity(candidate) === identity : false
  })

  return matches.length === 1 ? matches[0].id : undefined
}
