import { backendScopeKey, type ConnectionRegistry, LOCAL_CONNECTION_ID } from './connection-registry'

export interface WindowConnectionRoute {
  connectionId: null | string
  profile: string | undefined
  registryScoped: boolean
}

export function normalizeWindowConnectionRoute(value: unknown): WindowConnectionRoute | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const input = value as Record<string, unknown>
  const connectionId = typeof input.connectionId === 'string' ? input.connectionId.trim() : ''

  const profile = typeof input.profile === 'string' && input.profile.trim() ? input.profile.trim() : undefined

  return {
    connectionId: connectionId || null,
    profile,
    registryScoped: input.registryScoped === true
  }
}

export function registrySshScopeForWindowRoute(
  route: WindowConnectionRoute | null | undefined,
  registry: ConnectionRegistry
): null | string {
  if (!route?.registryScoped || !route.connectionId) {
    return null
  }

  const source = registry.connections.find(connection => connection.id === route.connectionId)

  if (!source || source.kind !== 'ssh') {
    return null
  }

  return backendScopeKey(route.connectionId, route.profile)
}

/**
 * The route a window must re-dial after a PRIMARY connection apply (#92352).
 *
 * An apply re-homes the primary, but the window's recorded route still names
 * the source it just LEFT. `resolveDesktopConnectionRequest` answers the
 * renderer's profile-less apply re-dial from that record, so the renderer kept
 * dialing the stale registry-scoped gateway — and every later reconnect, plus
 * any window launched from that route, re-asked the same stale question until a
 * restart dropped the in-memory record. Re-point the record at the newly
 * applied primary, keeping the profile the window was viewing: an applied
 * registry source (remote/cloud/ssh) stays registry-scoped to that exact
 * identity, while a This-device apply stays unscoped so the dial follows the
 * freshly written v1 config.
 */
export function appliedPrimaryWindowRoute(
  registry: ConnectionRegistry,
  profile: null | string | undefined
): WindowConnectionRoute {
  const key = String(profile ?? '').trim() || 'default'
  const primary = String(registry?.primary ?? '').trim()

  return primary && primary !== LOCAL_CONNECTION_ID
    ? { connectionId: primary, profile: key, registryScoped: true }
    : { connectionId: null, profile: key, registryScoped: false }
}

export interface RegistrySshPoolEntry {
  registryConnectionId?: null | string
  ssh?: unknown
}

// The sshConnections pool has a single writer that publishes every tunnel under
// its per-profile bootstrap key while stamping the entry with the registry
// connection that owns it. A registry-scoped lookup under the composite
// backendScopeKey can therefore miss a live tunnel; this recovers the writer's
// actual key from the stamped identity, the same match managedSshScopeRole
// applies to pool entries.
export function registrySshPoolScopeByConnectionId(
  entries: Iterable<readonly [string, RegistrySshPoolEntry | undefined]>,
  connectionId: string
): null | string {
  for (const [scope, entry] of entries) {
    if (entry?.ssh && entry.registryConnectionId === connectionId) {
      return scope
    }
  }

  return null
}

export class WindowConnectionRouteRegistry {
  private readonly routes = new Map<number, WindowConnectionRoute>()

  set(webContentsId: number, value: unknown): WindowConnectionRoute | null {
    const route = normalizeWindowConnectionRoute(value)

    if (!route) {
      this.routes.delete(webContentsId)

      return null
    }

    this.routes.set(webContentsId, route)

    return route
  }

  get(webContentsId: number): WindowConnectionRoute | null {
    return this.routes.get(webContentsId) ?? null
  }

  delete(webContentsId: number): void {
    this.routes.delete(webContentsId)
  }
}
