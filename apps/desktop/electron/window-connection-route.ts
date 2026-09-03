import { backendScopeKey, type ConnectionRegistry } from './connection-registry'

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
