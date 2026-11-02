// Routes Desktop keeps connected for the window's lifetime (store/gateway-persistence).
// A leaf module, so store/gateway can consult it without importing its consumer.
const kept = new Set<string>()
const openedListeners = new Set<() => void>()

const routeKey = (connectionId: null | string, profile: string) => `${connectionId ?? ''}\u0000${profile}`

/** Keep one route's socket open against every idle-dispose path. Returns the release. */
export function keepRoute(connectionId: string, profile: string): () => void {
  const key = routeKey(connectionId, profile)
  kept.add(key)

  return () => kept.delete(key)
}

export function isKeptRoute(connectionId: null | string, profile: string): boolean {
  return kept.has(routeKey(connectionId, profile))
}

/** Fires each time a kept route's socket opens, first dial or automatic reconnect alike. */
export function onKeptRouteOpened(handler: () => void): () => void {
  openedListeners.add(handler)

  return () => openedListeners.delete(handler)
}

export function keptRouteOpened(connectionId: null | string, profile: string): void {
  if (isKeptRoute(connectionId, profile)) {
    openedListeners.forEach(handler => handler())
  }
}
