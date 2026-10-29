import type { HermesConnection } from '@/global'
import { $connection } from '@/store/session'

export interface WindowRoute {
  connectionId: null | string
  profile?: string
  registryScoped: boolean
}

export function windowRouteFor(connection: HermesConnection | null): WindowRoute | null {
  return connection
    ? {
        connectionId: connection.connectionId ?? null,
        profile: connection.profile,
        registryScoped: connection.registryScoped === true
      }
    : null
}

// What this renderer last told main; undefined = nothing sent since mount.
let lastSent: undefined | WindowRoute | null

const sameRoute = (a: WindowRoute | null, b: WindowRoute | null) =>
  a === b ||
  (a !== null &&
    b !== null &&
    a.connectionId === b.connectionId &&
    a.profile === b.profile &&
    a.registryScoped === b.registryScoped)

export function sendWindowRoute(connection: HermesConnection | null, { force = false } = {}): void {
  const route = windowRouteFor(connection)

  if (!force && lastSent !== undefined && sameRoute(lastSent, route)) {
    return
  }

  lastSent = route
  window.hermesDesktop?.setActiveConnectionRoute?.(route)
}

// Main's window route (New Window, profile-less getConnection, terminals)
// must name what this window shows, and several stores write $connection
// without going through the boot hook's publish() (#143725). `listen`, not `subscribe`:
// the initial null must not clobber the route main seeded before boot. A null
// write is a reconnect blip, and the retry's profile-less getConnection()
// resolves through this very route, so only an explicit publish(null) clears it.
export function followWindowRoute(): () => void {
  lastSent = undefined

  return $connection.listen(connection => {
    if (connection) {
      sendWindowRoute(connection)
    }
  })
}
