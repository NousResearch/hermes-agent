/**
 * Which profile a caller of the primary backend asked for, and whether it
 * named one at all. A profile-less caller follows the primary, and only that
 * caller may be moved when a remote boot re-pins the primary onto a profile its
 * gateway serves. The resolved key alone cannot say this: `hermes:connection`
 * fills an empty request with the primary's name before it reaches
 * ensureBackend/startHermes.
 */
export interface PrimaryBootRequest {
  key: string
  inheritsPrimary: boolean
}

export function primaryBootRequest(profile: unknown, primaryKey: string, inheritsPrimary = false): PrimaryBootRequest {
  const named = profile === null || profile === undefined ? '' : String(profile).trim()

  return { key: named || primaryKey, inheritsPrimary: inheritsPrimary || !named }
}

/**
 * Join (or start) the primary boot, then answer with the route the caller now
 * owns. An inherited caller re-reads the primary after the wait: a remote boot
 * that restored a served profile re-pinned it, and the route resolved before
 * the wait still names the profile that gateway does not serve. An explicit
 * caller keeps its own profile and route.
 */
export async function joinPrimaryBoot<Connection, Route>(
  request: PrimaryBootRequest,
  route: Route,
  deps: {
    primaryKey: () => string
    resolveRoute: (key: string) => Route
    start: (request: PrimaryBootRequest) => Promise<Connection>
  }
): Promise<{ connection: Connection; key: string; route: Route }> {
  const connection = await deps.start(request)

  if (!request.inheritsPrimary) {
    return { connection, key: request.key, route }
  }

  const key = deps.primaryKey()

  return { connection, key, route: key === request.key ? route : deps.resolveRoute(key) }
}

/**
 * A startHermes attempt may swap a remote primary's stale profile for a served
 * one only when its profile was inherited: no caller named it, no --profile
 * launch argument and no per-profile route pins it.
 */
export function primaryProfileRestorable(launch: {
  explicitDefaultEntry: boolean
  inheritsPrimary: boolean
  launchArgProfile: null | string
  requestedProfile: string | undefined
}): boolean {
  return (!launch.requestedProfile || launch.inheritsPrimary) && !launch.launchArgProfile && !launch.explicitDefaultEntry
}
