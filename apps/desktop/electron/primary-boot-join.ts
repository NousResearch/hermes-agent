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
export async function joinPrimaryBoot<Connection extends object, Route extends { descriptorProfile: null | string }>(
  request: PrimaryBootRequest,
  route: Route,
  deps: {
    primaryKey: () => string
    resolveRoute: (key: string) => Route
    start: (request: PrimaryBootRequest) => Promise<Connection>
  }
) {
  const connection = await deps.start(request)
  const key = request.inheritsPrimary ? deps.primaryKey() : request.key
  const settled = key === request.key ? route : deps.resolveRoute(key)

  return { connection, descriptor: primaryRouteDescriptor(connection, key, settled), key, route: settled }
}

/**
 * The descriptor a primary-route caller gets. A shared backend still owes the
 * caller its profile scope, so renderer-side WebSocket, filesystem and cache
 * routing target the selected profile. `sharedPrimary` marks the
 * shared-primary route: pooled backends also carry `profile`, so only this
 * descriptor gets the flag. The unshared primary carries its own key too: a
 * profile-less descriptor reads as "default" downstream, which breaks
 * per-source profile memory (the primary IS "default" only when it actually
 * booted as default).
 */
function primaryRouteDescriptor<Connection extends object>(
  connection: Connection,
  key: string,
  route: { descriptorProfile: null | string }
) {
  return route.descriptorProfile
    ? { ...connection, profile: route.descriptorProfile, sharedPrimary: true }
    : { ...connection, profile: key }
}

/** Inherited and explicit dials of one name answer different profiles once a
 *  remote boot re-pins the primary, so they never share a dial claim. */
export function primaryDialClaimKey(scopeKey: string, inheritsPrimary: boolean): string {
  return inheritsPrimary ? `${scopeKey}\u0000inherited` : scopeKey
}

/**
 * A startHermes attempt may swap a remote primary's stale profile for a served
 * one only when its profile was inherited: no caller named it, no --profile
 * launch argument and no per-profile route pins it.
 */
export function primaryProfileRestorable(
  requestedProfile: string | undefined,
  inheritsPrimary: boolean,
  launchArgProfile: null | string,
  defaultProfileRoute: unknown
): boolean {
  return (!requestedProfile || inheritsPrimary) && !launchArgProfile && !defaultProfileRoute
}
