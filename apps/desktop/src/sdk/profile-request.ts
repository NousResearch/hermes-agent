import { requestGatewayForAgent, requestGatewayForProfile, type SpawnPriority } from '@/store/gateway'
import { $profiles } from '@/store/profile'

/** Plugin-facing profile routing for `host.requestProfile` and friends. */

export interface PluginProfileRoute {
  connectionId: string
  mode: 'local' | 'remote'
  /** Electron's authoritative registry primary. Absent on older shells. */
  primary?: true
  /** Desktop profile used to select the connection route. */
  profile: string
  /** Backend Hermes profile served by that route. */
  targetProfile: string
}

/** Options a plugin may attach to one `host.requestProfile` call. */
export interface PluginProfileRequestOptions {
  /** Tag the dial that may cold-spawn this route's backend. Default
   *  'background'; an explicit user action passes 'foreground' so its spawn
   *  takes the pool's reserved interactive slot (#102281 primitive). */
  spawnPriority?: SpawnPriority
}

export async function requestPluginProfile<T>(
  route: PluginProfileRoute | string,
  method: string,
  params: Record<string, unknown>,
  timeoutMs?: number,
  options?: PluginProfileRequestOptions
): Promise<T> {
  const spawnPriority = options?.spawnPriority

  // Preserve the exact call arity the pool tests pin: pass the deadline and the
  // dial options only when the caller set them, so a plain routed RPC keeps its
  // four-argument shape and a timeout-only caller its five-argument shape.
  const dialProfile = (profile: string): Promise<T> =>
    spawnPriority
      ? requestGatewayForProfile<T>(profile, method, params, timeoutMs, undefined, { spawnPriority })
      : timeoutMs === undefined
        ? requestGatewayForProfile<T>(profile, method, params)
        : requestGatewayForProfile<T>(profile, method, params, timeoutMs)

  if (typeof route !== 'string') {
    if (!route.connectionId.trim() || !route.profile.trim() || !route.targetProfile.trim()) {
      throw new Error('Profile route must include connectionId, profile, and targetProfile')
    }

    if (spawnPriority) {
      return requestGatewayForAgent<T>(route.connectionId, route.profile, method, params, timeoutMs, undefined, {
        spawnPriority
      })
    }

    return timeoutMs === undefined
      ? requestGatewayForAgent<T>(route.connectionId, route.profile, method, params)
      : requestGatewayForAgent<T>(route.connectionId, route.profile, method, params, timeoutMs)
  }

  const getAgentRoster = window.hermesDesktop?.getAgentRoster

  if (!getAgentRoster) {
    return dialProfile(route)
  }

  const roster = await getAgentRoster()
  const profile = route.trim() || 'default'
  const soleLocalSource = roster.sources.length === 1 && roster.sources[0]?.kind === 'local'

  // The string overload is compatibility-only. A sole local registry is the
  // one topology where a profile name is intrinsically unambiguous, even when
  // its live enumeration transiently failed. Any additional source requires a
  // descriptor because an undialed/unreachable source may expose the same name.
  if (soleLocalSource) {
    return dialProfile(profile)
  }

  throw new Error(
    `Profile "${profile}" requires a route descriptor from host.profileRoutes(); profile-only routing is limited to legacy/local profiles.`
  )
}

/** Re-read Electron's current registry before retrying an exact-owner wake.
 *  A route that was removed or replaced while the first hydration wait ran is
 *  no longer authority to touch that backend, even when its labels still look
 *  identical. */
export async function pluginRouteStillRegistered(route: PluginProfileRoute): Promise<boolean> {
  const getProfileRoutes = window.hermesDesktop?.getProfileRoutes

  if (!getProfileRoutes) {
    return false
  }

  try {
    const routes = await getProfileRoutes($profiles.get().map(profile => profile.name))

    return routes.some(
      candidate =>
        candidate.connectionId === route.connectionId &&
        candidate.profile === route.profile &&
        candidate.targetProfile === route.targetProfile
    )
  } catch {
    return false
  }
}
