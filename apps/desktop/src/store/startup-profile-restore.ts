import type { ProfileInfo } from '@/types/hermes'

/**
 * The profiles a gateway serves, from its `/api/status`: `/api/profiles` lists
 * every INSTALLED profile, parked and standalone ones included, while a
 * gateway-hosted listener refuses those with `profile_mismatch`.
 * `gateway_shared_with` is the multiplexer's served set, its launch profile
 * included. [] (a single-profile gateway, serving only the profile it was
 * launched as) and null (no gateway running, an older backend, a failed or
 * signed-out read) are unknown, and every installed profile counts as served. Electron main's
 * remote primary restore (servedProfileSet) applies the same rule.
 */
export function servedProfileSet(status: unknown): null | Set<string> {
  const shared = (status as { gateway_shared_with?: unknown } | null)?.gateway_shared_with

  if (!Array.isArray(shared)) {
    return null
  }

  const names = shared.filter((name): name is string => typeof name === 'string' && Boolean(name.trim()))

  // [] is a single-profile gateway serving only the profile it was launched as,
  // which the list does not name: unknown here, not `default`.
  return names.length ? new Set(names) : null
}

/** A profile remembered globally is not necessarily on the selected gateway,
 *  nor served there. Null keeps `active`. */
export function startupProfileForConnection(
  active: string,
  remembered: string | undefined,
  profiles: Pick<ProfileInfo, 'name'>[],
  served: null | ReadonlySet<string> = null
): string | null {
  const names = profiles.map(profile => profile.name).filter(name => served === null || served.has(name))
  const servable = new Set(names)

  if (servable.has(active)) {
    return null
  }

  if (remembered && servable.has(remembered)) {
    return remembered
  }

  if (servable.has('default')) {
    return 'default'
  }

  return names[0] ?? null
}
