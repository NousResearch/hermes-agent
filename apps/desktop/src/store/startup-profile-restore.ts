import type { DesktopAgentRoster } from '@/global'
import type { ProfileInfo } from '@/types/hermes'

/** The roster has reported the launch target down: the window has fallen back
 *  to the source already in front. Connect-on-demand SSH counts as reachable. */
export function launchTargetUnreachable(roster: DesktopAgentRoster | null, connectionId: string): boolean {
  return roster?.sources.some(source => source.connectionId === connectionId && !source.reachable) === true
}

/**
 * The profiles a gateway serves, from its `/api/status`: `/api/profiles` lists
 * every INSTALLED profile, parked and standalone ones included, while a
 * gateway-hosted listener refuses those with `profile_mismatch`.
 * `gateway_shared_with` is the multiplexer's served set; [] means a
 * single-profile gateway that serves only the host's own `default`. Null
 * (no gateway running, an older backend, a failed or signed-out read) is
 * unknown, and every installed profile counts as served. Electron main's
 * remote primary restore (servedProfileSet) applies the same rule.
 */
export function servedProfileSet(status: unknown): null | Set<string> {
  const shared = (status as { gateway_shared_with?: unknown } | null)?.gateway_shared_with

  if (!Array.isArray(shared)) {
    return null
  }

  return new Set([
    'default',
    ...shared.filter((name): name is string => typeof name === 'string' && Boolean(name.trim()))
  ])
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
