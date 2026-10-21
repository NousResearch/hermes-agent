import { atom, host, useValue } from '@hermes/plugin-sdk'
import { useEffect, useMemo } from 'react'

import type { CanonicalGroupRoom } from './canonical-group-rows'
import { captureCanonicalGroupRoute, discoverCanonicalGroups } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalGroupRoute, CanonicalRoom } from './canonical-groups'

export const $canonicalGroupBindings = atom<Record<string, CanonicalGroupBinding>>({})

/** Every hosted room this Desktop knows, by `canonical:` key. Creation
 *  (`registerCanonicalGroup`) and discovery both publish here, and the left
 *  roster renders from it — that is what makes a new group appear. */
export const $canonicalGroupRooms = atom<Record<string, { route: CanonicalGroupRoute; room: CanonicalRoom }>>({})

/** The stable identity of one hosted room on one source. */
export function canonicalGroupKey(route: CanonicalGroupRoute, room: CanonicalRoom): string {
  return `canonical:${encodeURIComponent(route.connectionId)}:${encodeURIComponent(route.profile)}:${room.room_id}`
}

export function registerCanonicalGroup(route: CanonicalGroupRoute, room: CanonicalRoom): string {
  const key = canonicalGroupKey(route, room)
  $canonicalGroupBindings.set({ ...$canonicalGroupBindings.get(), [key]: { ...route, roomId: room.room_id, roomName: room.name } })
  $canonicalGroupRooms.set({
    ...$canonicalGroupRooms.get(),
    [key]: { room, route: { connectionId: route.connectionId, profile: route.profile } }
  })

  return key
}

/** Re-list one source's rooms and publish them. `groups.list` is the whole
 *  answer for that (connection, profile), so a room the gateway no longer
 *  holds leaves the roster with it. */
export async function refreshCanonicalGroupRooms(route: CanonicalGroupRoute): Promise<void> {
  const { rooms } = await discoverCanonicalGroups(route)
  const bindings = { ...$canonicalGroupBindings.get() }
  const next: Record<string, { route: CanonicalGroupRoute; room: CanonicalRoom }> = {}

  for (const room of rooms) {
    const key = canonicalGroupKey(route, room)
    next[key] = { room, route: { connectionId: route.connectionId, profile: route.profile } }
    bindings[key] = { ...route, roomId: room.room_id, roomName: room.name }
  }

  $canonicalGroupBindings.set(bindings)
  $canonicalGroupRooms.set(next)
}

/** Keep the roster's hosted rooms in step with the source this window is on,
 *  and with the gateway coming back. Failures keep the last good list. */
export function useCanonicalGroupRooms(): CanonicalGroupRoom[] {
  const connectionId = useValue(host.state.connectionId)
  const profile = useValue(host.state.profile)
  const gateway = useValue(host.state.gateway)
  const rooms = useValue($canonicalGroupRooms)

  useEffect(() => {
    void refreshCanonicalGroupRooms(captureCanonicalGroupRoute()).catch(() => {
      // No canonical driver (or the gateway is restarting): show what we have.
    })
  }, [connectionId, gateway, profile])

  return useMemo(() => Object.entries(rooms).map(([key, entry]) => ({ key, room: entry.room })), [rooms])
}

/** Where a group string entered. Room and group names are free text (hosted_rooms
 *  has no pattern), so a name can spell `canonical:<conn>:<profile>:<room>`
 *  exactly; the string alone cannot tell the two apart, the caller can.
 *  - 'key': minted by `registerCanonicalGroup`, or a persisted/restored tab
 *    string. Parsed as main always parsed it, so pre-existing keys keep opening.
 *  - 'name': a user-typed group name, a room record's name, a roster row. Never
 *    parsed: it resolves only through the alias for the current source. */
export type CanonicalGroupOrigin = 'key' | 'name'

/** The route `group` names when it is a key: null for a name, whatever its shape. */
export function canonicalGroupKeyRoute(group: string, origin: CanonicalGroupOrigin): ReturnType<typeof parseCanonicalGroupKey> {
  return origin === 'name' ? null : parseCanonicalGroupKey(group)
}

/** The route a `canonical:<connection>:<profile>:<room_id>` key was minted on, or null for a name.
 *  Format-only: callers holding a possible NAME go through `canonicalGroupKeyRoute`. */
export function parseCanonicalGroupKey(key: string): (CanonicalGroupRoute & { roomId: string }) | null {
  const match = /^canonical:([^:]*):([^:]*):(.+)$/.exec(key)

  if (!match) {return null}

  try {
    return { connectionId: decodeURIComponent(match[1]), profile: decodeURIComponent(match[2]), roomId: match[3] }
  } catch {
    return null
  }
}

/** The binding key for a group opened by name on one (connection, profile).
 *  A bare name is not an identity: two gateways can each hold a "Team", so a
 *  name resolved on one source must never answer for another. */
export function canonicalGroupAliasKey(route: CanonicalGroupRoute, group: string): string {
  return `alias:${encodeURIComponent(route.connectionId)}:${encodeURIComponent(route.profile)}:${group}`
}

/** The binding `group` opens on `route`: a `canonical:` key names its own route;
 *  a name only matches the alias published for the current source. */
export function canonicalGroupBindingFor(
  bindings: Record<string, CanonicalGroupBinding>,
  route: CanonicalGroupRoute,
  group: string,
  origin: CanonicalGroupOrigin
): CanonicalGroupBinding | undefined {
  return canonicalGroupKeyRoute(group, origin) ? bindings[group] : bindings[canonicalGroupAliasKey(route, group)]
}

/** #140805: bindings live in memory, so after a restart a group opened by name
 *  (roster row) or by a restored `canonical:` tab key misses. Look it up in the
 *  gateway's `groups.list`. Read-only: the caller publishes the result with
 *  `publishCanonicalGroup` only if its lookup is still current, so a superseded
 *  source's late answer never opens its room. Returns null only when the
 *  authority answered and holds no such room; a failed lookup THROWS, so the
 *  caller fails closed (retry, never mint). Same-name forks resolve to the most
 *  recently updated room (list order). */
export async function resolveCanonicalGroup(
  route: CanonicalGroupRoute,
  group: string,
  origin: CanonicalGroupOrigin
): Promise<{ authority: CanonicalGroupRoute; room: CanonicalRoom } | null> {
  const parsed = canonicalGroupKeyRoute(group, origin)
  const authority = parsed ? { connectionId: parsed.connectionId, profile: parsed.profile } : route
  const { rooms } = await discoverCanonicalGroups(authority)
  const room = parsed ? rooms.find(r => r.room_id === parsed.roomId) : rooms.find(r => r.name === group)

  return room ? { authority, room } : null
}

/** Publish a resolved room under its `canonical:` key and, for a name, under
 *  the alias scoped to the route it was resolved on. */
export function publishCanonicalGroup(
  route: CanonicalGroupRoute,
  group: string,
  resolved: { authority: CanonicalGroupRoute; room: CanonicalRoom },
  origin: CanonicalGroupOrigin
): void {
  const key = registerCanonicalGroup(resolved.authority, resolved.room)

  if (origin === 'key' && key === group) {return}
  const binding = { ...resolved.authority, roomId: resolved.room.room_id, roomName: resolved.room.name }
  const alias = canonicalGroupKeyRoute(group, origin) ? group : canonicalGroupAliasKey(route, group)
  $canonicalGroupBindings.set({ ...$canonicalGroupBindings.get(), [alias]: binding })
}


