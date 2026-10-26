import { atom, Button, host, useValue } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { useCanonicalGroupLabels } from './canonical-group-labels'
import { captureCanonicalGroupRoute, discoverCanonicalGroups } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalGroupRoute, CanonicalRoom } from './canonical-groups'

export const $canonicalGroupBindings = atom<Record<string, CanonicalGroupBinding>>({})
// Bumped when a room leaves outside the list's own Refresh, so the sidebar re-reads `groups.list`.
const $canonicalGroupListRevision = atom(0)

/** Drop a disbanded room's key and every alias to it, then re-read the sidebar list. */
export function forgetCanonicalGroup({ connectionId, profile, roomId }: CanonicalGroupBinding): void {
  $canonicalGroupBindings.set(
    Object.fromEntries(
      Object.entries($canonicalGroupBindings.get()).filter(
        ([, binding]) =>
          binding.connectionId !== connectionId || binding.profile !== profile || binding.roomId !== roomId
      )
    )
  )
  $canonicalGroupListRevision.set($canonicalGroupListRevision.get() + 1)
}

export function registerCanonicalGroup(route: CanonicalGroupRoute, room: CanonicalRoom): string {
  const key = `canonical:${encodeURIComponent(route.connectionId)}:${encodeURIComponent(route.profile)}:${room.room_id}`
  $canonicalGroupBindings.set({
    ...$canonicalGroupBindings.get(),
    [key]: { ...route, roomId: room.room_id, roomName: room.name }
  })

  return key
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
export function canonicalGroupKeyRoute(
  group: string,
  origin: CanonicalGroupOrigin
): ReturnType<typeof parseCanonicalGroupKey> {
  return origin === 'name' ? null : parseCanonicalGroupKey(group)
}

/** The route a `canonical:<connection>:<profile>:<room_id>` key was minted on, or null for a name.
 *  Format-only: callers holding a possible NAME go through `canonicalGroupKeyRoute`. */
export function parseCanonicalGroupKey(key: string): (CanonicalGroupRoute & { roomId: string }) | null {
  const match = /^canonical:([^:]*):([^:]*):(.+)$/.exec(key)

  if (!match) {
    return null
  }

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

  if (origin === 'key' && key === group) {
    return
  }
  const binding = { ...resolved.authority, roomId: resolved.room.room_id, roomName: resolved.room.name }
  const alias = canonicalGroupKeyRoute(group, origin) ? group : canonicalGroupAliasKey(route, group)
  $canonicalGroupBindings.set({ ...$canonicalGroupBindings.get(), [alias]: binding })
}

export function CanonicalGroupList({ onOpen }: { onOpen: (key: string, origin: CanonicalGroupOrigin) => void }) {
  const labels = useCanonicalGroupLabels()
  const connectionId = useValue(host.state.connectionId)
  const profile = useValue(host.state.profile)
  const [rooms, setRooms] = useState<Array<{ key: string; name: string }>>([])
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)
  const revision = useValue($canonicalGroupListRevision)
  useEffect(() => {
    let cancelled = false
    setRooms([])
    setError('')
    void (async () => {
      const route = captureCanonicalGroupRoute()
      const result = await discoverCanonicalGroups(route)

      if (!cancelled) {
        setRooms(result.rooms.map(room => ({ key: registerCanonicalGroup(route, room), name: room.name })))
      }
    })().catch(e => {
      if (!cancelled) {
        setError(e instanceof Error ? e.message : String(e))
      }
    })

    return () => {
      cancelled = true
    }
  }, [connectionId, profile, refresh, revision])

  return (
    <div className="grid gap-1 px-2">
      <Button onClick={() => setRefresh(value => value + 1)} variant="ghost">
        {labels.refreshGroups}
      </Button>
      {error && <p role="alert">{error}</p>}
      {rooms.map(room => (
        <Button key={room.key} onClick={() => onOpen(room.key, 'key')} variant="ghost">
          {room.name}
        </Button>
      ))}
    </div>
  )
}
