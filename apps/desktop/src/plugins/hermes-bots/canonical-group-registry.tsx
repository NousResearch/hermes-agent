import { atom, Button, host, useValue } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { useCanonicalGroupLabels } from './canonical-group-labels'
import { captureCanonicalGroupRoute, discoverCanonicalGroups } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalGroupRoute, CanonicalRoom } from './canonical-groups'

export const $canonicalGroupBindings = atom<Record<string, CanonicalGroupBinding>>({})

const roomKey = (route: CanonicalGroupRoute, roomId: string) =>
  `canonical:${encodeURIComponent(route.connectionId)}:${encodeURIComponent(route.profile)}:${roomId}`

// Rooms the gateway confirmed it ended: a list that was already in flight never brings one back.
const ended = new Set<string>()

export function registerCanonicalGroup(route: CanonicalGroupRoute, room: CanonicalRoom): string {
  const key = roomKey(route, room.room_id)

  if (ended.has(key)) {return key}
  const binding = { ...route, roomId: room.room_id, roomName: room.name }
  $canonicalGroupBindings.set({ ...$canonicalGroupBindings.get(), [key]: binding })
  renameCanonicalGroup(binding, room.name)

  return key
}

export const sameCanonicalGroup = (a: CanonicalGroupBinding | undefined, b: CanonicalGroupBinding) =>
  a?.connectionId === b.connectionId && a.profile === b.profile && a.roomId === b.roomId

/** A room's current name, from any read of it, reaches every key and alias bound to it: the group
 *  list and the tab labels follow. An alias keeps its old name as its key, so a view open by that
 *  name stays on the room; a new open by that name drops it (`openCanonicalGroupName`). */
export function renameCanonicalGroup(binding: CanonicalGroupBinding, name: string): void {
  const bindings = $canonicalGroupBindings.get()

  if (!Object.values(bindings).some(bound => sameCanonicalGroup(bound, binding) && bound.roomName !== name)) {return}
  $canonicalGroupBindings.set(Object.fromEntries(Object.entries(bindings)
    .map(([key, bound]) => [key, sameCanonicalGroup(bound, binding) ? { ...bound, roomName: name } : bound])))
}

/** Drop every key and alias of a room the gateway confirmed it disbanded. */
export function forgetCanonicalGroup(binding: CanonicalGroupBinding): void {
  ended.add(roomKey(binding, binding.roomId))
  $canonicalGroupBindings.set(Object.fromEntries(Object.entries($canonicalGroupBindings.get())
    .filter(([, bound]) => !sameCanonicalGroup(bound, binding))))
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

  if ((origin === 'key' && key === group) || ended.has(key)) {return}
  const binding = { ...resolved.authority, roomId: resolved.room.room_id, roomName: resolved.room.name }
  const alias = canonicalGroupKeyRoute(group, origin) ? group : canonicalGroupAliasKey(route, group)
  $canonicalGroupBindings.set({ ...$canonicalGroupBindings.get(), [alias]: binding })
}

/** The source a room view resolves its binding on: the window's current connection and profile. */
export const currentCanonicalGroupRoute = (): CanonicalGroupRoute =>
  ({ connectionId: host.state.connectionId.get() ?? '', profile: host.state.profile.get() })

/** Opening a group by name resolves the name again, as after a restart: an alias whose room has
 *  since been renamed answers only for a view already open on it, never for a new open. */
export function openCanonicalGroupName(group: string, origin: CanonicalGroupOrigin): void {
  const alias = canonicalGroupAliasKey(currentCanonicalGroupRoute(), group)
  const { [alias]: bound, ...rest } = $canonicalGroupBindings.get()

  if (origin === 'name' && bound?.roomName && bound.roomName !== group) {$canonicalGroupBindings.set(rest)}
}

/** A room tab's label: the current name of the room `group` shows on `route`, else `group`. */
export const canonicalGroupTitle = (bindings: Record<string, CanonicalGroupBinding>, route: CanonicalGroupRoute,
  group: string, origin: CanonicalGroupOrigin) => canonicalGroupBindingFor(bindings, route, group, origin)?.roomName || group

function CanonicalGroupTitle({ group, origin }: { group: string; origin: CanonicalGroupOrigin }) {
  const route = { connectionId: useValue(host.state.connectionId) ?? '', profile: useValue(host.state.profile) }

  return <>{canonicalGroupTitle(useValue($canonicalGroupBindings), route, group, origin)}</>
}

/** The label fields of a room tab, resolved on the current source as the room view resolves its
 *  binding. The label subscribes for itself, so a rename here or on another surface retitles the
 *  tab without registering it again or bringing it to the front. */
export function canonicalGroupTabLabel(group: string, origin: CanonicalGroupOrigin) {
  const text = () => canonicalGroupTitle($canonicalGroupBindings.get(), currentCanonicalGroupRoute(), group, origin)

  return { title: text(), tabTitle: () => <CanonicalGroupTitle group={group} origin={origin} />, tabTitleText: text }
}

export function CanonicalGroupList({ onOpen }: { onOpen: (key: string, origin: CanonicalGroupOrigin) => void }) {
  const labels = useCanonicalGroupLabels()
  const connectionId = useValue(host.state.connectionId)
  const profile = useValue(host.state.profile)
  // Rows are room keys; each row's name and presence come from the bindings, so a rename or an
  // end shows here as soon as it is confirmed.
  const bindings = useValue($canonicalGroupBindings)
  const [keys, setKeys] = useState<string[]>([])
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)
  useEffect(() => {
    let cancelled = false
    setKeys([])
    setError('')
    void (async () => {
      const route = captureCanonicalGroupRoute()
      const result = await discoverCanonicalGroups(route)

      if (!cancelled) {setKeys(result.rooms.map(room => registerCanonicalGroup(route, room)))}
    })().catch(e => { if (!cancelled) {setError(e instanceof Error ? e.message : String(e))} })

    return () => { cancelled = true }
  }, [connectionId, profile, refresh])

  return <div className="grid gap-1 px-2">
    <Button onClick={() => setRefresh(value => value + 1)} variant="ghost">{labels.refreshGroups}</Button>
    {error && <p role="alert">{error}</p>}
    {keys.map(key => bindings[key] &&
      <Button key={key} onClick={() => onOpen(key, 'key')} variant="ghost">{bindings[key].roomName}</Button>)}
  </div>
}
