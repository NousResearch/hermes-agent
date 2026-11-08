import { atom, host, randomUuid, useQuery, useValue } from '@hermes/plugin-sdk'
import { useEffect, useMemo } from 'react'

import { canonicalGroupRequest, discoverCanonicalGroups, listCanonicalRooms } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalGroupRoute, CanonicalRoom, CanonicalRoomMember } from './canonical-groups'
import { closeGroupChatMainTab } from './group-panes'
import { rpcMethodMissing } from './rpc-errors'
import { ID } from './shared'

export const $canonicalGroupBindings = atom<Record<string, CanonicalGroupBinding>>({})
// Bumped when this window creates or disbands a room, so the roster re-reads `groups.list` now.
const $canonicalGroupListRevision = atom(0)
// Rooms this window saw disbanded. A `groups.list` that started before the
// disband (or a failed re-read keeping the last good list) never lists them again.
const $disbandedGatewayRooms = atom<ReadonlySet<string>>(new Set())

export function refreshGatewayGroupRooms(): void {
  $canonicalGroupListRevision.set($canonicalGroupListRevision.get() + 1)
}

/** Drop a disbanded room's key and every alias to it, then re-read the sidebar list. */
export function forgetCanonicalGroup({ connectionId, profile, roomId }: CanonicalGroupBinding): void {
  $disbandedGatewayRooms.set(new Set($disbandedGatewayRooms.get()).add(canonicalGroupKey({ connectionId, profile }, roomId)))
  $canonicalGroupBindings.set(Object.fromEntries(Object.entries($canonicalGroupBindings.get()).filter(([, binding]) =>
    binding.connectionId !== connectionId || binding.profile !== profile || binding.roomId !== roomId)))
  refreshGatewayGroupRooms()
}

function canonicalGroupKey(route: CanonicalGroupRoute, roomId: string): string {
  return `canonical:${encodeURIComponent(route.connectionId)}:${encodeURIComponent(route.profile)}:${roomId}`
}

export function registerCanonicalGroup(route: CanonicalGroupRoute, room: CanonicalRoom): string {
  const key = canonicalGroupKey(route, room.room_id)
  const current = $canonicalGroupBindings.get()[key]

  // The roster re-lists on a poll: an unchanged room must not wake every binding reader.
  if (current?.connectionId !== route.connectionId || current.profile !== route.profile || current.roomName !== room.name) {
    $canonicalGroupBindings.set({ ...$canonicalGroupBindings.get(), [key]: { ...route, roomId: room.room_id, roomName: room.name } })
  }

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

/** Every view string showing this room: its `canonical:` keys, and the name a
 *  name-opened tab and selection are recorded under (from its route-scoped alias). */
function canonicalGroupViews({ connectionId, profile, roomId }: CanonicalGroupBinding): Set<string> {
  const aliasPrefix = canonicalGroupAliasKey({ connectionId, profile }, '')
  const views = new Set<string>()

  for (const [key, binding] of Object.entries($canonicalGroupBindings.get())) {
    if (binding.connectionId === connectionId && binding.profile === profile && binding.roomId === roomId) {
      views.add(key.startsWith(aliasPrefix) ? key.slice(aliasPrefix.length) : key)
    }
  }

  return views
}

/** One gateway Group Chat as the roster lists it: its exact authority binding and
 *  the `canonical:` key the open path takes. */
export interface GatewayGroupRoom {
  binding: CanonicalGroupBinding
  key: string
  members: CanonicalRoomMember[]
  name: string
  room: CanonicalRoom
  /** Milliseconds; the room's last change, its recency in the roster. */
  updatedAt: number
}

/** Bind the room the user clicked, then hand back the key to open. Binding
 *  happens here, at the gesture, never in the list read: a slow read must not
 *  re-bind a room disbanded since, or rename one re-read since. */
export function bindGatewayGroupRoom({ binding, room }: GatewayGroupRoom): string {
  return registerCanonicalGroup(binding, room)
}

/** The active source's rooms. A backend that predates gateway Group Chats has none. */
async function listGatewayGroupRooms(route: CanonicalGroupRoute): Promise<GatewayGroupRoom[]> {
  let rooms: CanonicalRoom[]

  try {
    rooms = await listCanonicalRooms(route)
  } catch (error) {
    if (rpcMethodMissing(error)) {return []}
    throw error
  }

  const listed = rooms.map(room => ({
    binding: { ...route, roomId: room.room_id, roomName: room.name },
    key: canonicalGroupKey(route, room.room_id),
    members: room.members ?? [],
    name: room.name,
    room,
    updatedAt: (room.updated_at ?? 0) * 1000
  }))

  // A disbanded room this read no longer lists is gone at the gateway: stop hiding it.
  const prefix = canonicalGroupKey(route, '')
  const keys = new Set(listed.map(room => room.key))
  const hidden = $disbandedGatewayRooms.get()
  const kept = [...hidden].filter(key => !key.startsWith(prefix) || keys.has(key))

  if (kept.length !== hidden.size) {$disbandedGatewayRooms.set(new Set(kept))}

  return listed
}

const ROOMS_KEY = [ID, 'gateway-group-rooms']

/** Gateway Group Chats for the Bots roster, kept current without a refresh
 *  button: re-read on a poll, on every socket/source change and after a
 *  create or disband here. A failed read keeps the last good list (the roster
 *  shows rooms, never a transport error); a window whose connection has no
 *  registry id cannot bind a room to an authority, so it lists none. */
export function useGatewayGroupRooms(): GatewayGroupRoom[] {
  const connectionId = useValue(host.state.connectionId)
  const profile = useValue(host.state.profile)
  const gateway = useValue(host.state.gateway)
  const revision = useValue($canonicalGroupListRevision)
  const disbanded = useValue($disbandedGatewayRooms)

  const { data, refetch } = useQuery({
    queryKey: [...ROOMS_KEY, connectionId, profile],
    enabled: Boolean(connectionId && profile) && gateway === 'open',
    queryFn: () => listGatewayGroupRooms({ connectionId: connectionId!, profile }),
    refetchInterval: 10_000,
    retry: false,
    // Always stale, so the socket reopening (boot, reconnect, wake) re-reads at once.
    staleTime: 0
  })

  // A create or disband here re-reads now; a failed re-read keeps the last good list.
  useEffect(() => {
    if (revision && host.state.gateway.get() === 'open') {void refetch()}
  }, [revision, refetch])

  return useMemo(
    () => (connectionId ? (data ?? []).filter(room => !disbanded.has(room.key)) : []),
    [connectionId, data, disbanded]
  )
}

/** Disband on the room's own gateway, then retire it everywhere this window shows it.
 *  A refusal rejects before anything is dropped, so the row stays and the confirm shows why. */
export async function disbandGatewayGroupRoom(room: GatewayGroupRoom, doneMessage: string): Promise<void> {
  await canonicalGroupRequest(room.binding, 'groups.disband', { room_id: room.binding.roomId, cancel_id: randomUuid() })
  // Collect before forgetting: the bindings are the only record of which views alias this room.
  canonicalGroupViews(room.binding).add(room.key).forEach(closeGroupChatMainTab)
  forgetCanonicalGroup(room.binding)
  host.notify({ kind: 'success', message: doneMessage })
}
