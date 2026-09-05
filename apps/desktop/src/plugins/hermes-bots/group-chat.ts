/**
 * The group-chat room store: the room atoms, the durable record in plugin
 * storage, the bounded cross-client projection that rides the default
 * profile's ui_meta, and the small pure helpers every room surface shares
 * (identity, thread ids, entry append, the #93127 decision predicates).
 *
 * The sync projection lives here rather than in its own module because
 * updateGroupChat schedules it and the scheduler reads the same atom — one
 * store, one writer.
 */

import { atom, host } from '@hermes/plugin-sdk'

import { $botMeta, $lastRoster, botRosterKey } from './data'
import { groupMemberReferencesConnection, markOrphanedGroupMemberDescriptor } from './hygiene'
import { getPluginCtx } from './shared'
import type {
  Attachment,
  GroupChat,
  GroupHold,
  GroupMember,
  GroupMessage,
  GroupMessageAuthor,
  GroupPrompt,
  RosterRow
} from './types'

/** Optional secondary navigation inside the Bots pane (group-chat rooms). */

/** Group-chat rooms: { [group]: { log: [{from:{kind,name},text,at}], watermarks:{[member]:idx}, epoch, running } }.
 *  Log + watermarks persist via plugin storage; epoch/running are runtime-only. */
export const $groupChats = atom<Record<string, GroupChatRoom>>({})
/** Group whose room view is open in the Bots pane (secondary navigation
 *  inside the pane; a normal row click returns to the roster). */
export const $groupChatWorkspace = atom<null | string>(null)
/** Groups whose latest room activity mentions @user — the needs-you badge. */
export const $groupNeedsYou = atom<Record<string, boolean>>({})
// Pending prompts (clarify questions AND command approvals) raised inside
// hidden group-member sessions, keyed `${group}::${memberKey}` (#90694).
// Members run in invisible plumbing sessions, so a member's blocking prompt
// used to park server-side with no surface to answer it — the user saw
// "is thinking…" until the prompt timeout. The turn poll mirrors each
// member's `pending_clarify` / `pending_approval` resume fields in here;
// the room renders answer cards from it.
export const $groupClarify = atom<Record<string, GroupPrompt>>({})

const GROUP_CHAT_SYNC_META_KEY = 'hermes-bots-groups'
// Gateway ui_meta is capped after Python JSON serialization. Keep a healthy
// margin below that limit because Python escapes Unicode while JS does not.
const GROUP_CHAT_SYNC_MAX_BYTES = 48000
const GROUP_CHAT_SYNC_MESSAGES = 16
const GROUP_CHAT_SYNC_TEXT_CHARS = 1200
const GROUP_CHAT_SYNC_IMAGE_CHARS = 24000
let groupChatSyncTimer: ReturnType<typeof setTimeout> | null = null

/** One room inside the bounded ui_meta projection: a compacted log plus the
 *  identity fields, without any of `GroupChat`'s runtime/orchestration state. */
interface GroupChatSyncRoom {
  image?: null | string
  log: GroupMessage[]
  members?: GroupMember[]
  name?: string
  revision?: number
  roomId?: string
}

/** The v3 envelope stored under the default profile's `hermes-bots-groups`
 *  ui_meta key. `deleted` maps a room key to its tombstone revision. */
interface GroupChatSyncSnapshot {
  deleted?: Record<string, number>
  rooms: Record<string, GroupChatSyncRoom>
  updatedAt?: number
  version: number
}

/** A queued publish for one gateway, coalesced while the debounce runs. */
interface GroupChatSyncJob {
  allowEmpty?: boolean
  changedRooms?: string[]
  connectionId: string
  deletedRooms?: string[]
}
// Fan-out scheduler state, keyed by gateway connectionId ('' = active/local).
// Every connected gateway carries the full projection so a room survives any
// single gateway being removed and surfaces on every remote backend.
const groupChatSyncPendingByConnection = new Map<string, GroupChatSyncJob>()
const groupChatSyncInFlightConnections = new Set<string>()
const groupChatSyncRetryTimers = new Map<string, ReturnType<typeof setTimeout>>()
const groupChatSyncRetryCounts = new Map<string, number>()
export let groupChatSyncDisposed = false

/** Conservative byte count for the gateway's ensure_ascii JSON encoding.
 *  Python also inserts separator spaces, so reserve one extra byte per JS
 *  structural separator on top of escaped Unicode code-point widths. */
export function groupChatGatewayJsonSize(value: unknown) {
  const json = JSON.stringify(value)
  let bytes = 0

  for (const character of json) {
    // Non-null: string iteration yields whole code points, never an empty string.
    const codePoint = character.codePointAt(0)!

    if (codePoint <= 0x7f) {
      bytes += 1

      if (character === ',' || character === ':') {
        bytes += 1
      }
    } else {
      bytes += codePoint <= 0xffff ? 6 : 12
    }
  }

  return bytes
}

/** Durable room identity for the sync projection. Rooms minted on current
 *  builds carry an immutable roomId; the projection keys rooms by
 *  `id:<roomId>` so rename is a display-name edit, not a distributed
 *  delete+create, and disband tombstones follow the room itself. Legacy
 *  rooms (no roomId) fall back to `name:<name>` keys with the older
 *  revision-gated tombstone semantics. */
export function groupChatRoomKey(name: string, room: GroupChat) {
  return typeof room?.roomId === 'string' && room.roomId ? `id:${room.roomId}` : `name:${String(name)}`
}

/** Lift any historical projection shape (v1 wall-clock, v2 name-keyed) to
 *  the v3 room-key shape so one merge path serves mixed-version fleets. */
function normalizeGroupChatSyncSnapshot(snapshot: GroupChatSyncSnapshot | null | undefined): GroupChatSyncSnapshot {
  if (!snapshot || typeof snapshot !== 'object') {
    return {
      version: 3,
      rooms: {},
      deleted: {}
    }
  }

  if (Number(snapshot.version || 0) >= 3) {
    return {
      version: 3,
      updatedAt: Number(snapshot.updatedAt || 0),
      rooms: snapshot.rooms && typeof snapshot.rooms === 'object' ? snapshot.rooms : {},
      deleted: snapshot.deleted && typeof snapshot.deleted === 'object' ? snapshot.deleted : {}
    }
  }

  const rooms: Record<string, GroupChatSyncRoom> = {}

  for (const [name, room] of Object.entries(snapshot.rooms || {})) {
    if (!room || !Array.isArray(room.log)) {
      continue
    }

    rooms[`name:${name}`] = {
      ...room,
      name
    }
  }

  const deleted: Record<string, number> = {}

  for (const [name, at] of Object.entries(snapshot.deleted || {})) {
    // v1 tombstones carried wall-clock ms, not gateway revisions — they must
    // not outrank real revisions.
    deleted[`name:${name}`] = Number(snapshot.version || 0) >= 2 ? Math.max(0, Number(at || 0)) : 0
  }

  return {
    version: 3,
    updatedAt: Number(snapshot.updatedAt || 0),
    rooms,
    deleted
  }
}

/** Compact, display-oriented copy of Desktop's room log for gateway clients.
 *  The live orchestration state stays in plugin storage; this bounded mirror
 *  rides the default profile's ui_meta so mobile can show the same messages.
 *  Newest rooms/messages win when the profile metadata size cap is reached. */
export function groupChatSyncSnapshot(
  // `revision` is the pre-`syncRevision` field name, still read below as a
  // fallback for rooms hydrated from an older plugin-storage record.
  all: Record<string, GroupChat & { revision?: number }> = $groupChats.get(),
  deleted: Record<string, number> = {}
): GroupChatSyncSnapshot {
  const ranked = Object.entries(all || {})
    // Empty runtime tombstones are used to stop an in-flight room after
    // disband. They are not real rooms and must never reappear on mobile.
    .filter(([, room]) => room && Array.isArray(room.log) && room.log.length > 0)
    .sort(([, left], [, right]) => {
      const leftAt = Number(left.log[left.log.length - 1]?.at || 0)
      const rightAt = Number(right.log[right.log.length - 1]?.at || 0)

      return rightAt - leftAt
    })

  const rooms: Record<string, GroupChatSyncRoom> = {}

  const boundedDeleted = Object.fromEntries(
    Object.entries(deleted)
      .sort(([, left], [, right]) => Number(right || 0) - Number(left || 0))
      .slice(0, 64)
  )

  const envelope: GroupChatSyncSnapshot = {
    version: 3,
    updatedAt: Date.now(),
    rooms,
    ...(Object.keys(boundedDeleted).length
      ? {
          deleted: boundedDeleted
        }
      : {})
  }

  for (const [name, room] of ranked) {
    const log: GroupMessage[] = room.log.slice(-GROUP_CHAT_SYNC_MESSAGES).map(entry => ({
      ...(entry?.id
        ? {
            id: String(entry.id).slice(0, 160)
          }
        : {}),
      from: {
        kind: entry?.from?.kind === 'member' ? 'member' : 'user',
        name: String(entry?.from?.name || (entry?.from?.kind === 'member' ? 'Bot' : 'You')).slice(0, 128),
        ...(entry?.from?.source
          ? {
              source: String(entry.from.source).slice(0, 128)
            }
          : {})
      },
      text: String(entry?.text || '').slice(0, GROUP_CHAT_SYNC_TEXT_CHARS),
      at: Number(entry?.at || 0),
      ...(entry?.thread
        ? {
            thread: String(entry.thread).slice(0, 128)
          }
        : {})
    }))

    const compact: GroupChatSyncRoom = {
      name: String(name).slice(0, 64),
      ...(typeof room?.roomId === 'string' && room.roomId
        ? {
            roomId: String(room.roomId).slice(0, 128)
          }
        : {}),
      log,
      revision: Math.max(0, Number(room?.syncRevision ?? room?.revision ?? 0)),
      members: (Array.isArray(room.members) ? room.members : []).slice(0, GROUP_CHAT_MAX_MEMBERS).map(member => ({
        name: String(member?.name || '').slice(0, 128),
        ...(member?.handle
          ? {
              handle: String(member.handle).slice(0, 128)
            }
          : {}),
        ...(member?.connectionId
          ? {
              connectionId: String(member.connectionId).slice(0, 128)
            }
          : {}),
        ...(member?.connectionKind
          ? {
              connectionKind: String(member.connectionKind).slice(0, 64)
            }
          : {}),
        ...(member?.connectionLabel
          ? {
              connectionLabel: String(member.connectionLabel).slice(0, 128)
            }
          : {}),
        ...(member?.sourceScoped
          ? {
              sourceScoped: true
            }
          : {})
      })),
      ...(typeof room?.image === 'string' && room.image.length <= GROUP_CHAT_SYNC_IMAGE_CHARS
        ? {
            image: room.image
          }
        : {})
    }

    const key = groupChatRoomKey(name, room)
    rooms[key] = compact

    while (compact.log.length > 1 && groupChatGatewayJsonSize(envelope) > GROUP_CHAT_SYNC_MAX_BYTES) {
      compact.log.shift()
    }

    if (compact.image && groupChatGatewayJsonSize(envelope) > GROUP_CHAT_SYNC_MAX_BYTES) {
      delete compact.image
    }

    if (groupChatGatewayJsonSize(envelope) > GROUP_CHAT_SYNC_MAX_BYTES) {
      delete rooms[key]
    }
  }

  return envelope
}

function groupChatSyncEntryKey(entry: GroupMessage) {
  if (entry?.id) {
    return `id:${String(entry.id)}`
  }

  return JSON.stringify([
    Number(entry?.at || 0),
    String(entry?.from?.kind || ''),
    String(entry?.from?.name || ''),
    String(entry?.from?.source || ''),
    // Threadless entries (pre-thread rooms, older Desktop builds) get
    // SYNTHETIC `legacy-N` ids from assignLegacyThreads. Those ids are
    // position-derived — not stable across a gateway round-trip (the
    // projection copy may be threadless or numbered differently). Collapse
    // the whole synthetic family to one bucket, or the merge duplicates
    // every id-less entry — shifting watermarks and manufacturing phantom
    // member turns that re-submit into busy sessions.
    String(entry?.thread || 'legacy').replace(/^legacy-\d+$/, 'legacy'),
    String(entry?.text || '')
  ])
}

/** Members dedupe on durable identity — the same (connectionId, name) pair
 *  botRosterKey seats them by everywhere else. `connectionLabel` and `handle`
 *  are display strings each machine re-derives (a connection rename, an older
 *  build with no handle), so keying on them seats one bot twice; both copies
 *  then answer to a single groupMemberKey in watermarks/sessions/stranded and
 *  the round engine gives that bot two turns. Deliberately unconditional,
 *  unlike groupMemberKey: the projection stamps `remoteSource` onto members it
 *  merges in, so a scoped/unscoped branch would fork a member from its own
 *  previously-merged copy. */
function groupChatSyncMemberKey(member: GroupMember) {
  return botRosterKey(member)
}

/** Merge two bounded projections without treating an absent room/message as
 *  deletion. Rooms are identified by durable room keys (id:<roomId> when the
 *  room carries one), so a rename is a same-key field update — never a
 *  distributed delete+create — and a disband tombstone follows the room
 *  itself. Gateway revisions order identity/membership/picture and
 *  tombstones; stable message ids make concurrent log union idempotent.
 *  `changedRooms`/`deletedRooms` accept display names or room keys. */
export function mergeGroupChatSyncSnapshots(
  remote: GroupChatSyncSnapshot | null | undefined,
  local: GroupChatSyncSnapshot | null | undefined,
  {
    changedRooms = [],
    deletedRooms = [],
    writeRevision = 0
  }: { changedRooms?: string[]; deletedRooms?: string[]; writeRevision?: number } = {}
) {
  const remoteNorm = normalizeGroupChatSyncSnapshot(remote)
  const localNorm = normalizeGroupChatSyncSnapshot(local)

  const keysFor = (label: string, norm: GroupChatSyncSnapshot) => {
    const keys = new Set<string>()

    for (const [key, room] of Object.entries(norm.rooms || {})) {
      if (key === label || String(room?.name || '') === label || key === `name:${label}`) {
        keys.add(key)
      }
    }

    if (String(label).startsWith('id:') || String(label).startsWith('name:')) {
      keys.add(label)
    } else if (!keys.size) {
      keys.add(`name:${label}`)
    }

    return keys
  }

  const changed = new Set<string>()

  for (const label of changedRooms) {
    for (const key of keysFor(label, localNorm)) {
      changed.add(key)
    }
  }

  const deleted: Record<string, number> = {}

  for (const source of [remoteNorm, localNorm]) {
    for (const [key, at] of Object.entries(source.deleted || {})) {
      deleted[key] = Math.max(Number(deleted[key] || 0), Math.max(0, Number(at || 0)))
    }
  }

  for (const label of deletedRooms) {
    for (const key of new Set([...keysFor(label, remoteNorm), ...keysFor(label, localNorm)])) {
      // Rename passes changedRooms:[newName] + deletedRooms:[oldName]. For an
      // id-keyed room both labels resolve to the SAME durable key (the remote
      // copy still carries the old display name), and id tombstones are
      // final — so tombstoning here would kill the room being renamed. A key
      // that is being written this cycle is a rename target, not a disband.
      if (changed.has(key)) {
        continue
      }

      deleted[key] = Math.max(Number(deleted[key] || 0), Number(writeRevision || 0))
    }
  }

  const rooms: Record<string, GroupChatSyncRoom> = {}
  const roomKeys = new Set([...Object.keys(remoteNorm.rooms || {}), ...Object.keys(localNorm.rooms || {})])

  for (const key of roomKeys) {
    const remoteRoom = remoteNorm.rooms?.[key]
    const localRoom = localNorm.rooms?.[key]

    if ((!remoteRoom || !Array.isArray(remoteRoom.log)) && (!localRoom || !Array.isArray(localRoom.log))) {
      continue
    }

    const remoteRevision = Math.max(0, Number(remoteRoom?.revision || 0))

    const localRevision = changed.has(key)
      ? Math.max(0, Number(writeRevision || 0))
      : Math.max(0, Number(localRoom?.revision || 0))

    const entries = new Map<string, GroupMessage>()

    for (const entry of [...(remoteRoom?.log || []), ...(localRoom?.log || [])]) {
      entries.set(groupChatSyncEntryKey(entry), entry)
    }

    // Identity fields (display name, membership, picture) follow the higher
    // revision; a tie unions members and prefers the local writer's fields.
    let identity: GroupChatSyncRoom | undefined
    let members: GroupMember[]
    let image: null | string | undefined

    if (localRevision > remoteRevision) {
      identity = localRoom
      members = [...(localRoom?.members || [])]
      image = localRoom?.image
    } else if (remoteRevision > localRevision) {
      identity = remoteRoom
      members = [...(remoteRoom?.members || [])]
      image = remoteRoom?.image
    } else {
      identity = localRoom || remoteRoom
      const byId = new Map<string, GroupMember>()

      for (const member of [...(remoteRoom?.members || []), ...(localRoom?.members || [])]) {
        byId.set(groupChatSyncMemberKey(member), member)
      }

      members = [...byId.values()]
      image = Object.prototype.hasOwnProperty.call(localRoom || {}, 'image') ? localRoom.image : remoteRoom?.image
    }

    rooms[key] = {
      ...(identity?.name
        ? {
            name: identity.name
          }
        : {}),
      ...(identity?.roomId || (key.startsWith('id:') ? key.slice(3) : '')
        ? {
            roomId: identity?.roomId || key.slice(3)
          }
        : {}),
      log: [...entries.values()].sort((left, right) => {
        const byTime = Number(left?.at || 0) - Number(right?.at || 0)

        return byTime || groupChatSyncEntryKey(left).localeCompare(groupChatSyncEntryKey(right))
      }),
      members,
      revision: Math.max(remoteRevision, localRevision),
      ...(typeof image === 'string' && image
        ? {
            image
          }
        : {})
    }
  }

  for (const [key, deletedRevision] of Object.entries(deleted)) {
    if (key.startsWith('id:')) {
      // Tombstones for id-keyed rooms are FINAL: the roomId is minted once
      // and never reused (same-name recreation mints a fresh id), so a
      // resurrect-by-revision race is structurally impossible. Keep the
      // tombstone even when a lagging gateway's copy carries a higher
      // revision — that copy is the resurrection this exists to prevent.
      delete rooms[key]
    } else if (Number(deletedRevision || 0) >= Number(rooms[key]?.revision || 0)) {
      delete rooms[key]
    } else {
      delete deleted[key]
    }
  }

  return groupChatSyncEnvelope(rooms, deleted)
}

/** Assemble + size-bound a v3 envelope from already-compacted rooms. */
function groupChatSyncEnvelope(
  rooms: Record<string, GroupChatSyncRoom>,
  deleted: Record<string, number> = {}
): GroupChatSyncSnapshot {
  const boundedDeleted = Object.fromEntries(
    Object.entries(deleted)
      .sort(([, left], [, right]) => Number(right || 0) - Number(left || 0))
      .slice(0, 64)
  )

  const envelope: GroupChatSyncSnapshot = {
    version: 3,
    updatedAt: Date.now(),
    rooms,
    ...(Object.keys(boundedDeleted).length
      ? {
          deleted: boundedDeleted
        }
      : {})
  }

  const ranked = Object.entries(rooms).sort(([, left], [, right]) => {
    const leftAt = Number(left?.log?.[left.log.length - 1]?.at || 0)
    const rightAt = Number(right?.log?.[right.log.length - 1]?.at || 0)

    return leftAt - rightAt
  })

  for (const [key, room] of ranked) {
    while ((room.log?.length || 0) > 1 && groupChatGatewayJsonSize(envelope) > GROUP_CHAT_SYNC_MAX_BYTES) {
      room.log.shift()
    }

    if (room.image && groupChatGatewayJsonSize(envelope) > GROUP_CHAT_SYNC_MAX_BYTES) {
      delete room.image
    }

    if (groupChatGatewayJsonSize(envelope) > GROUP_CHAT_SYNC_MAX_BYTES) {
      delete rooms[key]
    }
  }

  return envelope
}

/** Merge the gateway's bounded display projection into Desktop's richer room
 *  state without discarding local session/watermark/runtime fields. Missing
 *  remote rooms/messages are not deletions; only explicit tombstones remove a
 *  room, and a genuinely newer local message wins over a stale tombstone. */
export function mergeRemoteGroupChatSnapshotIntoRooms(
  remote: GroupChatSyncSnapshot | null | undefined,
  current: Record<string, GroupChat> = $groupChats.get(),
  { preserveRooms = [], deletedRooms = [] }: { deletedRooms?: string[]; preserveRooms?: string[] } = {}
) {
  const remoteNorm = normalizeGroupChatSyncSnapshot(remote)

  const rooms: Record<string, GroupChat> = {
    ...(current || {})
  }

  const preserved = new Set(preserveRooms)
  const locallyDeleted = new Set(deletedRooms)

  // Local rooms indexed by durable identity so an id-keyed projection room
  // finds its local twin even when the display name changed remotely.
  const localByRoomId = new Map<string, string>()

  for (const [name, room] of Object.entries(rooms)) {
    if (typeof room?.roomId === 'string' && room.roomId) {
      localByRoomId.set(room.roomId, name)
    }
  }

  for (const [key, projected] of Object.entries(remoteNorm.rooms || {})) {
    if (!projected || !Array.isArray(projected.log)) {
      continue
    }

    const projectedRoomId = projected.roomId || (key.startsWith('id:') ? key.slice(3) : null)

    const localName =
      projectedRoomId && localByRoomId.has(projectedRoomId)
        ? localByRoomId.get(projectedRoomId)
        : projected.name && rooms[projected.name]
          ? projected.name
          : null

    const displayName = String(projected.name || localName || (key.startsWith('name:') ? key.slice(5) : key))

    if (locallyDeleted.has(displayName) || (localName && locallyDeleted.has(localName))) {
      // Mid-rename guard: the remote copy may still be under the OLD display
      // name while the local record was already re-keyed (same roomId, new
      // name). That old name sits in deletedRooms, but the local record is
      // the rename in flight — deleting it here would kill the renamed room.
      if (localName && localName !== displayName && !locallyDeleted.has(localName)) {
        continue
      }

      delete rooms[displayName]

      if (localName) {
        delete rooms[localName]
      }

      continue
    }

    const existing = (localName ? rooms[localName] : rooms[displayName]) || {}
    const remoteRevision = Math.max(0, Number(projected.revision || 0))
    const localRevision = Math.max(0, Number(existing.syncRevision || 0))

    const entries = new Map<string, GroupMessage>(
      (Array.isArray(existing.log) ? existing.log : []).map(entry => [groupChatSyncEntryKey(entry), entry])
    )

    const members = new Map<string, GroupMember>(
      (Array.isArray(existing.members) ? existing.members : []).map(member => [groupChatSyncMemberKey(member), member])
    )

    for (const entry of projected.log) {
      const entryKey = groupChatSyncEntryKey(entry)

      // The projection is COMPACT (truncated text, no images). When the same
      // entry exists locally, the local rich copy is authoritative — merging
      // the compact twin over it would strip attachments and retrigger
      // watermark deltas for members that already saw it (phantom rounds).
      if (!entries.has(entryKey)) {
        entries.set(entryKey, entry)
      }
    }

    const isPreserved = preserved.has(displayName) || (localName && preserved.has(localName))

    if (!isPreserved) {
      if (remoteRevision > localRevision) {
        members.clear()
      }

      for (const member of Array.isArray(projected.members) ? projected.members : []) {
        members.set(groupChatSyncMemberKey(member), {
          ...member,
          remoteSource: true
        })
      }
    }

    const log = assignLegacyThreads(
      [...entries.values()].sort((left, right) => {
        const byTime = Number(left?.at || 0) - Number(right?.at || 0)

        return byTime || groupChatSyncEntryKey(left).localeCompare(groupChatSyncEntryKey(right))
      })
    )

    const bounded = trimGroupChatLog(log, existing.watermarks || {})

    // A remote rename with a higher revision moves the local record to the
    // new display name; local views keyed by the old name follow on the
    // next repaint (roster derives from $groupChats keys).
    const targetName = !isPreserved && remoteRevision > localRevision ? displayName : localName || displayName

    if (localName && targetName !== localName) {
      delete rooms[localName]
    }

    rooms[targetName] = {
      ...existing,
      log: bounded.log,
      watermarks: bounded.watermarks,
      sessions: existing.sessions && typeof existing.sessions === 'object' ? existing.sessions : {},
      stranded: existing.stranded && typeof existing.stranded === 'object' ? existing.stranded : {},
      members: [...members.values()],
      ...(projectedRoomId || existing.roomId
        ? {
            roomId: existing.roomId || projectedRoomId
          }
        : {}),
      image: isPreserved
        ? existing.image || null
        : remoteRevision >= localRevision && Object.prototype.hasOwnProperty.call(projected, 'image')
          ? projected.image || null
          : existing.image || null,
      syncRevision: isPreserved ? localRevision : Math.max(remoteRevision, localRevision),
      epoch: Number(existing.epoch || 0),
      running: Boolean(existing.running)
    }
  }

  for (const [key, deletedAt] of Object.entries(remoteNorm.deleted || {})) {
    const deletedRoomId = key.startsWith('id:') ? key.slice(3) : null

    const targetName =
      deletedRoomId && localByRoomId.has(deletedRoomId)
        ? localByRoomId.get(deletedRoomId)
        : key.startsWith('name:')
          ? key.slice(5)
          : null

    if (!targetName || preserved.has(targetName)) {
      continue
    }

    if (deletedRoomId) {
      // Id tombstones are final — the id is never reused, so there is no
      // legitimate higher-revision recreation to protect.
      delete rooms[targetName]
    } else {
      const deletedRevision = Math.max(0, Number(deletedAt || 0))

      if (deletedRevision >= Number(rooms[targetName]?.syncRevision || 0)) {
        delete rooms[targetName]
      }
    }
  }

  for (const name of locallyDeleted) {
    delete rooms[name]
  }

  return rooms
}

export function durableGroupChatRooms(all: Record<string, GroupChat> = $groupChats.get()) {
  const durable: Record<string, GroupChat> = {}

  for (const [name, room] of Object.entries(all || {})) {
    if (!room || !Array.isArray(room.log)) {
      continue
    }

    // Disband tombstones are runtime-only coordination state (they hold the
    // epoch bump for an in-flight drive). Persisting one would resurrect the
    // room as an empty record on the next load AND keep its name "taken" for
    // same-name recreates. Mirrors updateGroupChat's inline durable map.
    if (room.tombstone) {
      continue
    }

    durable[name] = {
      log: room.log,
      watermarks: room.watermarks || {},
      sessions: room.sessions || {},
      stranded: room.stranded || {},
      ...(room.pendingThread ? { pendingThread: room.pendingThread } : {}),
      members: Array.isArray(room.members) ? room.members : [],
      // Immutable room identity: without this, a room merged in via the
      // remote-sync path (the only caller of this function) loses its
      // roomId on the next cold hydrate and falls back to legacy
      // name-keyed identity — same field updateGroupChat's inline map
      // already carries.
      roomId: typeof room.roomId === 'string' && room.roomId ? room.roomId : null,
      image: room.image || null,
      syncRevision: Math.max(0, Number(room.syncRevision || 0))
    }
  }

  return durable
}

export function persistGroupChatRooms(all: Record<string, GroupChat> = $groupChats.get()) {
  try {
    return Promise.resolve(getPluginCtx()?.storage?.set?.('group-chats', durableGroupChatRooms(all))).catch(
      () => undefined
    )
  } catch {
    return Promise.resolve()
  }
}

/** Register-removed sweep: annotate (not delete) every persisted group-chat
 *  member owned by the deleted connection, in the atom AND plugin storage.
 *  Writes ride updateGroupChat so the durable record keeps its full shape
 *  (sessionOwners, holds — durableGroupChatRooms would drop them).
 *  Returns whether anything changed. */
export function sweepGroupChatMembersForRemovedConnection(connectionId: string) {
  const id = String(connectionId || '').trim()

  if (!id) {
    return false
  }

  let changed = false

  for (const [name, room] of Object.entries($groupChats.get())) {
    const members = Array.isArray(room?.members) ? room.members : []

    if (!members.some(member => groupMemberReferencesConnection(member, id) && !member?.sourceMissing)) {
      continue
    }

    changed = true
    updateGroupChat(name, (current: GroupChat) => ({
      ...current,
      members: (Array.isArray(current.members) ? current.members : []).map(member =>
        groupMemberReferencesConnection(member, id) ? markOrphanedGroupMemberDescriptor(member) : member
      )
    }))
  }

  return changed
}

function groupChatSyncConnectionId() {
  return String(host.state.connectionId?.get?.() || host.activeConnectionId?.() || '')
}

/** Route a sync job back to the gateway that was active when it was queued.
 *  A foreground switch during debounce must not write the old snapshot into
 *  the newly active gateway. */
async function groupChatSyncRequest<T>(
  job: GroupChatSyncJob,
  method: string,
  params: Record<string, unknown>
): Promise<T> {
  if (job.connectionId && typeof host.profileRoutes === 'function' && typeof host.requestProfile === 'function') {
    const routes = await host.profileRoutes()

    const route = (Array.isArray(routes) ? routes : []).find(candidate => {
      const profile = String(candidate?.targetProfile || candidate?.profile || '')

      return String(candidate?.connectionId || '') === job.connectionId && profile === 'default'
    })

    if (route) {
      return host.requestProfile(route, method, params)
    }
  }

  const currentConnectionId = groupChatSyncConnectionId()

  if (job.connectionId && currentConnectionId && job.connectionId !== currentConnectionId) {
    throw new Error('Group chat gateway changed before sync')
  }

  return host.request(method, params)
}

async function groupChatRemoteSnapshot(job: GroupChatSyncJob) {
  const result = await groupChatSyncRequest<{ profiles?: RosterRow[] }>(job, 'profiles.list', {
    include_sessions: false
  })

  const profile = (Array.isArray(result?.profiles) ? result.profiles : []).find(row => row?.name === 'default')
  const snapshot = profile?.ui_meta?.[GROUP_CHAT_SYNC_META_KEY] as GroupChatSyncSnapshot | undefined
  const supportsCas = Boolean(profile && Object.prototype.hasOwnProperty.call(profile, 'ui_meta_revisions'))

  return {
    snapshot: snapshot && typeof snapshot === 'object' && !Array.isArray(snapshot) ? snapshot : null,
    revision: Math.max(0, Number(profile?.ui_meta_revisions?.[GROUP_CHAT_SYNC_META_KEY] || 0)),
    supportsCas
  }
}

/** Pull the shared room projection into this Desktop before it publishes any
 *  local state. This is the receive half of the client-only sync contract. */
export async function pullGroupChatServerState(connectionId: string = groupChatSyncConnectionId()) {
  const { snapshot: remote } = await groupChatRemoteSnapshot({
    connectionId
  })

  if (!remote) {
    return false
  }

  const pending = groupChatSyncPendingByConnection.get(String(connectionId || ''))

  const merged = mergeRemoteGroupChatSnapshotIntoRooms(remote, $groupChats.get(), {
    preserveRooms: pending?.changedRooms || [],
    deletedRooms: pending?.deletedRooms || []
  })

  $groupChats.set(merged)
  await persistGroupChatRooms(merged)

  return true
}

function groupChatSyncBackoff(connectionId: string) {
  const count = Number(groupChatSyncRetryCounts.get(connectionId) || 0)

  return Math.min(30000, 1000 * 2 ** Math.min(count, 5))
}

function mergeGroupChatSyncJobs(existing: GroupChatSyncJob | undefined, incoming: GroupChatSyncJob): GroupChatSyncJob {
  if (!existing || existing.connectionId !== incoming.connectionId) {
    return incoming
  }

  return {
    connectionId: incoming.connectionId,
    allowEmpty: Boolean(existing.allowEmpty || incoming.allowEmpty),
    changedRooms: [...new Set([...(existing.changedRooms || []), ...(incoming.changedRooms || [])])],
    deletedRooms: [...new Set([...(existing.deletedRooms || []), ...(incoming.deletedRooms || [])])]
  }
}

function groupChatSyncPayloadEqual(
  left: GroupChatSyncSnapshot | null | undefined,
  right: GroupChatSyncSnapshot | null | undefined
) {
  return (
    JSON.stringify(left?.rooms || {}) === JSON.stringify(right?.rooms || {}) &&
    JSON.stringify(left?.deleted || {}) === JSON.stringify(right?.deleted || {})
  )
}

/** Every default-profile gateway route this Desktop can currently reach.
 *  The projection fans out to ALL of them, so any single gateway can die or
 *  be removed without losing the shared room state, and gateway-only
 *  clients (Hermes Go, headless backends) see rooms regardless of which
 *  gateway a Desktop was foregrounding when the room was used. */
async function groupChatSyncTargetConnections() {
  const targets = new Set<string>()
  const active = groupChatSyncConnectionId()
  targets.add(String(active || ''))

  if (typeof host.profileRoutes === 'function' && typeof host.requestProfile === 'function') {
    try {
      const routes = await host.profileRoutes()

      for (const route of Array.isArray(routes) ? routes : []) {
        const profile = String(route?.targetProfile || route?.profile || '')
        const connectionId = String(route?.connectionId || '')

        if (profile === 'default' && connectionId) {
          targets.add(connectionId)
        }
      }
    } catch {
      // Route inventory unavailable — the active gateway alone still syncs.
    }
  }

  return [...targets]
}

async function flushGroupChatServerSync(connectionId?: string) {
  if (connectionId === undefined) {
    // Drain every connection with pending work.
    for (const pendingId of [...groupChatSyncPendingByConnection.keys()]) {
      void flushGroupChatServerSync(pendingId)
    }

    return
  }

  const id = String(connectionId || '')

  if (groupChatSyncDisposed || groupChatSyncInFlightConnections.has(id) || !groupChatSyncPendingByConnection.has(id)) {
    return
  }

  // Non-null: the `has(id)` guard directly above is the entry condition.
  const job = groupChatSyncPendingByConnection.get(id)!
  groupChatSyncPendingByConnection.delete(id)
  groupChatSyncInFlightConnections.add(id)

  try {
    const remoteState = await groupChatRemoteSnapshot(job)
    const local = groupChatSyncSnapshot($groupChats.get())
    const writeRevision = remoteState.revision + 1

    const snapshot = mergeGroupChatSyncSnapshots(remoteState.snapshot, local, {
      changedRooms: job.changedRooms,
      deletedRooms: job.deletedRooms,
      writeRevision
    })

    // Reconnect/startup reconciliation often discovers that the gateway
    // already holds the exact merged projection. Avoid advancing a revision
    // merely because a view reopened.
    if (
      !(job.changedRooms || []).length &&
      !(job.deletedRooms || []).length &&
      groupChatSyncPayloadEqual(snapshot, remoteState.snapshot)
    ) {
      if (remoteState.snapshot) {
        const pending = groupChatSyncPendingByConnection.get(id)

        const mergedRooms = mergeRemoteGroupChatSnapshotIntoRooms(remoteState.snapshot, $groupChats.get(), {
          preserveRooms: pending?.changedRooms || [],
          deletedRooms: pending?.deletedRooms || []
        })

        $groupChats.set(mergedRooms)
        await persistGroupChatRooms(mergedRooms)
      }

      groupChatSyncRetryCounts.delete(id)

      return
    }

    const configureParams: {
      name: string
      ui_meta: Record<string, GroupChatSyncSnapshot>
      ui_meta_expected_revisions?: Record<string, number>
    } = {
      name: 'default',
      ui_meta: {
        [GROUP_CHAT_SYNC_META_KEY]: snapshot
      }
    }

    if (remoteState.supportsCas) {
      configureParams.ui_meta_expected_revisions = {
        [GROUP_CHAT_SYNC_META_KEY]: remoteState.revision
      }
    }

    const result = await groupChatSyncRequest<{
      applied?: { ui_meta?: boolean; ui_meta_revisions?: Record<string, number> }
    }>(job, 'profiles.configure', configureParams)

    if (result?.applied?.ui_meta !== true) {
      throw new Error('Gateway rejected group chat ui_meta')
    }

    if (
      remoteState.supportsCas &&
      Number(result?.applied?.ui_meta_revisions?.[GROUP_CHAT_SYNC_META_KEY] || 0) !== writeRevision
    ) {
      throw new Error('Gateway did not advance group chat ui_meta revision')
    }

    const confirmedState = await groupChatRemoteSnapshot(job)

    if (remoteState.supportsCas && confirmedState.revision < writeRevision) {
      throw new Error('Group chat ui_meta revision missing after read-back')
    }

    if (confirmedState.snapshot) {
      const pending = groupChatSyncPendingByConnection.get(id)

      const mergedRooms = mergeRemoteGroupChatSnapshotIntoRooms(confirmedState.snapshot, $groupChats.get(), {
        preserveRooms: pending?.changedRooms || [],
        deletedRooms: pending?.deletedRooms || []
      })

      $groupChats.set(mergedRooms)
      await persistGroupChatRooms(mergedRooms)
    }

    groupChatSyncRetryCounts.delete(id)
  } catch {
    if (!groupChatSyncDisposed) {
      const retries = Number(groupChatSyncRetryCounts.get(id) || 0) + 1

      // A gateway that was REMOVED (not just flaky) has no route anymore and
      // would otherwise retry forever. Give up after the backoff ladder tops
      // out; local storage remains authoritative and a future reconnect of
      // that gateway re-seeds it via the gateway-transition pull/publish.
      if (retries > 8) {
        groupChatSyncRetryCounts.delete(id)

        return
      }

      groupChatSyncPendingByConnection.set(id, mergeGroupChatSyncJobs(groupChatSyncPendingByConnection.get(id), job))
      groupChatSyncRetryCounts.set(id, retries)

      if (typeof setTimeout === 'function' && !groupChatSyncRetryTimers.has(id)) {
        groupChatSyncRetryTimers.set(
          id,
          setTimeout(() => {
            groupChatSyncRetryTimers.delete(id)
            void flushGroupChatServerSync(id)
          }, groupChatSyncBackoff(id))
        )
      }
    }
  } finally {
    groupChatSyncInFlightConnections.delete(id)

    if (groupChatSyncPendingByConnection.has(id) && !groupChatSyncRetryTimers.has(id) && !groupChatSyncDisposed) {
      void flushGroupChatServerSync(id)
    }
  }
}

export function stopGroupChatServerSync() {
  groupChatSyncDisposed = true
  groupChatSyncPendingByConnection.clear()

  if (groupChatSyncTimer !== null) {
    clearTimeout(groupChatSyncTimer)
    groupChatSyncTimer = null
  }

  for (const timer of groupChatSyncRetryTimers.values()) {
    clearTimeout(timer)
  }

  groupChatSyncRetryTimers.clear()
  groupChatSyncRetryCounts.clear()
}

/** Debounced, pull-merge-write server mirror, fanned out to every reachable
 *  default-profile gateway. Local storage keeps the complete orchestration
 *  log; ui_meta is a bounded cross-client projection per gateway, each with
 *  its own CAS revision stream. */
export function scheduleGroupChatServerSync(
  all: Record<string, GroupChat> = $groupChats.get(),
  {
    allowEmpty = false,
    changedRooms = [],
    deletedRooms = []
  }: { allowEmpty?: boolean; changedRooms?: string[]; deletedRooms?: string[] } = {}
) {
  // Browser shells provide timers; source-level VM tests and older embedded
  // hosts may not. Room persistence must never break the surrounding gateway
  // lifecycle when the optional mirror cannot be scheduled.
  if (typeof setTimeout !== 'function') {
    return
  }

  const snapshot = groupChatSyncSnapshot(all)

  // A newly installed Desktop has no local room cache. Publishing that empty
  // state on hydrate/reconnect would erase a valid mirror produced elsewhere.
  // Only an explicit final-room disband is allowed to clear the projection.
  if (Object.keys(snapshot.rooms).length === 0 && !allowEmpty) {
    return
  }

  if (groupChatSyncTimer !== null) {
    clearTimeout(groupChatSyncTimer)
  }

  // Queue on the ACTIVE gateway synchronously (tests and older hosts have no
  // async route inventory), then widen to every reachable gateway before the
  // debounce fires.
  const activeId = String(groupChatSyncConnectionId() || '')

  const queueFor = (connectionId: string) => {
    const id = String(connectionId || '')
    const retryTimer = groupChatSyncRetryTimers.get(id)

    if (retryTimer !== undefined) {
      clearTimeout(retryTimer)
      groupChatSyncRetryTimers.delete(id)
    }

    groupChatSyncPendingByConnection.set(
      id,
      mergeGroupChatSyncJobs(groupChatSyncPendingByConnection.get(id), {
        connectionId: id,
        allowEmpty,
        changedRooms,
        deletedRooms
      })
    )
  }

  queueFor(activeId)
  groupChatSyncTimer = setTimeout(() => {
    groupChatSyncTimer = null
    void groupChatSyncTargetConnections()
      .then(targets => {
        for (const target of targets) {
          if (String(target || '') !== activeId) {
            queueFor(target)
          }
        }
      })
      .catch(() => undefined)
      .then(() => flushGroupChatServerSync())
  }, 350)
}

export function handleSessionsGatewayTransition() {
  // A gateway swap invalidates any in-flight room drive: bump every room's
  // epoch so running loops bail at their next member boundary.
  const rooms = {
    ...$groupChats.get()
  }

  for (const name of Object.keys(rooms)) {
    rooms[name] = {
      ...rooms[name],
      epoch: (rooms[name].epoch || 0) + 1,
      running: false
    }
  }

  $groupChats.set(rooms)
  // Pull before re-publishing so a reconnect or source swap never lets this
  // client's stale cache hide a room written by another Desktop/mobile client.
  // The returned promise is also the recovery barrier: callers may only
  // re-drive unread room turns after the reconnect merge has settled.

  return pullGroupChatServerState()
    .catch(() => false)
    .then(() => {
      scheduleGroupChatServerSync($groupChats.get())
    })
}

/** Re-arm the mirror after a dispose. `register()` owns this door: an
 *  imported binding cannot be assigned, so the flag's reset crosses the
 *  module edge as an accessor (same pattern as shared.ts's plugin context). */
export function setGroupChatSyncDisposed(disposed: boolean) {
  groupChatSyncDisposed = disposed
}

// ── one room's budget ────────────────────────────────────────────────────────
// Every ceiling a single user send can spend, in one block on purpose: making
// them configurable (per room, or model-aware from config.yaml) is live
// contributor work — #92213 (per-room limits) and #96842 (config + token
// budget) — and both need exactly one seam to hook. Carried over at the same
// values the old plugin.js shipped so neither rebase inherits a behavior
// change on top of a rewrite; deciding the shape of the override belongs to
// those PRs, not to a design-system pass.
export const GROUP_CHAT_MAX_ROUNDS = 3

// #94478 review: continuation rounds are bounded independently of the message cap so a pathological mention chain can't consume the room's whole budget on handoffs.
export const GROUP_CHAT_MAX_MESSAGES = 10
export const GROUP_CHAT_MAX_CONTINUATIONS = 2
export const GROUP_CHAT_HISTORY_LIMIT = 24
export const GROUP_CHAT_MAX_MEMBERS = 6

/** Transcript form of a room speaker's profile name. Friendly identity wins:
 *  a Bot Mode title or a core profile display_name (e.g. default renamed to
 *  "Lucy") labels the speaker everywhere this helper feeds — the "X is
 *  thinking…" working line, the activity feed, and transcript lines — so a
 *  renamed bot never shows up as its raw profile id or a stale "Hermes"
 *  (community report, Aug 21 2026: renamed default still read "Hermes is
 *  thinking…" in group rooms). The untitled primary profile is literally
 *  named "default" — render it as Hermes (matching displayName and the
 *  @hermes handle) so the main agent never loses its name in rooms. */
export function groupSpeakerLabel(name?: null | string) {
  const trimmed = (name || '').trim()

  if (!trimmed) {
    return trimmed
  }

  // Bot Mode title (edit dialog) — same first rung as displayName().
  const title = String($botMeta.get()?.[trimmed]?.title || '').trim()

  if (title) {
    return title
  }

  // Core profile display_name (`hermes profile rename …` / dashboard) from
  // the ACTIVE gateway's roster row. Source-scoped remote speakers carry
  // their device suffix separately and keep their raw name here.
  const roster = $lastRoster.get()

  const row = Array.isArray(roster)
    ? roster.find(bot => bot?.name === trimmed && !bot?.remoteSource && !bot?.sourceScoped)
    : null

  const renamed = typeof row?.display_name === 'string' ? row.display_name.trim() : ''

  if (renamed) {
    return renamed
  }

  return trimmed.toLowerCase() === 'default' ? 'Hermes' : trimmed
}

/** Trim a room log + its watermarks to the retained window, keeping
 *  watermark indices consistent with the trimmed array. */
export function trimGroupChatLog(
  log: GroupMessage[],
  watermarks: Record<string, number>,
  limit = GROUP_CHAT_HISTORY_LIMIT * 4
) {
  if (log.length <= limit) {
    return {
      log,
      watermarks
    }
  }

  const drop = log.length - limit
  const trimmed: Record<string, number> = {}

  for (const [name, index] of Object.entries(watermarks || {})) {
    trimmed[name] = Math.max(0, index - drop)
  }

  return {
    log: log.slice(drop),
    watermarks: trimmed
  }
}

interface UpdateGroupChatOptions {
  sync?: boolean
}

/** Mutate one group's room state through the atom + persist the durable part. */
export function updateGroupChat(
  group: string,
  mutate: (room: GroupChat) => GroupChat,
  { sync = true }: UpdateGroupChatOptions = {}
) {
  const all = {
    ...$groupChats.get()
  }

  const current = all[group] || {
    log: [],
    watermarks: {},
    epoch: 0,
    running: false
  }

  const next = mutate({
    ...current,
    log: [...current.log],
    watermarks: {
      ...current.watermarks
    }
  })

  const bounded = trimGroupChatLog(next.log, next.watermarks)
  next.log = bounded.log
  next.watermarks = bounded.watermarks
  all[group] = next
  $groupChats.set(all)

  try {
    const durable: Record<string, GroupChat> = {}

    for (const [name, room] of Object.entries(all)) {
      // Disband tombstones are runtime-only coordination state (they hold the
      // epoch bump for an in-flight drive). Persisting one would resurrect
      // the room as an empty record on the next load AND keep its name
      // "taken" for same-name recreates.
      if (room.tombstone) {
        continue
      }

      durable[name] = {
        log: room.log,
        watermarks: room.watermarks,
        sessions: room.sessions || {},
        sessionOwners: room.sessionOwners || {},
        // Timed-out turns awaiting a late reply — keyed by member, valued
        // with the pre-turn message baseline. Survives reloads so finished
        // work is still harvested after a window restart.
        stranded: room.stranded || {},
        ...(room.pendingThread ? { pendingThread: room.pendingThread } : {}),
        // #93129: sticky per-member stop holds. Watermarks persist, so holds
        // must too — otherwise a window restart silently releases a bot the
        // user explicitly stopped.
        holds: room.holds || {},
        // Source-qualified member descriptors keep the room whole when the
        // active connection changes and today's local members become remote.
        members: Array.isArray(room.members) ? room.members : [],
        // Immutable room identity: the member-session title for new rooms.
        roomId: typeof room.roomId === 'string' && room.roomId ? room.roomId : null,
        // Room picture (small data URL, same normalization as bot avatars).
        image: room.image || null,
        syncRevision: Math.max(0, Number(room.syncRevision || 0))
      }
    }

    Promise.resolve(getPluginCtx()?.storage?.set?.('group-chats', durable)).catch(() => undefined)
  } catch {
    /* storage unavailable — room survives for this window only */
  }

  if (sync) {
    scheduleGroupChatServerSync(all, {
      changedRooms: [group]
    })
  }

  return next
}

/** A #93129 member hold as this file mints it. `GroupHold` models only the
 *  two fields that survive a reload; the live stamp also records WHICH user
 *  message, in which thread, put the member on hold. */
export interface GroupHoldStamp extends GroupHold {
  byMessageId?: null | string
  thread?: null | string
}

/** The room record as the coordination engine handles it: `GroupChat` plus
 *  `turn`, the runtime-only name of the member currently mid-turn. Like
 *  `running`/`epoch` it never persists, so it has no place in the durable
 *  shape. Holds carry the fuller live stamp. */
export interface GroupChatRoom extends GroupChat {
  holds?: Record<string, GroupHoldStamp>
  turn?: null | string
}

/** Set or clear a group chat's room picture (small data URL, normalized by
 *  the same pipeline as bot avatars). Persists with the room record. */
export function setGroupChatImage(group: string, image: null | string | undefined) {
  updateGroupChat(group, (room: GroupChatRoom) => {
    room.image = image || null

    return room
  })
}

function groupChatEntryId(): string {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function') {
    return globalThis.crypto.randomUUID()
  }

  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}

/** The agent loop's "(empty)" terminal sentinel (empty_response_exhausted) is
 *  a FAILURE marker, never bot text. Mirror gateway/run.py's user-friendly
 *  substitution so the room log never shows the raw sentinel. */
const GROUP_EMPTY_SENTINEL = '(empty)'

const GROUP_EMPTY_FRIENDLY =
  '⚠️ The model returned no response after processing tool results. ' +
  'This can happen with some models — try again or rephrase your question.'

function normalizeGroupChatText(text: string): string {
  const trimmed = String(text || '').trim()

  return trimmed === GROUP_EMPTY_SENTINEL ? GROUP_EMPTY_FRIENDLY : trimmed
}

export function appendGroupChatEntry(
  group: string,
  from: GroupMessageAuthor,
  text: string,
  thread?: null | string,
  images?: Attachment[]
): GroupMessage {
  const entry: GroupMessage = {
    id: groupChatEntryId(),
    at: Date.now(),
    from,
    text: normalizeGroupChatText(text),
    thread: thread || 'legacy'
  }

  if (Array.isArray(images) && images.length) {
    // [{ name, data }] — data URLs. Persisted with the room log so reloads
    // keep showing what the members were shown.
    entry.images = images
  }

  // #93127 insurance: a residual double-append path (stale loop + fresh
  // loop both committing the same member reply) lands back-to-back and
  // byte-identical. Drop the echo instead of flooding the room. User
  // entries and non-adjacent repeats are never touched.
  const priorLog = ($groupChats.get()[group] || {}).log || []
  const lastEntry = priorLog[priorLog.length - 1]

  if (isDuplicateGroupAppend(lastEntry, from, entry.text, entry.thread)) {
    return lastEntry
  }

  updateGroupChat(group, (room: GroupChatRoom) => {
    room.log.push(entry)

    return room
  })

  // Needs-you: a member addressing @user badges the group header.
  if (from.kind === 'member' && /@user\b/i.test(entry.text)) {
    $groupNeedsYou.set({
      ...$groupNeedsYou.get(),
      [group]: true
    })
  }

  return entry
}

/** Fresh room identity for a group. Independent of the editable display
 *  name: a disbanded-and-recreated group mints a new roomId even when the
 *  display name is identical, so member sessions never resume by title. */
export function mintGroupRoomId(): string {
  return `r${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
}

/** Unique display name for a NEW group. Collisions get a " 2", " 3", …
 *  suffix; the BASE is truncated (never the joined string), so a 64-char
 *  base keeps its suffix instead of colliding with the original. */
export function uniqueGroupChatName(base: string, taken: Set<string>): string {
  if (!taken.has(base)) {
    return base
  }

  for (let n = 2; n < 100; n++) {
    const suffix = ` ${n}`
    const candidate = base.slice(0, 64 - suffix.length) + suffix

    if (!taken.has(candidate)) {
      return candidate
    }
  }

  throw new Error('No free name for the group.')
}

// --- room-turn decision helpers (#93127) — pure, unit-tested ---

/** #93127: whether a finished member turn may still commit (append its reply
 *  and advance its watermark). A turn dispatched under an older epoch was
 *  superseded mid-flight by a newer user send — its late result must be
 *  dropped, because the new send's own loop re-drives this member with the
 *  full delta and committing both is exactly the double-delivery bug.
 *
 *  The re-drive premise is only true for a send in the SAME thread (delta
 *  filters are thread-scoped): a cross-thread epoch bump must NOT discard
 *  finished work no fresh loop will regenerate. Callers pass whether a newer
 *  USER entry landed in this thread since dispatch; the default (true)
 *  preserves the conservative drop when the caller can't tell. */
export function shouldCommitMemberTurn(epochAtDispatch: number, currentEpoch: number, newerUserEntryInThread = true) {
  if (epochAtDispatch === currentEpoch) {
    return true
  }

  return !newerUserEntryInThread
}

/** #93127 insurance: byte-identical member echo detection. TRUE only when
 *  the immediately-preceding log entry has the same author (kind + name +
 *  source), same thread, and identical text, within a short recency window —
 *  a residual double-append fires back-to-back; two legitimately identical
 *  replies hours apart (or with anything in between) are never dropped. */
const GROUP_DUPLICATE_APPEND_WINDOW_MS = 10 * 60 * 1000

function isDuplicateGroupAppend(
  lastEntry: GroupMessage | undefined,
  from: GroupMessageAuthor,
  text: string,
  thread: null | string | undefined,
  now = Date.now()
): boolean {
  if (!lastEntry || !from || from.kind !== 'member' || lastEntry.from?.kind !== 'member') {
    return false
  }

  if (String(lastEntry.from?.name || '') !== String(from.name || '')) {
    return false
  }

  if (String(lastEntry.from?.source || '') !== String(from.source || '')) {
    return false
  }

  if (String(lastEntry.thread || 'legacy') !== String(thread || 'legacy')) {
    return false
  }

  if (now - (lastEntry.at || 0) > GROUP_DUPLICATE_APPEND_WINDOW_MS) {
    return false
  }

  return String(lastEntry.text || '') === String(text || '').trim()
}

// --- end room-turn decision helpers ---

export function groupThreadOf(entry: GroupMessage): string {
  return entry?.thread || 'legacy'
}

export function mintGroupThreadId(): string {
  return `t${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
}

// Pre-thread logs (hydrated from storage) get synthetic thread ids: a user
// entry after a real lull starts one, so multi-turn tasks stay whole instead
// of splitting on every follow-up.
const GROUP_THREAD_GAP_MS = 15 * 60000

export function assignLegacyThreads(log: GroupMessage[]): GroupMessage[] {
  let current: null | string = null
  let n = 0

  return (log || []).map((entry, i) => {
    if (entry?.thread) {
      current = null

      return entry
    }

    const prev = log[i - 1]
    const lull = !prev || (entry.at || 0) - (prev.at || 0) > GROUP_THREAD_GAP_MS

    if (!current || (entry.from?.kind === 'user' && lull)) {
      current = `legacy-${n++}`
    }

    return {
      ...entry,
      thread: current
    }
  })
}
