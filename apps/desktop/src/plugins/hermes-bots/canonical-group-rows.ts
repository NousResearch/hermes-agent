/**
 * Canonical (hosted) rooms → roster rows.
 *
 * A hosted room lives in the gateway, not in this Desktop's `$groupChats`, so
 * the left roster — which lists the rooms this Desktop recorded itself — never
 * showed one. These helpers turn `groups.list` rooms into the rows the roster
 * already renders: a room the Desktop records keeps its row (pin, section,
 * order, log preview) and gains the `canonical:` key that opens the authority;
 * a room it has never seen becomes a plain row.
 */

import type { CanonicalRoom, CanonicalRoomMember } from './canonical-groups'
import type { GroupMember } from './types'

export interface CanonicalGroupRoom {
  key: string
  room: CanonicalRoom
}

export interface CanonicalRosterRow {
  name: string
  members: GroupMember[]
  /** The `canonical:` key that opens the authority; absent for a Desktop-only room. */
  canonicalKey?: string
}

/** One room's members in the roster's vocabulary. `handle` is the room's FROZEN
 *  handle: it is the only id `resolve_mentions` accepts at the gateway, so the
 *  composer inserts exactly it — never a friendly name. */
export function canonicalRoomMembers(room: CanonicalRoom): GroupMember[] {
  const seen = new Set<string>()

  return (room.members || []).flatMap((member: CanonicalRoomMember) => {
    const name = String(member?.profile || '').trim()
    const key = name.toLowerCase()

    // A room can only seat a profile once; the roster keys on the profile.
    if (!key || seen.has(key)) {
      return []
    }

    seen.add(key)

    return [
      {
        name,
        display_name: member.display_name || undefined,
        handle: String(member.handle || '').trim() || undefined,
        targetProfile: name
      }
    ]
  })
}

/** The mention token the gateway resolves for a canonical member: its frozen
 *  handle. A friendly slug ("Atlas Prime" → `atlas-prime`) resolves to nobody. */
export function canonicalMentionHandle(member: GroupMember): string | undefined {
  const handle = String(member?.handle || '')
    .trim()
    .toLowerCase()

  return handle || undefined
}

/** Merge the gateway's rooms into the roster's own rows, by name. Rooms the
 *  roster has no record of are appended; rooms it does keep their members so a
 *  pinned, sectioned, previewed row looks exactly as it did. */
export function mergeCanonicalGroupRows(
  rows: CanonicalRosterRow[],
  rooms: CanonicalGroupRoom[]
): CanonicalRosterRow[] {
  const pending = new Map<string, CanonicalGroupRoom>()

  for (const room of rooms) {
    const name = String(room?.room?.name || '').trim()

    if (!name || room.room.disbanded_at) {
      continue
    }

    // Same-name forks: `groups.list` is most-recently-updated last.
    pending.set(name, room)
  }

  const merged = rows.map(row => {
    const room = pending.get(row.name)

    if (!room) {
      return row
    }

    pending.delete(row.name)

    return {
      ...row,
      canonicalKey: room.key,
      members: row.members.length ? row.members : canonicalRoomMembers(room.room)
    }
  })

  for (const room of pending.values()) {
    merged.push({
      canonicalKey: room.key,
      members: canonicalRoomMembers(room.room),
      name: String(room.room.name).trim()
    })
  }

  return merged
}
