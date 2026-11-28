import { stickyGroupEngine } from './group-engine'
import type { DurableGroupChat, GroupChat, GroupMessage } from './types'

/** Characters of attachment payload one room may persist.
 *
 *  The room map shares a ~5M-char origin quota with every other plugin key, and
 *  the text budget already claims up to 256k per room. Attachments used to ride
 *  the log unbounded — one 15 MB PDF is ~20M data-URL characters, four times the
 *  whole quota — so a single paste made every later room write fail silently.
 *  A room's attachments therefore get one turn-window's worth of characters:
 *  enough for a small image, never enough to turn a reload into a lost store. */
export const GROUP_CHAT_ATTACHMENT_RETAIN_CHARS = 32_000

/** The durable copy of a room's log: attachment payloads bounded to the room's
 *  attachment budget, newest first.
 *
 *  A `pdf`/`file` chip renders from its name alone, so its payload is never
 *  stored. An image keeps its data URL while the budget allows and falls back to
 *  the same chip when it does not — the message and its attachment name survive
 *  either way, which is the difference between a degraded preview and a room
 *  store that stops accepting writes. */
function durableGroupChatLog(log: GroupMessage[]): GroupMessage[] {
  let remaining = GROUP_CHAT_ATTACHMENT_RETAIN_CHARS
  const durable = log.slice()

  for (let i = durable.length - 1; i >= 0; i--) {
    const entry = durable[i]

    if (!Array.isArray(entry.images) || !entry.images.length) {
      continue
    }

    const images = entry.images.map(image => {
      if (image.kind !== 'image' || !image.data || image.data.length > remaining) {
        return { ...image, data: '' }
      }

      remaining -= image.data.length

      return image
    })

    durable[i] = { ...entry, images }
  }

  return durable
}

export function durableGroupChatRoom(room: GroupChat): DurableGroupChat {
  return {
    log: durableGroupChatLog(room.log),
    holdDetection: room.holdDetection !== false,
    heldMessages: room.heldMessages || {},
    watermarks: room.watermarks || {},
    sessions: room.sessions || {},
    sessionOwners: room.sessionOwners || {},
    stranded: room.stranded || {},
    holds: room.holds || {},
    externalCursors: room.externalCursors || {},
    members: Array.isArray(room.members) ? room.members : [],
    roomId: typeof room.roomId === 'string' && room.roomId ? room.roomId : null,
    engine: stickyGroupEngine(room).engine,
    image: room.image || null,
    rosterOrder: room.rosterOrder,
    pinned: room.pinned,
    sectionId: room.sectionId ?? null,
    syncRevision: Math.max(0, Number(room.syncRevision || 0))
  }
}

const objectOrEmpty = <T>(value: T) => (value && typeof value === 'object' ? value : {}) as NonNullable<T>

export function hydrateDurableGroupChatRoom(room: DurableGroupChat): DurableGroupChat {
  return {
    log: room.log,
    holdDetection: room.holdDetection !== false,
    heldMessages: objectOrEmpty(room.heldMessages),
    watermarks: objectOrEmpty(room.watermarks),
    sessions: objectOrEmpty(room.sessions),
    sessionOwners: objectOrEmpty(room.sessionOwners),
    stranded: objectOrEmpty(room.stranded),
    holds: objectOrEmpty(room.holds),
    externalCursors: objectOrEmpty(room.externalCursors),
    members: Array.isArray(room.members) ? room.members : [],
    roomId: typeof room.roomId === 'string' && room.roomId ? room.roomId : null,
    engine: stickyGroupEngine(room).engine,
    image: typeof room.image === 'string' && room.image ? room.image : null,
    rosterOrder: Number.isFinite(room.rosterOrder) ? room.rosterOrder : undefined,
    pinned: Boolean(room.pinned),
    sectionId: room.sectionId ?? null,
    syncRevision: Math.max(0, Number(room.syncRevision || 0))
  }
}
