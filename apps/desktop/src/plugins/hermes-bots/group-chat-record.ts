import { stickyGroupEngine } from './group-engine'
import type { DurableGroupChat, GroupChat } from './types'

export function durableGroupChatRoom(room: GroupChat): DurableGroupChat {
  return {
    log: room.log,
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
