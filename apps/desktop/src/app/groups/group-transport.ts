import type { GroupRoom } from './group-model'

export type GroupRequester = (method: string, params?: Record<string, unknown>) => Promise<unknown>

export interface CreateGroupRoomInput { name: string; profiles: string[]; workspace?: string }
export interface GroupRoomPageInput { beforeSeq?: number; cursor?: string }
export interface GroupRoomListResponse { rooms?: GroupRoom[] }
export interface GroupRoomResponse { room?: GroupRoom; cursor?: string | null; has_more?: boolean }

export function createGroupTransport(request: GroupRequester) {
  return {
    listRooms: () => request('group.room.list') as Promise<GroupRoomListResponse>,
    createRoom: (input: CreateGroupRoomInput) => request('group.room.create', { ...input }) as Promise<GroupRoomResponse>,
    getRoom: (roomId: string, page?: GroupRoomPageInput) => request('group.room.get', {
      room_id: roomId,
      ...(page?.beforeSeq !== undefined ? { before_seq: page.beforeSeq } : {}),
      ...(page?.cursor ? { cursor: page.cursor } : {})
    }) as Promise<GroupRoomResponse>,
    deleteRoom: (roomId: string) => request('group.room.delete', { room_id: roomId }),
    sendMessage: (roomId: string, content: string, mentions: string[]) =>
      request('group.message.send', { room_id: roomId, content, mentions }),
    interrupt: (roomId: string, profile?: string) =>
      request('group.run.interrupt', { room_id: roomId, ...(profile ? { profile } : {}) }),
    subscribe: (roomId: string) => request('group.subscribe', { room_id: roomId }),
    unsubscribe: (roomId: string) => request('group.unsubscribe', { room_id: roomId }),
    respondToApproval: (runtimeSessionId: string, choice: 'once' | 'session' | 'always' | 'deny') =>
      request('approval.respond', { choice, session_id: runtimeSessionId })
  }
}

export function mentionsFromText(content: string, profiles: readonly string[]): string[] {
  const found = new Set<string>()

  for (const match of content.matchAll(/(^|\s)@([\w.-]+)/g)) {
    const mention = match[2]

    if (mention === 'all' || profiles.includes(mention)) {found.add(mention)}
  }

  return [...found]
}
