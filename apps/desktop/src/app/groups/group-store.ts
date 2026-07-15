import { atom } from 'nanostores'

import { applyGroupEvent, emptyGroupState, type GroupEvent, type GroupRoom, type GroupState, mergeGroupRoom } from './group-model'

export const $groupState = atom<GroupState>(emptyGroupState())
let roomsRequestGeneration = 0

export function beginGroupRoomsRequest(): number {
  roomsRequestGeneration += 1

  return roomsRequestGeneration
}

export function cacheGroupRoom(room: GroupRoom): void {
  roomsRequestGeneration += 1
  $groupState.set(mergeGroupRoom($groupState.get(), room))
}

export function cacheGroupRooms(rooms: GroupRoom[]): void {
  let state = $groupState.get()

  for (const room of rooms) {state = mergeGroupRoom(state, room)}
  $groupState.set(state)
}

export function reconcileGroupRooms(generation: number, rooms: GroupRoom[]): void {
  if (generation !== roomsRequestGeneration) {return}
  let state = emptyGroupState()

  for (const room of rooms) {state = mergeGroupRoom(state, room)}
  $groupState.set(state)
}

export function cacheSentGroupMessage(value: unknown): void {
  const result = value && typeof value === 'object' ? value as Record<string, unknown> : {}
  const event = result.event && typeof result.event === 'object' ? result.event as Record<string, unknown> : {}
  const payload = event.payload && typeof event.payload === 'object' ? event.payload as Record<string, unknown> : {}

  handleGroupEvent({
    type: 'group.message.complete',
    payload: {
      room_id: event.room_id,
      message_id: `group-${String(event.seq ?? '')}`,
      role: 'user',
      text: payload.text,
      created_at: event.created_at
    }
  })
}

export function clearCachedGroupApproval(roomId: string, messageId: string): void {
  const state = $groupState.get()
  const room = state.rooms[roomId]

  if (!room) {return}

  const messages = room.messages.map(message => message.id === messageId
    ? { ...message, approval: undefined, status: 'streaming' as const }
    : message)

  $groupState.set({ rooms: { ...state.rooms, [roomId]: { ...room, messages } } })
}

export function removeCachedGroupRoom(roomId: string): void {
  roomsRequestGeneration += 1
  const rooms = { ...$groupState.get().rooms }
  delete rooms[roomId]
  $groupState.set({ rooms })
}

export function handleGroupEvent(event: GroupEvent): void {
  const next = applyGroupEvent($groupState.get(), event)

  if (next !== $groupState.get()) {
    roomsRequestGeneration += 1
    $groupState.set(next)
  }
}

export function resetGroupCache(): void {
  $groupState.set(emptyGroupState())
}
