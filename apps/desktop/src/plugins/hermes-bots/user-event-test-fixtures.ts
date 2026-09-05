import { createHostedRoomReplayState, reduceHostedRoomEvents } from './hosted-room-client'
import type { GroupChat, GroupMessage } from './types'

export const CLIENT_ID = '55f2f6a5-c2d2-4a8e-a8da-d767b41b41f0'
export const EVENT_ID = 'user:ce70d3e8f5bde75b9d14f2a890a26869cc5ab650163938590e177ded380720e1'
export const USER_TEXT = 'x'.repeat(1144)

export function userEvent(id = EVENT_ID, seq = 6, payload = { text: USER_TEXT, thread_id: 'work' }) {
  return {
    room_id: 'room-1',
    event_id: id,
    seq,
    kind: 'message.user',
    actor: { kind: 'user', id: 'desktop' },
    payload,
    created_at: 1000
  }
}

export function canonicalUser(id = EVENT_ID, seq = 6, overrides: Partial<GroupMessage> = {}): GroupMessage {
  const [message] = reduceHostedRoomEvents(createHostedRoomReplayState({ roomId: 'room-1', cursor: seq - 1 }), [
    userEvent(id, seq)
  ]).messages

  return { ...message, id: message.eventId, ...overrides }
}

export function optimisticUser(overrides: Partial<GroupMessage> = {}): GroupMessage {
  return {
    id: CLIENT_ID,
    from: { kind: 'user', name: 'You' },
    text: USER_TEXT,
    thread: 'work',
    at: 1000001,
    ...overrides
  }
}

export function userRoom(log: GroupMessage[]): GroupChat {
  return {
    log,
    roomId: 'room-1',
    hosted: 'install:home',
    hostedConnectionId: 'gateway-a',
    hostedEpoch: 1,
    hostedSeq: 5,
    watermarks: {},
    members: [
      { name: 'research', connectionId: 'gateway-a', sourceScoped: true, targetProfile: 'research' },
      { name: 'builder', connectionId: 'gateway-a', sourceScoped: true, targetProfile: 'builder' }
    ]
  }
}
