import { createHostedRoomReplayState, reduceHostedRoomEvents } from './hosted-room-client'
import type { GroupChat, GroupMember, GroupMessage } from './types'

export function speakerMember(memberId = 'pm', profile = 't2oracle', display = 'Product'): GroupMember {
  return {
    name: profile,
    targetProfile: profile,
    display_name: display,
    handle: memberId,
    connectionId: `connection-${memberId}`,
    connectionLabel: `Device ${memberId}`,
    remoteSource: true,
    sourceScoped: true,
    hostedIdentity: { roomId: 'room-1', memberId, profile, installationId: `install:${memberId}` }
  }
}

export function speakerEvent(seq = 1, memberId = 'pm', profile = 't2oracle', display = 'Product') {
  return {
    room_id: 'room-1',
    seq,
    event_id: `event-${seq}`,
    kind: 'message.member',
    actor: { kind: 'member', id: memberId, profile, display_name: display, connection_id: 'old-device-label' },
    payload: { text: 'Decision ready.', thread_id: 'thread-1' },
    created_at: 1000
  }
}

export function speakerReplay(events = [speakerEvent()]): GroupMessage[] {
  return reduceHostedRoomEvents(createHostedRoomReplayState({ roomId: 'room-1' }), events).messages.map(message => ({
    ...message,
    id: message.eventId
  }))
}

export function speakerRoom(log = speakerReplay(), members = [speakerMember()]): GroupChat {
  return { roomId: 'room-1', hosted: 'install:home', hostedEpoch: 1, log, members, watermarks: {} }
}
