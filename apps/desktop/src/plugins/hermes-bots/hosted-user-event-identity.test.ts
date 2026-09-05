import { createHash } from 'node:crypto'

import { describe, expect, it, vi } from 'vitest'

import type { GroupChat, GroupMessage } from './types'
import {
  canonicalUser,
  CLIENT_ID,
  EVENT_ID,
  optimisticUser,
  USER_TEXT,
  userEvent,
  userRoom
} from './user-event-test-fixtures'

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock({ state: {} })
})

function hashed(value: string) {
  return `user:${createHash('sha256').update(value.trim()).digest('hex')}`
}

async function merge(current: GroupChat, incoming: GroupMessage[]) {
  const chat = await import('./group-chat')
  const snapshot = chat.groupChatSyncSnapshot({ Board: { ...current, log: incoming } })

  return chat.mergeRemoteGroupChatSnapshotIntoRooms(snapshot, { Board: current }).Board
}

describe('hosted client-key/canonical user event identity', () => {
  it('matches the physical UAT client key using an independent standard hash', () => {
    expect(hashed(CLIENT_ID)).toBe(EVENT_ID)
  })

  it.each([false, true])('heals the existing two-entry JSON cache, reversed=%s', async reverse => {
    const chat = await import('./group-chat')
    const canonical = canonicalUser()
    const pair = reverse ? [canonical, optimisticUser()] : [optimisticUser(), canonical]
    const cold = JSON.parse(JSON.stringify(chat.durableGroupChatRooms({ Board: userRoom(pair) })))
    const projection = chat.groupChatSyncSnapshot(cold)
    const restored = chat.mergeRemoteGroupChatSnapshotIntoRooms(projection, cold).Board

    expect(restored.log).toHaveLength(1)
    expect(restored.log[0]).toMatchObject({ id: EVENT_ID, eventId: EVENT_ID, seq: 6, text: USER_TEXT, thread: 'work' })
    expect(projection.rooms['id:room-1'].log).toHaveLength(1)
  })

  it('heals an old mirror that arrives after canonical replay', async () => {
    const restored = await merge(userRoom([canonicalUser()]), [optimisticUser()])

    expect(restored.log).toHaveLength(1)
    expect(restored.log[0].id).toBe(EVENT_ID)
  })

  it('does not invent authority from two unsequenced mirrors alone', async () => {
    const projected = { ...canonicalUser(), seq: undefined }
    const restored = await merge(userRoom([projected]), [optimisticUser()])

    expect(restored.log).toHaveLength(2)
  })

  it.each(['member', 'other-room', 'wrong-hash'])('refuses %s as user-alias proof', async scenario => {
    const canonical = canonicalUser()

    if (scenario === 'member') {
      canonical.from = { kind: 'member', name: 'Bot' }
    }

    if (scenario === 'other-room') {
      canonical.roomId = 'different-room'
    }

    if (scenario === 'wrong-hash') {
      canonical.id = canonical.eventId = hashed('another-key')
    }

    const restored = await merge(userRoom([canonical]), [optimisticUser()])

    expect(restored.log).toHaveLength(2)
  })

  it('preserves genuine old-server raw-ID and newer hashed-ID canonical events', async () => {
    const chat = await import('./group-chat')
    const old = canonicalUser(CLIENT_ID, 4, { text: 'Old backend event' })
    const current = userRoom([old, canonicalUser(EVENT_ID, 6)])
    const mirror = chat.groupChatSyncSnapshot({ Board: current })
    const restored = chat.mergeRemoteGroupChatSnapshotIntoRooms(mirror, { Board: current }).Board

    expect(restored.log.map(entry => [entry.id, entry.seq])).toEqual([
      [CLIENT_ID, 4],
      [EVENT_ID, 6]
    ])
    expect(restored.log[0].text).toBe('Old backend event')
  })

  it('preserves a raw-ID canonical marker across a compact/cold round trip', async () => {
    const chat = await import('./group-chat')
    const old = canonicalUser(CLIENT_ID, 4)
    const coldMirror = JSON.parse(JSON.stringify(chat.groupChatSyncSnapshot({ Board: userRoom([old]) })))

    const restored = chat.mergeRemoteGroupChatSnapshotIntoRooms(coldMirror, {
      Board: userRoom([canonicalUser()])
    }).Board

    expect(restored.log).toHaveLength(2)
    expect(restored.log.find(entry => entry.id === CLIENT_ID)?.eventId).toBe(CLIENT_ID)
  })

  it('never double-hashes canonical-looking IDs or aliases two sequenced events', async () => {
    const first = canonicalUser(EVENT_ID, 6)
    const second = canonicalUser(hashed(EVENT_ID), 7)
    const current = userRoom([first, second])
    const restored = await merge(current, [optimisticUser({ id: EVENT_ID })])

    expect(restored.log.map(entry => [entry.id, entry.seq])).toEqual([
      [EVENT_ID, 6],
      [hashed(EVENT_ID), 7]
    ])
    const unknown = await merge(userRoom([second]), [optimisticUser({ id: EVENT_ID })])
    expect(unknown.log).toHaveLength(2)
  })

  it.each(['body', 'thread', 'attachments'])('canonical data never inherits a mirror %s extension', async changed => {
    const canonical = canonicalUser(EVENT_ID, 6, {
      text: 'C'.repeat(1200),
      images: [{ attachmentId: 'accepted-file', kind: 'file', name: 'accepted.txt' }]
    })

    const mirror = { ...canonical, seq: undefined }

    if (changed === 'body') {
      mirror.text += ' FORGED'
    }

    if (changed === 'thread') {
      mirror.thread = 'wrong-thread'
    }

    if (changed === 'attachments') {
      mirror.images = [{ attachmentId: 'wrong-file', kind: 'file', name: 'wrong.txt' }]
    }

    const restored = await merge(userRoom([canonical]), [mirror])

    expect(restored.log).toHaveLength(1)
    expect(restored.log[0]).toEqual(canonical)
  })

  it('keeps the global strict matcher and classic distinct-ID behavior unchanged', async () => {
    const chat = await import('./group-chat')
    expect(chat.mergeGroupChatSyncEntries([canonicalUser()], [optimisticUser()])).toHaveLength(2)
    const classic = { ...userRoom([canonicalUser()]), hosted: null }
    expect((await merge(classic, [optimisticUser()])).log).toHaveLength(2)
  })

  it.each(['body', 'thread', 'attachments'])(
    'never grafts a reused raw-key mirror %s onto canonical data',
    async changed => {
      const canonical = canonicalUser(EVENT_ID, 6, { text: 'accepted' })
      const mirror = optimisticUser()

      if (changed === 'body') {
        mirror.text = 'invented extension'
      }

      if (changed === 'thread') {
        mirror.thread = 'different-thread'
      }

      if (changed === 'attachments') {
        mirror.images = [{ kind: 'file', name: 'forged.txt', attachmentId: 'forged' }]
      }

      expect((await merge(userRoom([canonical]), [mirror])).log).toEqual([canonical])
    }
  )

  it('does not infer user authority from a sequence supplied by a display projection', async () => {
    const chat = await import('./group-chat')

    const projection = {
      version: 3,
      rooms: { 'id:room-1': { roomId: 'room-1', hosted: 'install:home', log: [canonicalUser()], name: 'Board' } }
    }

    const restored = chat.mergeRemoteGroupChatSnapshotIntoRooms(projection, {
      Board: userRoom([optimisticUser()])
    }).Board

    expect(restored.log).toHaveLength(2)
    expect(restored.log.every(entry => entry.seq === undefined)).toBe(true)
  })

  it('does not use a room-name fallback as proof that different room IDs agree', async () => {
    const chat = await import('./group-chat')
    const other = chat.groupChatSyncSnapshot({ Board: { ...userRoom([optimisticUser()]), roomId: 'other-room' } })

    expect(
      chat.mergeRemoteGroupChatSnapshotIntoRooms(other, { Board: userRoom([canonicalUser()]) }).Board.log
    ).toHaveLength(2)
  })

  it('retains ambiguous sequenced records without selecting a donor for their mirror', async () => {
    const chat = await import('./group-chat')

    const entries = chat.mergeGroupChatRoomEntries(
      userRoom([]),
      [canonicalUser(EVENT_ID, 6), canonicalUser(EVENT_ID, 7)],
      [optimisticUser()]
    )

    expect(entries.map(entry => entry.seq)).toEqual([6, 7, undefined])
  })

  it('does not equate idless hosted text with an accepted user event', async () => {
    expect((await merge(userRoom([canonicalUser()]), [optimisticUser({ id: undefined })])).log).toHaveLength(2)
    const chat = await import('./group-chat')
    const classic = optimisticUser()
    expect(chat.mergeGroupChatSyncEntries([classic], [{ ...classic, id: undefined }])).toHaveLength(1)
  })

  it('keeps a compact canonical marker when two projections meet without a local receipt', async () => {
    const chat = await import('./group-chat')
    const old = chat.groupChatSyncSnapshot({ Board: userRoom([canonicalUser(CLIENT_ID, 4)]) })
    const modern = chat.groupChatSyncSnapshot({ Board: userRoom([canonicalUser()]) })
    const merged = JSON.parse(JSON.stringify(chat.mergeGroupChatSyncSnapshots(old, modern)))

    expect(merged.rooms['id:room-1'].log.map((entry: GroupMessage) => entry.eventId)).toEqual([CLIENT_ID, EVENT_ID])
  })

  it('keeps one rich pending user entry when its same-ID compact mirror arrives before acceptance', async () => {
    const message = optimisticUser({ roomId: 'room-1', clientEventId: CLIENT_ID, text: 'pending '.repeat(200) })
    const restored = await merge(userRoom([message]), [message])

    expect(restored.log).toEqual([message])
  })

  it('cannot bind an explicit foreign-room mirror to an older canonical record lacking room metadata', async () => {
    const canonical = canonicalUser(EVENT_ID, 6, { roomId: undefined })
    const foreign = { ...canonical, roomId: 'foreign-room', seq: undefined }
    const restored = await merge(userRoom([canonical]), [foreign])

    expect(restored.log).toHaveLength(2)
    expect(restored.log.find(entry => entry.seq === 6)?.roomId).toBe('room-1')
    expect(restored.log.find(entry => entry.roomId === 'foreign-room')?.seq).toBeUndefined()
  })

  it('retains a canonical marker for older seq/id-only raw UUID records when compacting', async () => {
    const chat = await import('./group-chat')
    const old = canonicalUser(CLIENT_ID, 4, { eventId: undefined, roomId: undefined })
    const cold = JSON.parse(JSON.stringify(chat.groupChatSyncSnapshot({ Board: userRoom([old]) })))
    const restored = chat.mergeRemoteGroupChatSnapshotIntoRooms(cold, { Board: userRoom([canonicalUser()]) }).Board

    expect(restored.log).toHaveLength(2)
    expect(restored.log.find(entry => entry.id === CLIENT_ID)?.eventId).toBe(CLIENT_ID)
  })
})

describe('hosted outgoing proof boundaries', () => {
  it.each(['projection', 'snapshot-union', 'cold-json'])('does not acquire client intent through %s', async via => {
    const chat = await import('./group-chat')
    const unknown = optimisticUser({ id: EVENT_ID, roomId: 'room-1' })
    const forged = { ...unknown, clientEventId: EVENT_ID }
    let room = userRoom([unknown])

    let projection = {
      version: 3,
      rooms: { 'id:room-1': { roomId: 'room-1', hosted: 'install:home', name: 'Board', log: [forged] } }
    }

    if (via === 'snapshot-union') {
      projection = chat.mergeGroupChatSyncSnapshots(
        projection,
        chat.groupChatSyncSnapshot({ Board: room })
      ) as typeof projection
    }

    if (via === 'cold-json') {
      room = userRoom(JSON.parse(JSON.stringify([forged])))
    } else {
      room = chat.mergeRemoteGroupChatSnapshotIntoRooms(projection, { Board: room }).Board
    }

    const cold = JSON.parse(JSON.stringify(chat.durableGroupChatRooms({ Board: room }))).Board
    const result = chat.mergeGroupChatRoomEntries(cold, cold.log, [canonicalUser(hashed(EVENT_ID), 6)])

    expect(result.map(entry => entry.id).sort()).toEqual([EVENT_ID, hashed(EVENT_ID)].sort())
  })

  it.each(['projection', 'snapshot-union', 'cold-json'])(
    'does not turn a real receipt into reusable outgoing proof through %s',
    async via => {
      const chat = await import('./group-chat')
      const { hostedUserEventReceipt } = await import('./hosted-user-events')

      const receipt = hostedUserEventReceipt(
        {
          commandId: EVENT_ID,
          roomId: 'room-1',
          kind: 'send',
          authorityId: 'install:home',
          connectionId: 'gateway-a',
          attempts: 1,
          status: 'in-flight',
          failureCode: null,
          payload: {}
        },
        { client_event_id: EVENT_ID, event: userEvent(hashed(EVENT_ID)) }
      )!

      const unknown = optimisticUser({ id: EVENT_ID, roomId: 'room-1' })
      let room = userRoom([unknown])

      let projection = {
        version: 3,
        rooms: { 'id:room-1': { roomId: 'room-1', hosted: 'install:home', name: 'Board', log: [receipt] } }
      }

      if (via === 'snapshot-union') {
        projection = chat.mergeGroupChatSyncSnapshots(
          projection,
          chat.groupChatSyncSnapshot({ Board: room })
        ) as typeof projection
      }

      if (via === 'cold-json') {
        room = userRoom([unknown, JSON.parse(JSON.stringify(receipt))])
      } else {
        room = chat.mergeRemoteGroupChatSnapshotIntoRooms(projection, { Board: room }).Board
      }

      const result = chat.mergeGroupChatRoomEntries(room, room.log, [canonicalUser(hashed(EVENT_ID))])
      expect(result.map(entry => entry.id).sort()).toEqual([EVENT_ID, hashed(EVENT_ID)].sort())
    }
  )

  it('keeps genuine live append intent through a local same-ID merge but not through JSON', async () => {
    const chat = await import('./group-chat')
    vi.useFakeTimers()

    try {
      chat.$groupChats.set({ Board: userRoom([]) })
      chat.appendGroupChatEntry('Board', { kind: 'user', name: 'You' }, USER_TEXT, 'work', undefined, EVENT_ID)
      const room = chat.$groupChats.get().Board
      const mirror = chat.groupChatSyncSnapshot({ Board: room })
      const live = chat.mergeRemoteGroupChatSnapshotIntoRooms(mirror, { Board: room }).Board
      const cold = JSON.parse(JSON.stringify(chat.durableGroupChatRooms({ Board: live }))).Board

      expect(chat.mergeGroupChatRoomEntries(live, live.log, [canonicalUser(hashed(EVENT_ID))])).toHaveLength(1)
      expect(chat.mergeGroupChatRoomEntries(cold, cold.log, [canonicalUser(hashed(EVENT_ID))])).toHaveLength(2)
      expect(mirror.rooms['id:room-1'].log[0]).not.toHaveProperty('clientEventId')
    } finally {
      chat.stopGroupChatServerSync()
      vi.useRealTimers()
    }
  })

  it.each(['body', 'thread', 'attachment', 'authority', 'room', 'failed', 'canonical-marker'])(
    'does not restore outgoing proof from an outbox %s mismatch',
    async mismatch => {
      const { restoreHostedUserOutboxIntents } = await import('./hosted-user-events')
      const { createHostedRoomOutbox } = await import('./hosted-room-client')
      const chat = await import('./group-chat')
      const pending = optimisticUser({ id: EVENT_ID, roomId: 'room-1' })

      const command = {
        commandId: EVENT_ID,
        kind: 'send',
        roomId: 'room-1',
        authorityId: 'install:home',
        connectionId: 'gateway-a',
        status: 'pending',
        payload: { text: pending.text, thread_id: pending.thread }
      }

      if (mismatch === 'body') {
        command.payload.text = 'another body'
      }

      if (mismatch === 'thread') {
        command.payload.thread_id = 'another-thread'
      }

      if (mismatch === 'attachment') {
        pending.images = [{ kind: 'file', name: 'unrelated', attachmentId: 'unrelated' }]
      }

      if (mismatch === 'authority') {
        command.authorityId = 'install:other'
      }

      if (mismatch === 'room') {
        command.roomId = 'room-2'
      }

      if (mismatch === 'failed') {
        command.status = 'failed'
      }

      if (mismatch === 'canonical-marker') {
        pending.eventId = pending.id
      }

      const room = userRoom([pending])
      const restored = restoreHostedUserOutboxIntents(room, createHostedRoomOutbox({ commands: [command] }))

      expect(chat.mergeGroupChatRoomEntries(room, restored, [canonicalUser(hashed(EVENT_ID))])).toHaveLength(2)
    }
  )
})

describe('hosted room-container namespaces', () => {
  it.each([
    [false, false],
    [false, true],
    [true, false],
    [true, true]
  ])('preserves local canonical scope with local/foreign entry scopes %s/%s', async (localScoped, foreignScoped) => {
    const chat = await import('./group-chat')
    const canonical = canonicalUser(EVENT_ID, 6, { roomId: localScoped ? 'room-1' : undefined })
    const foreign = { ...canonical, roomId: foreignScoped ? 'foreign-room' : undefined, seq: undefined }

    const projection = {
      version: 3,
      rooms: {
        'id:foreign-room': {
          name: 'Board',
          roomId: 'foreign-room',
          hosted: 'install:other',
          revision: 100,
          log: [foreign]
        }
      }
    }

    const room = chat.mergeRemoteGroupChatSnapshotIntoRooms(projection, { Board: userRoom([canonical]) }).Board

    expect(room.roomId).toBe('room-1')
    expect(room.log.filter(entry => entry.seq === 6)).toHaveLength(1)
    expect(room.log.find(entry => entry.seq === 6)?.roomId).toBe('room-1')
    expect(room.log.find(entry => !entry.seq)?.roomId).toBe('foreign-room')
    expect(
      chat.mergeGroupChatRoomEntries(room, room.log, [optimisticUser()]).some(entry => entry.id === CLIENT_ID)
    ).toBe(false)
  })

  it('cannot hide the known foreign container by supplying a local-looking per-entry scope', async () => {
    const chat = await import('./group-chat')
    const canonical = canonicalUser()
    const foreign = { ...canonical, seq: undefined, roomId: 'room-1' }

    const projection = {
      version: 3,
      rooms: {
        'id:foreign-room': {
          name: 'Board',
          roomId: 'foreign-room',
          hosted: 'install:other',
          log: [foreign]
        }
      }
    }

    const room = chat.mergeRemoteGroupChatSnapshotIntoRooms(projection, { Board: userRoom([canonical]) }).Board

    expect(room.log).toHaveLength(2)
    expect(room.log.find(entry => entry.seq === 6)?.roomId).toBe('room-1')
    expect(room.log.find(entry => !entry.seq)?.roomId).toBe('foreign-room')
  })

  it('preserves same-room legacy recovery, a real rename, and unrelated room discovery', async () => {
    const chat = await import('./group-chat')
    const canonical = canonicalUser(EVENT_ID, 6, { roomId: undefined })
    const current = userRoom([canonical])

    const projection = chat.groupChatSyncSnapshot({
      Renamed: { ...userRoom([optimisticUser()]), syncRevision: 10 },
      New: { ...userRoom([optimisticUser({ id: 'other-client' })]), roomId: 'room-new' }
    })

    const result = chat.mergeRemoteGroupChatSnapshotIntoRooms(projection, { Board: current })

    expect(result.Board).toBeUndefined()
    expect(result.Renamed.roomId).toBe('room-1')
    expect(result.Renamed.log).toHaveLength(1)
    expect(result.Renamed.log[0]).toMatchObject({ id: EVENT_ID, seq: 6, roomId: 'room-1' })
    expect(result.New.roomId).toBe('room-new')
    expect(result.New.log[0].id).toBe('other-client')
  })
})
