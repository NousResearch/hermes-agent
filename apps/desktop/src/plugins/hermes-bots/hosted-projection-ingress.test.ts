import { describe, expect, it, vi } from 'vitest'

import type { GroupChat, GroupMessage } from './types'
import { canonicalUser, CLIENT_ID, EVENT_ID, optimisticUser, userRoom } from './user-event-test-fixtures'

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock({ state: {} })
})

function legacyPair(): GroupMessage[] {
  return [
    canonicalUser(EVENT_ID, 6, {
      roomId: undefined,
      from: { kind: 'member', name: 'Legacy bot' },
      text: 'Shared display text'
    }),
    optimisticUser({ id: EVENT_ID, eventId: EVENT_ID, text: 'Shared display text' })
  ]
}

function projection(log: GroupMessage[], roomId = 'room-1') {
  return { version: 3, rooms: { [`id:${roomId}`]: { name: 'Board', roomId, hosted: 'install:home', log } } }
}

describe('display projection ingress never establishes replay authority', () => {
  for (const mode of ['new', 'existing', 'union']) {
    for (const cold of [false, true]) {
      for (const reverse of [false, true]) {
        it(`keeps actor kinds and pending text for ${mode}, cold=${cold}, reverse=${reverse}`, async () => {
          const chat = await import('./group-chat')
          const pair = legacyPair()

          if (reverse) {
            pair.reverse()
          }

          let input = projection(pair)

          if (mode === 'union') {
            input = chat.mergeGroupChatSyncSnapshots(input, { version: 3, rooms: {} }) as typeof input
          }

          let room = chat.mergeRemoteGroupChatSnapshotIntoRooms(
            input,
            mode === 'existing' ? { Board: userRoom([]) } : {}
          ).Board

          if (cold) {
            room = JSON.parse(JSON.stringify(chat.durableGroupChatRooms({ Board: room }))).Board
          }

          expect(room.log.map(entry => entry.from.kind).sort()).toEqual(['member', 'user'])
          expect(room.log.every(entry => entry.seq === undefined)).toBe(true)
          expect(room.log.every(entry => entry.roomId === 'room-1')).toBe(true)
          const pending = optimisticUser({ text: 'Actual unsent local command' })
          expect(
            chat
              .mergeGroupChatRoomEntries(room, room.log, [pending])
              .some(entry => entry.id === CLIENT_ID && entry.text === pending.text)
          ).toBe(true)
        })
      }
    }
  }

  it.each(['user', 'member'] as const)('never imports %s sequence into a new room or projection union', async kind => {
    const chat = await import('./group-chat')
    const entry = canonicalUser(EVENT_ID, 6, { from: { kind, name: 'Legacy author' }, roomId: undefined })
    const input = projection([entry])
    const room = chat.mergeRemoteGroupChatSnapshotIntoRooms(input, {}).Board
    const union = chat.mergeGroupChatSyncSnapshots(input, { version: 3, rooms: {} })

    expect(room.log[0].seq).toBeUndefined()
    expect(union.rooms['id:room-1'].log[0].seq).toBeUndefined()
    expect(room.log[0].from.kind).toBe(kind)
  })

  it.each([1, 3])('keeps legacy actor kinds separate without inventing room scope, projection v%s', async version => {
    const chat = await import('./group-chat')
    const key = version === 1 ? 'Board' : 'name:Board'
    const input = { version, rooms: { [key]: { name: 'Board', log: legacyPair() } } }
    const room = chat.mergeRemoteGroupChatSnapshotIntoRooms(input, {}).Board

    expect(room.log.map(entry => entry.from.kind).sort()).toEqual(['member', 'user'])
    expect(room.log.every(entry => entry.seq === undefined && !entry.roomId)).toBe(true)
  })

  it.each(['user', 'member'] as const)(
    'preserves a genuine local %s sequence without lending it to the other kind',
    async kind => {
      const chat = await import('./group-chat')
      const canonical = canonicalUser(EVENT_ID, 6, { roomId: undefined, from: { kind, name: 'Actual local actor' } })
      const otherKind = kind === 'member' ? 'user' : 'member'
      const imported = { ...canonical, seq: 999, from: { kind: otherKind, name: 'Imported actor' } } as GroupMessage

      const room = chat.mergeRemoteGroupChatSnapshotIntoRooms(projection([imported]), {
        Board: userRoom([canonical])
      }).Board

      expect(room.log).toHaveLength(2)
      expect(room.log.find(entry => entry.from.kind === kind)).toMatchObject({
        seq: 6,
        roomId: 'room-1',
        text: canonical.text
      })
      expect(room.log.find(entry => entry.from.kind === otherKind)?.seq).toBeUndefined()
    }
  )

  it('keeps real stored member history intact beside a legacy user hint, including after later room merges', async () => {
    const chat = await import('./group-chat')
    const room = userRoom(legacyPair())
    const restored = chat.mergeRemoteGroupChatSnapshotIntoRooms(projection([]), { Board: room }).Board

    const later = chat.mergeGroupChatRoomEntries(restored, restored.log, [
      optimisticUser({ text: 'Unsent local text' })
    ])

    expect(later.find(entry => entry.from.kind === 'member')).toMatchObject({ id: EVENT_ID, seq: 6 })
    expect(later.find(entry => entry.id === EVENT_ID && entry.from.kind === 'user')?.seq).toBeUndefined()
    expect(later.some(entry => entry.id === CLIENT_ID)).toBe(true)
  })

  it.each([null, 'install:home'])(
    'does not lend a stored member sequence while room identity is still unknown, hosted=%s',
    async hosted => {
      const chat = await import('./group-chat')
      const entries = chat.mergeGroupChatRoomEntries({ hosted, roomId: null }, legacyPair())

      expect(entries).toHaveLength(2)
      expect(entries.find(entry => entry.from.kind === 'member')?.seq).toBe(6)
      expect(entries.find(entry => entry.from.kind === 'user')?.seq).toBeUndefined()
    }
  )

  it('does not heuristically delete or demote already stored sequenced contradictions', async () => {
    const chat = await import('./group-chat')
    const first = canonicalUser(EVENT_ID, 6, { text: 'First retained canonical body' })
    const second = canonicalUser(EVENT_ID, 6, { text: 'Second retained canonical body' })
    const room = chat.mergeRemoteGroupChatSnapshotIntoRooms(projection([]), { Board: userRoom([first, second]) }).Board

    expect(room.log.map(entry => [entry.seq, entry.text])).toEqual([
      [6, first.text],
      [6, second.text]
    ])
  })

  it('binds foreign member hints to their container without borrowing a local member sequence', async () => {
    const chat = await import('./group-chat')
    const local = canonicalUser(EVENT_ID, 6, { roomId: undefined, from: { kind: 'member', name: 'Member' } })
    const foreign = { ...local, seq: 99, roomId: 'room-1' }

    const room = chat.mergeRemoteGroupChatSnapshotIntoRooms(projection([foreign], 'room-2'), {
      Board: userRoom([local])
    }).Board

    expect(room.log).toHaveLength(2)
    expect(room.log.find(entry => entry.seq === 6)?.roomId).toBe('room-1')
    expect(room.log.find(entry => entry.roomId === 'room-2')?.seq).toBeUndefined()
  })

  for (const kind of ['user', 'member'] as const) {
    for (const sameId of [false, true]) {
      it(`preserves ${kind} room scope through snapshot export and reimport, sameId=${sameId}`, async () => {
        const chat = await import('./group-chat')

        const local = canonicalUser(EVENT_ID, 6, {
          roomId: undefined,
          from: { kind, name: 'Legacy author' },
          text: 'Shared display text'
        })

        const foreign = {
          ...local,
          id: sameId ? EVENT_ID : 'foreign-event',
          eventId: sameId ? EVENT_ID : 'foreign-event',
          seq: 99
        }

        const rooms = chat.mergeRemoteGroupChatSnapshotIntoRooms(projection([foreign], 'room-2'), {
          Board: userRoom([local])
        })

        expect(rooms.Board.log).toHaveLength(2)
        expect(rooms.Board.log.find(entry => entry.seq === 6)?.roomId).toBe('room-1')

        const exported = JSON.parse(JSON.stringify(chat.groupChatSyncSnapshot(rooms)))
        const exportedLog = exported.rooms['id:room-1'].log as GroupMessage[]

        expect(exportedLog.map(entry => entry.roomId).sort()).toEqual(['room-1', 'room-2'])
        expect(exportedLog.every(entry => entry.seq === undefined && entry.clientEventId === undefined)).toBe(true)

        const restored = chat.mergeRemoteGroupChatSnapshotIntoRooms(exported, {}).Board

        expect(restored.log).toHaveLength(2)
        expect(restored.log.map(entry => entry.roomId).sort()).toEqual(['room-1', 'room-2'])
        expect(restored.log.every(entry => entry.from.kind === kind && entry.seq === undefined)).toBe(true)
      })
    }
  }

  it('preserves valid same-kind legacy merging, room discovery and a genuine same-room rename', async () => {
    const chat = await import('./group-chat')
    const plain = optimisticUser()

    const classic = chat.mergeRemoteGroupChatSnapshotIntoRooms(
      { version: 1, rooms: { Classic: { log: [plain] } } },
      { Classic: { log: [plain], watermarks: {} } as GroupChat }
    ).Classic

    expect(classic.log).toHaveLength(1)

    const input = projection([canonicalUser(EVENT_ID, 6)])
    input.rooms['id:room-1'] = { ...input.rooms['id:room-1'], name: 'Renamed' }
    const current = { ...userRoom([canonicalUser()]), syncRevision: 0 }

    const renamed = chat.mergeRemoteGroupChatSnapshotIntoRooms(
      {
        ...input,
        rooms: {
          'id:room-1': { ...input.rooms['id:room-1'], revision: 10 }
        }
      },
      { Board: current }
    )

    expect(renamed.Board).toBeUndefined()
    expect(renamed.Renamed.log).toHaveLength(1)
    expect(renamed.Renamed.log[0].seq).toBe(6)
    expect(chat.mergeRemoteGroupChatSnapshotIntoRooms(input, {}).Renamed.roomId).toBe('room-1')
  })
})
