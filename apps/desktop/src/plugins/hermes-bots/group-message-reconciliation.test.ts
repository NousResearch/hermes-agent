import { describe, expect, it, vi } from 'vitest'

import * as chat from './group-chat'
import { GROUP_CHAT_SYNC_TEXT_CHARS } from './group-message-author'
import { speakerEvent, speakerMember, speakerReplay, speakerRoom } from './group-speaker-test-fixtures'
import type { GroupChat, GroupMessage } from './types'

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock({})
})

function compact(log: GroupMessage[]) {
  return chat.groupChatSyncSnapshot({ Board: speakerRoom(log) }).rooms['id:room-1'].log
}

function coldRoom(room: GroupChat) {
  const saved = JSON.parse(JSON.stringify(chat.durableGroupChatRooms({ Board: room })))

  return chat.mergeRemoteGroupChatSnapshotIntoRooms(chat.groupChatSyncSnapshot(saved), {}).Board
}

describe('speaker reconciliation across real room mirrors', () => {
  it.each([false, true])('preserves every old stable anchor through mirror-first hydration, rename=%s', rename => {
    const log = speakerReplay([
      speakerEvent(1, 'ux', 'default', 'Team'),
      speakerEvent(2, 'reviewer', 'default', 'Team'),
      speakerEvent(3, 'reviewer', 'default', 'Team')
    ])

    const current = { Board: speakerRoom(log) }
    const mirror = chat.groupChatSyncSnapshot(current)

    for (const message of mirror.rooms['id:room-1'].log) {
      delete message.from.hostedIdentity

      if (rename) {
        message.from.name = 'Former Team'
      }
    }

    const restored = chat.mergeRemoteGroupChatSnapshotIntoRooms(mirror, current).Board

    expect(
      restored.log.map(entry => [entry.id, entry.eventId, entry.seq, entry.from.hostedIdentity?.memberId])
    ).toEqual([
      ['event-1', 'event-1', 1, 'ux'],
      ['event-2', 'event-2', 2, 'reviewer'],
      ['event-3', 'event-3', 3, 'reviewer']
    ])
  })

  for (const sequenced of [false, true]) {
    it.each(['ux', 'reviewer'])(
      'keeps conflict evidence through repeated %s copies and cold hydration, sequenced=' + sequenced,
      repeat => {
        const a = speakerReplay([speakerEvent(1, 'ux', 'default', 'Team')])
        const b = speakerReplay([speakerEvent(1, 'reviewer', 'default', 'Team')])
        let room = speakerRoom(chat.mergeGroupChatSyncEntries(...(sequenced ? [a, b] : [compact(a), compact(b)])))
        const repeated = repeat === 'ux' ? a : b

        for (let iteration = 0; iteration < 3; iteration++) {
          room = coldRoom(room)
          const snapshot = chat.groupChatSyncSnapshot({ Board: speakerRoom(compact(repeated)) })
          room = chat.mergeRemoteGroupChatSnapshotIntoRooms(snapshot, { Board: room }).Board
          expect(room.log).toHaveLength(1)
          expect(room.log[0].from.hostedIdentity).toBeUndefined()

          if (sequenced) {
            room.log = chat.mergeGroupChatSyncEntries(room.log, repeated)
            expect(room.log[0].from.hostedIdentity).toBeUndefined()
          }
        }
      }
    )
  }

  it.each(['ux', 'reviewer'])('allows stronger coherent replay for %s to recover a mirror conflict', memberId => {
    const a = speakerReplay([speakerEvent(1, 'ux', 'default', 'Team')])
    const b = speakerReplay([speakerEvent(1, 'reviewer', 'default', 'Team')])
    const unresolved = coldRoom(speakerRoom(chat.mergeGroupChatSyncEntries(compact(a), compact(b))))
    const canonical = memberId === 'ux' ? a : b
    const recovered = chat.mergeGroupChatSyncEntries(unresolved.log, canonical, compact(a), compact(b))

    expect(recovered).toHaveLength(1)
    expect(recovered[0].from.hostedIdentity).toEqual(canonical[0].from.hostedIdentity)
    expect(recovered[0].seq).toBe(1)
  })

  it('does not promote a weak actor by borrowing a sequence from an identity-less copy', () => {
    const a = speakerReplay([speakerEvent(1, 'ux', 'default', 'Team')])
    const b = speakerReplay([speakerEvent(1, 'reviewer', 'default', 'Team')])
    const identityless = { ...a[0], from: { kind: 'member' as const, name: 'Team' } }
    const weak = chat.mergeGroupChatSyncEntries(compact(a), [identityless])
    const mirrored = coldRoom(speakerRoom(weak))
    const recovered = chat.mergeGroupChatSyncEntries(mirrored.log, b)

    expect(weak[0].seq).toBe(1)
    expect(recovered).toHaveLength(1)
    expect(recovered[0].from.hostedIdentity).toEqual(b[0].from.hostedIdentity)
  })

  it('cannot choose a donor when a sequence-only copy has two possible stable IDs', () => {
    const [a] = speakerReplay()
    const b = { ...a, id: 'other-event', eventId: 'other-event' }

    const legacy = { ...a, id: undefined, eventId: undefined, from: { kind: 'member' as const, name: 'Legacy author' } }

    const merged = chat.mergeGroupChatSyncEntries([a], [b], [legacy])

    expect(merged).toHaveLength(3)
    expect(merged.find(entry => entry.from.name === 'Legacy author')?.from.hostedIdentity).toBeUndefined()
    expect(chat.mergeGroupChatSyncEntries(merged, [a], [b])).toHaveLength(3)
  })

  for (const reverse of [false, true]) {
    it.each([-1, 0, 1])(
      'does not extend a complete replay from a mirror at the text boundary, offset=%s, reverse=' + reverse,
      offset => {
        const [canonical] = speakerReplay()
        canonical.text = 'A'.repeat(GROUP_CHAT_SYNC_TEXT_CHARS + offset)
        const room = speakerRoom([canonical])
        const snapshot = chat.groupChatSyncSnapshot({ Board: room })
        const mirror = snapshot.rooms['id:room-1'].log[0]
        delete mirror.from.hostedIdentity
        mirror.from.name = 'Other author'
        mirror.text += ' NOT_FROM_THE_EVENT'

        const pair = reverse ? [[mirror], [canonical]] : [[canonical], [mirror]]
        const merged = chat.mergeGroupChatSyncEntries(...pair)
        const hydrated = chat.mergeRemoteGroupChatSnapshotIntoRooms(snapshot, { Board: room }).Board
        const saved = JSON.parse(JSON.stringify(chat.durableGroupChatRooms({ Board: hydrated }))).Board
        const replayed = chat.mergeGroupChatSyncEntries(saved.log, [canonical])

        for (const log of [merged, hydrated.log, replayed]) {
          const bound = log.filter((entry: GroupMessage) => entry.from.hostedIdentity?.memberId === 'pm')
          expect(bound).toHaveLength(1)
          expect(bound[0].text).toBe(canonical.text)
          expect(bound[0].seq).toBe(canonical.seq)
        }
      }
    )

    it.each(['event-id', 'sequence', 'internal-id', 'text', 'thread', 'attachments'])(
      'does not mix %s evidence or payload with a canonical actor, reverse=' + reverse,
      mismatch => {
        const [canonical] = speakerReplay()
        canonical.images = [{ attachmentId: 'original', kind: 'image', name: 'one.png' }]

        const other: GroupMessage = {
          ...canonical,
          from: { kind: 'member', name: 'Other author' },
          text: 'Unrelated body',
          ...(mismatch === 'event-id' ? { id: 'other-event', eventId: 'other-event' } : {}),
          ...(mismatch === 'sequence' ? { seq: 2 } : {}),
          ...(mismatch === 'internal-id' ? { id: 'other-event' } : {}),
          ...(mismatch === 'thread' ? { text: canonical.text, thread: 'unrelated-thread' } : {}),
          ...(mismatch === 'attachments'
            ? { text: canonical.text, images: [{ attachmentId: 'other', kind: 'image', name: 'two.png' }] }
            : {})
        }

        const pair = reverse ? [[other], [canonical]] : [[canonical], [other]]
        const merged = chat.mergeGroupChatSyncEntries(...pair, ...pair)
        const bound = merged.filter(entry => entry.from.hostedIdentity)

        expect(bound).toHaveLength(1)
        expect(bound[0]).toEqual(canonical)
        expect(
          merged.filter(entry => !entry.from.hostedIdentity).every(entry => entry.from.name === 'Other author')
        ).toBe(true)
        expect(merged.length).toBeGreaterThanOrEqual(2)
      }
    )

    it('keeps a valid bounded text projection attached to the full payload, reverse=' + reverse, () => {
      const [canonical] = speakerReplay()
      canonical.text = 'A'.repeat(1400)
      const mirror = compact([canonical])
      delete mirror[0].from.hostedIdentity
      expect(mirror[0].text).toHaveLength(1200)
      const merged = chat.mergeGroupChatSyncEntries(...(reverse ? [mirror, [canonical]] : [[canonical], mirror]))

      expect(merged).toHaveLength(1)
      expect(merged[0].text).toBe(canonical.text)
      expect(merged[0].from.hostedIdentity).toEqual(canonical.from.hostedIdentity)
    })

    it('retains classic idless and acknowledged optimistic echo behavior, reverse=' + reverse, () => {
      const legacy: GroupMessage = { at: 1, text: 'Classic', from: { kind: 'member', name: 'default' } }
      expect(chat.mergeGroupChatSyncEntries([legacy], [{ ...legacy, thread: 'legacy-4' }])).toHaveLength(1)
      const optimistic: GroupMessage = { at: 1, id: 'accepted-id', from: { kind: 'user', name: 'You' }, text: 'Hello' }
      const accepted = { ...optimistic, at: 2, eventId: 'accepted-id', seq: 4 }

      const merged = chat.mergeGroupChatSyncEntries(
        ...(reverse ? [[accepted], [optimistic]] : [[optimistic], [accepted]])
      )

      expect(merged).toHaveLength(1)
      expect(merged[0]).toMatchObject({ id: 'accepted-id', eventId: 'accepted-id', seq: 4, text: 'Hello' })
    })
  }

  it('does not lend a known actor to an unrelated body via the actual room merger', () => {
    const [canonical] = speakerReplay()
    const current = { Board: speakerRoom([canonical], [speakerMember()]) }
    const remote = chat.groupChatSyncSnapshot(current)
    const copy = remote.rooms['id:room-1'].log[0]
    copy.text = 'Unrelated body'
    delete copy.from.hostedIdentity
    const room = chat.mergeRemoteGroupChatSnapshotIntoRooms(remote, current).Board

    expect(room.log.find(entry => entry.text === 'Unrelated body')?.from.hostedIdentity).toBeUndefined()
    expect(room.log.find(entry => entry.text === canonical.text)?.from.hostedIdentity).toEqual(
      canonical.from.hostedIdentity
    )
  })
})
