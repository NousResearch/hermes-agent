import { createHash } from 'node:crypto'

import { afterEach, describe, expect, it, vi } from 'vitest'

import { pluginSdkMock, scriptedStorage } from './group-test-utils'
import type { GroupMessage } from './types'
import { canonicalUser, CLIENT_ID, EVENT_ID, optimisticUser, userEvent, userRoom } from './user-event-test-fixtures'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))
vi.mock('@hermes/plugin-sdk', async () => pluginSdkMock(host))

async function runtimeFixture(
  send: (params: Record<string, unknown>) => unknown,
  log: GroupMessage[] = [optimisticUser()]
) {
  vi.resetModules()
  vi.useFakeTimers()
  const storage = new Map<string, unknown>()
  const ctx = scriptedStorage(storage)
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []

  const room = {
    room_id: 'room-1',
    name: 'Board',
    authority_gateway_id: 'install:home',
    authority_epoch: 1,
    latest_seq: 6,
    members: [
      { member_id: 'research', profile: 'research' },
      { member_id: 'builder', profile: 'builder' }
    ]
  }

  let events: unknown[] = []

  const request = async (method: string, params: Record<string, unknown>) => {
    calls.push({ method, params })

    if (method === 'groups.capabilities') {
      return { driver: true, persistent_process: true, authority_gateway_id: 'install:home' }
    }

    if (method === 'groups.list') {
      return { rooms: [room] }
    }

    if (method === 'groups.state') {
      return { room, driver_status: { working: false } }
    }

    if (method === 'groups.log') {
      return {
        events,
        latest_seq: Math.max(6, ...events.map(event => Number((event as { seq: number }).seq))),
        has_more: false
      }
    }

    if (method === 'groups.send') {
      return send(params)
    }

    throw new Error(`Unexpected RPC: ${method}`)
  }

  Object.assign(host, {
    activeConnectionId: () => 'gateway-a',
    notify: vi.fn(),
    profileRoutes: async () => [
      { connectionId: 'gateway-a', mode: 'remote', profile: 'default', targetProfile: 'default' }
    ],
    request,
    requestProfile: (_route: unknown, method: string, params: Record<string, unknown>) => request(method, params),
    state: Object.fromEntries(
      [
        ['connectionId', 'gateway-a'],
        ['gateway', 'open'],
        ['profile', 'default']
      ].map(([key, value]) => [key, { get: () => value, listen: () => () => undefined }])
    )
  })
  const chat = await import('./group-chat')
  const runtime = await import('./hosted-room-runtime')
  const { setPluginCtx } = await import('./shared')
  setPluginCtx(ctx)
  chat.$groupChats.set({ Board: userRoom(log) })
  await runtime.startHostedRoomRuntime(ctx.storage)

  return {
    chat,
    runtime,
    storage,
    calls,
    ctx,
    setEvents: (next: unknown[]) => {
      events = next
      room.latest_seq = Math.max(6, ...events.map(event => Number((event as { seq: number }).seq)))
    }
  }
}

afterEach(async () => {
  const runtime = await import('./hosted-room-runtime')
  runtime.stopHostedRoomRuntime()
  const { setPluginCtx } = await import('./shared')
  setPluginCtx(null)
  vi.clearAllTimers()
  vi.useRealTimers()
})

describe('real hosted send response and replay identity', () => {
  it('uses client_event_id and canonical event immediately, without advancing the replay cursor', async () => {
    const loaded = await runtimeFixture(() => ({ client_event_id: CLIENT_ID, event: userEvent() }))
    const message = loaded.chat.$groupChats.get().Board.log[0]
    expect(await loaded.runtime.sendHostedGroupChat('Board', message, 'work')).toBe(true)
    const room = loaded.chat.$groupChats.get().Board

    expect(loaded.calls.find(call => call.method === 'groups.send')?.params.event_id).toBe(CLIENT_ID)
    expect(room.log).toHaveLength(1)
    expect(room.log[0]).toMatchObject({ id: EVENT_ID, eventId: EVENT_ID, seq: 6 })
    expect(room.hostedSeq).toBe(5)
    loaded.setEvents([userEvent()])
    await loaded.runtime.refreshHostedRooms()
    expect(loaded.chat.$groupChats.get().Board.log).toHaveLength(1)
  })

  it('heals a lost response after restart, then ignores late old-cache twins', async () => {
    let lost = true

    const loaded = await runtimeFixture(() => {
      if (lost) {
        throw new Error('Lost response')
      }

      return { client_event_id: CLIENT_ID, event: userEvent() }
    })

    const message = loaded.chat.$groupChats.get().Board.log[0]
    expect(await loaded.runtime.sendHostedGroupChat('Board', message, 'work')).toBe(false)
    const cold = JSON.parse(JSON.stringify(loaded.chat.durableGroupChatRooms()))
    loaded.runtime.stopHostedRoomRuntime()
    loaded.chat.$groupChats.set(cold)
    loaded.setEvents([userEvent()])
    lost = false
    await loaded.runtime.startHostedRoomRuntime(loaded.ctx.storage)
    const canonical = loaded.chat.$groupChats.get().Board
    const late = loaded.chat.groupChatSyncSnapshot({ Board: userRoom([optimisticUser()]) })
    const restored = loaded.chat.mergeRemoteGroupChatSnapshotIntoRooms(late, { Board: canonical }).Board

    expect(restored.log).toHaveLength(1)
    expect(restored.log[0].id).toBe(EVENT_ID)
    expect(loaded.calls.filter(call => call.method === 'groups.send').map(call => call.params.event_id)).toEqual([
      CLIENT_ID,
      CLIENT_ID
    ])
  })

  it('keeps an actual old-server raw UUID event when a later partial replay adds its hashed-name neighbor', async () => {
    const loaded = await runtimeFixture(() => ({}), [canonicalUser(CLIENT_ID, 4), optimisticUser()])
    loaded.setEvents([userEvent(EVENT_ID, 6)])
    await loaded.runtime.refreshHostedRooms()

    expect(loaded.chat.$groupChats.get().Board.log.map(entry => [entry.id, entry.seq])).toEqual([
      [CLIENT_ID, 4],
      [EVENT_ID, 6]
    ])
  })

  it('honors old backend responses with raw canonical event IDs', async () => {
    const loaded = await runtimeFixture(() => ({ event: userEvent(CLIENT_ID) }))
    await loaded.runtime.sendHostedGroupChat('Board', loaded.chat.$groupChats.get().Board.log[0], 'work')

    expect(loaded.chat.$groupChats.get().Board.log).toHaveLength(1)
    expect(loaded.chat.$groupChats.get().Board.log[0]).toMatchObject({ id: CLIENT_ID, eventId: CLIENT_ID, seq: 6 })
  })

  it.each(['room', 'client-key', 'kind', 'event-id', 'sequence'])(
    'does not use a mismatched response %s as proof',
    async mismatch => {
      const reply = { client_event_id: CLIENT_ID, event: userEvent() }

      if (mismatch === 'room') {
        reply.event.room_id = 'other-room'
      }

      if (mismatch === 'client-key') {
        reply.client_event_id = 'other-key'
      }

      if (mismatch === 'kind') {
        reply.event.kind = 'message.member'
      }

      if (mismatch === 'event-id') {
        reply.event.event_id = 'user:' + 'f'.repeat(64)
      }

      if (mismatch === 'sequence') {
        reply.event.seq = 0
      }

      const loaded = await runtimeFixture(() => reply)
      await loaded.runtime.sendHostedGroupChat('Board', loaded.chat.$groupChats.get().Board.log[0], 'work')

      expect(loaded.chat.$groupChats.get().Board.log[0].id).toBe(CLIENT_ID)
      expect(loaded.chat.$groupChats.get().Board.log[0].seq).toBeUndefined()
    }
  )

  it('a reused key cannot graft a new body, thread, or attachments onto the accepted event', async () => {
    const event = {
      ...userEvent(),
      payload: {
        text: 'Accepted first',
        thread_id: 'original',
        attachments: [
          { attachment_id: 'att_' + 'a'.repeat(32), kind: 'file', name: 'accepted.txt', mime: 'text/plain', size: 4 }
        ]
      }
    }

    const loaded = await runtimeFixture(
      () => ({ client_event_id: CLIENT_ID, event }),
      [optimisticUser({ text: 'New body', thread: 'new-thread' })]
    )

    await loaded.runtime.sendHostedGroupChat('Board', loaded.chat.$groupChats.get().Board.log[0], 'new-thread')
    const room = loaded.chat.$groupChats.get().Board

    expect(room.log).toHaveLength(1)
    expect(room.log[0]).toMatchObject({
      id: EVENT_ID,
      seq: 6,
      text: 'Accepted first',
      thread: 'original',
      images: [expect.objectContaining({ name: 'accepted.txt' })]
    })
  })

  it.each([false, true])(
    'proves a deliberately canonical-looking client key only from actual outgoing intent, loss=%s',
    async loss => {
      const nestedId = `user:${createHash('sha256').update(EVENT_ID).digest('hex')}`

      const loaded = await runtimeFixture(() => {
        if (loss) {
          throw new Error('Lost response')
        }

        return { client_event_id: EVENT_ID, event: userEvent(nestedId) }
      }, [optimisticUser({ id: EVENT_ID })])

      await loaded.runtime.sendHostedGroupChat('Board', loaded.chat.$groupChats.get().Board.log[0], 'work')
      const cold = JSON.parse(JSON.stringify(loaded.chat.durableGroupChatRooms()))
      loaded.chat.$groupChats.set(cold)
      loaded.setEvents([userEvent(nestedId)])
      await loaded.runtime.refreshHostedRooms()

      expect(loaded.calls.find(call => call.method === 'groups.send')?.params.event_id).toBe(EVENT_ID)
      expect(loaded.chat.$groupChats.get().Board.log.map(entry => [entry.id, entry.seq])).toEqual([[nestedId, 6]])
    }
  )

  it('does not consume intermediate events when a send receipt is ahead of replay', async () => {
    const loaded = await runtimeFixture(() => ({ client_event_id: CLIENT_ID, event: userEvent(EVENT_ID, 8) }))
    await loaded.runtime.sendHostedGroupChat('Board', loaded.chat.$groupChats.get().Board.log[0], 'work')
    expect(loaded.chat.$groupChats.get().Board.hostedSeq).toBe(5)
    loaded.setEvents([userEvent('prior-event', 6)])
    await loaded.runtime.refreshHostedRooms()
    expect(loaded.chat.$groupChats.get().Board.hostedSeq).toBe(6)
    expect(loaded.chat.$groupChats.get().Board.log.map(entry => entry.seq)).toEqual([6, 8])
    loaded.setEvents([userEvent('another-prior-event', 7), userEvent(EVENT_ID, 8)])
    await loaded.runtime.refreshHostedRooms()

    expect(loaded.chat.$groupChats.get().Board.hostedSeq).toBe(8)
    expect(loaded.chat.$groupChats.get().Board.log.map(entry => entry.seq)).toEqual([6, 7, 8])
  })

  it('normalizes the client key while preserving the unhashed outbound API contract', async () => {
    const raw = `  ${CLIENT_ID}  `

    const loaded = await runtimeFixture(
      () => ({ client_event_id: CLIENT_ID, event: userEvent() }),
      [optimisticUser({ id: raw })]
    )

    await loaded.runtime.sendHostedGroupChat('Board', loaded.chat.$groupChats.get().Board.log[0], 'work')

    expect(loaded.calls.find(call => call.method === 'groups.send')?.params.event_id).toBe(CLIENT_ID)
    expect(loaded.chat.$groupChats.get().Board.log.map(entry => entry.id)).toEqual([EVENT_ID])
  })

  it('keeps one entry when the durable send receipt precedes the actual optimistic append', async () => {
    let accepted: ReturnType<typeof userEvent> | undefined

    const loaded = await runtimeFixture(params => {
      const key = String(params.event_id)
      accepted = userEvent(
        `user:${createHash('sha256').update(key).digest('hex')}`,
        6,
        params.payload as { text: string; thread_id: string }
      )

      return { client_event_id: key, event: accepted }
    }, [])

    const rounds = await import('./group-rounds')
    const room = loaded.chat.$groupChats.get().Board
    expect(await rounds.sendToGroupChatDurably('Board', room.members || [], 'A real durable send', 'work')).toBe('work')

    expect(loaded.chat.$groupChats.get().Board.log.map(entry => entry.id)).toEqual([accepted?.event_id])
    loaded.setEvents([accepted])
    await loaded.runtime.refreshHostedRooms()
    expect(loaded.chat.$groupChats.get().Board.log).toHaveLength(1)
  })

  it.each([false, true])(
    'uses a recovered command receipt without reclassifying a marked canonical mirror, marked=%s',
    async marked => {
      const nestedId = `user:${createHash('sha256').update(EVENT_ID).digest('hex')}`
      const mirror = optimisticUser({ id: EVENT_ID, ...(marked ? { eventId: EVENT_ID } : {}) })
      const loaded = await runtimeFixture(() => ({ client_event_id: EVENT_ID, event: userEvent(nestedId) }), [mirror])
      loaded.runtime.stopHostedRoomRuntime()
      loaded.storage.set('hosted-room-outbox-v1', {
        version: 1,
        commands: [
          {
            commandId: EVENT_ID,
            kind: 'send',
            status: 'in-flight',
            attempts: 1,
            roomId: 'room-1',
            authorityId: 'install:home',
            connectionId: 'gateway-a',
            payload: { text: mirror.text, thread_id: 'work' }
          }
        ]
      })
      await loaded.runtime.startHostedRoomRuntime(loaded.ctx.storage)

      expect(loaded.chat.$groupChats.get().Board.log).toHaveLength(marked ? 2 : 1)
      expect(loaded.chat.$groupChats.get().Board.log.some(entry => entry.id === nestedId && entry.seq === 6)).toBe(true)

      if (marked) {
        expect(loaded.chat.$groupChats.get().Board.log.find(entry => entry.id === EVENT_ID)?.eventId).toBe(EVENT_ID)
      }
    }
  )
})

describe('hosted receipt display payload validation', () => {
  const file = {
    attachment_id: 'att_' + 'a'.repeat(32),
    kind: 'file',
    name: 'accepted.txt',
    mime: 'text/plain',
    size: 4
  }

  it.each([
    null,
    [],
    {},
    { text: 42 },
    { text: null },
    { text: ['not text'] },
    { text: { value: 'not text' } },
    { text: '' },
    { text: '   ' },
    { text: 'valid', thread_id: 42 },
    { text: 'valid', thread: [] },
    { text: 'valid', thread_id: 'first', thread: 'second' },
    { text: 'valid', attachments: null },
    { text: 'valid', attachments: {} },
    { attachments: [null] },
    { attachments: [file, {}] },
    { text: 'valid', attachments: [{ ...file, size: '4' }] },
    { text: 'valid', attachments: [{ ...file, size: null }] },
    { text: 'valid', attachments: [{ ...file, attachment_id: 'not-an-attachment' }] },
    { text: 'valid', attachments: Array.from({ length: 9 }, () => file) }
  ])('rejects malformed payload %j before persistence or subsequent valid replay', async payload => {
    const loaded = await runtimeFixture(() => ({ client_event_id: CLIENT_ID, event: { ...userEvent(), payload } }))
    const pending = loaded.chat.$groupChats.get().Board.log[0]
    await loaded.runtime.sendHostedGroupChat('Board', pending, 'work')
    const cold = JSON.parse(JSON.stringify(loaded.storage.get('group-chats')))

    expect(cold.Board.log.map((entry: GroupMessage) => [entry.id, entry.seq, entry.text])).toEqual([
      [CLIENT_ID, undefined, pending.text]
    ])
    expect(cold.Board.hostedSeq).toBe(5)
    loaded.runtime.stopHostedRoomRuntime()
    loaded.chat.$groupChats.set(cold)
    loaded.setEvents([userEvent()])
    await loaded.runtime.startHostedRoomRuntime(loaded.ctx.storage)
    expect(loaded.chat.$groupChats.get().Board.log.map(entry => [entry.id, entry.seq, entry.text])).toEqual([
      [EVENT_ID, 6, pending.text]
    ])
  })

  it.each([
    { text: 'Previously accepted body' },
    { text: 'Previously accepted body', thread: 'old-thread' },
    { text: 'Previously accepted body', thread_id: 'old-thread', thread: 'old-thread', attachments: [] },
    { attachments: [file] },
    { text: '', thread_id: 'old-thread', attachments: [file] }
  ])('accepts valid older-body or attachment-only payload %j without comparing it to the retry body', async payload => {
    const loaded = await runtimeFixture(() => ({ client_event_id: CLIENT_ID, event: { ...userEvent(), payload } }))
    await loaded.runtime.sendHostedGroupChat('Board', loaded.chat.$groupChats.get().Board.log[0], 'work')
    const room = loaded.chat.$groupChats.get().Board

    expect(room.log).toHaveLength(1)
    expect(room.log[0]).toMatchObject({ id: EVENT_ID, seq: 6, text: 'text' in payload ? payload.text : '' })
    expect(room.hostedSeq).toBe(5)

    if ('attachments' in payload && payload.attachments?.length) {
      expect(room.log[0].images?.[0].name).toBe('accepted.txt')
    }
  })

  it.each(['pending', 'canonical'])(
    'ignores legacy JSON intent claims on a cold %s record without any outgoing command',
    async kind => {
      const nestedId = `user:${createHash('sha256').update(EVENT_ID).digest('hex')}`
      const unknown = optimisticUser({ id: EVENT_ID, roomId: 'room-1' })
      const nested = canonicalUser(nestedId, 6)

      if (kind === 'pending') {
        unknown.clientEventId = EVENT_ID
      } else {
        nested.clientEventId = EVENT_ID
      }

      const loaded = await runtimeFixture(
        () => {
          throw new Error('No send expected')
        },
        JSON.parse(JSON.stringify([unknown, nested]))
      )

      expect(
        loaded.chat.$groupChats
          .get()
          .Board.log.map(entry => entry.id)
          .sort()
      ).toEqual([EVENT_ID, nestedId].sort())
      expect(loaded.calls.filter(call => call.method === 'groups.send')).toHaveLength(0)
    }
  )

  it('never serializes live outgoing or validated receipt proof through either storage writer or a projection', async () => {
    let lost = true

    const loaded = await runtimeFixture(() => {
      if (lost) {
        throw new Error('response lost')
      }

      return { client_event_id: CLIENT_ID, event: userEvent() }
    })

    await loaded.runtime.sendHostedGroupChat('Board', loaded.chat.$groupChats.get().Board.log[0], 'work')

    for (const accepted of [false, true]) {
      if (accepted) {
        lost = false
        await loaded.runtime.dispatchHostedRoomOutbox()
      }

      const room = loaded.chat.$groupChats.get().Board

      const values = [
        (loaded.storage.get('group-chats') as Record<string, { log: GroupMessage[] }>).Board.log[0],
        loaded.chat.durableGroupChatRooms().Board.log[0],
        loaded.chat.groupChatSyncSnapshot().rooms['id:room-1'].log[0]
      ]

      expect(room.log[0].seq !== undefined).toBe(accepted)

      for (const value of values) {
        expect(value).not.toHaveProperty('clientEventId')
        expect(Object.getOwnPropertySymbols(value)).toHaveLength(0)
      }
    }
  })
})

describe('hosted projection sequence isolation', () => {
  it.each([false, true])('retains actual unsent text and subsequently accepts genuine replay, cold=%s', async cold => {
    const loaded = await runtimeFixture(() => {
      throw new Error('No send expected')
    }, [])

    const importedMember = canonicalUser(EVENT_ID, 6, {
      roomId: undefined,
      from: { kind: 'member', name: 'Legacy bot' },
      text: 'Imported text'
    })

    const hint = optimisticUser({ id: EVENT_ID, eventId: EVENT_ID, text: 'Imported text' })

    const projection = {
      version: 3,
      rooms: {
        'id:room-1': {
          name: 'Board',
          roomId: 'room-1',
          hosted: 'install:home',
          log: [importedMember, hint]
        }
      }
    }

    loaded.chat.$groupChats.set(
      loaded.chat.mergeRemoteGroupChatSnapshotIntoRooms(projection, loaded.chat.$groupChats.get())
    )

    if (cold) {
      const rooms = JSON.parse(JSON.stringify(loaded.chat.durableGroupChatRooms()))
      loaded.runtime.stopHostedRoomRuntime()
      loaded.chat.$groupChats.set(rooms)
      await loaded.runtime.startHostedRoomRuntime(loaded.ctx.storage)
    }

    loaded.chat.appendGroupChatEntry(
      'Board',
      { kind: 'user', name: 'You' },
      'Actual unsent local command',
      'work',
      undefined,
      CLIENT_ID
    )
    const room = loaded.chat.$groupChats.get().Board

    expect(room.log.some(entry => entry.id === CLIENT_ID && entry.text === 'Actual unsent local command')).toBe(true)
    expect(room.log.every(entry => entry.seq === undefined)).toBe(true)
    expect(room.hostedSeq).toBe(5)
    expect(loaded.calls.filter(call => call.method === 'groups.send')).toHaveLength(0)

    const memberReplay = {
      ...userEvent(EVENT_ID, 6, { text: 'Imported text', thread_id: 'work' }),
      kind: 'message.member',
      actor: { kind: 'member', id: 'research', profile: 'research' }
    }

    loaded.setEvents([memberReplay, userEvent('actual-other-user-event', 7)])
    await loaded.runtime.refreshHostedRooms()
    const replayed = loaded.chat.$groupChats.get().Board
    expect(replayed.hostedSeq).toBe(7)
    expect(replayed.log.find(entry => entry.id === EVENT_ID && entry.from.kind === 'member')?.seq).toBe(6)
    expect(replayed.log.find(entry => entry.id === EVENT_ID && entry.from.kind === 'user')?.seq).toBeUndefined()
    expect(replayed.log.find(entry => entry.id === 'actual-other-user-event')?.seq).toBe(7)
    expect(replayed.log.some(entry => entry.id === CLIENT_ID)).toBe(true)
  })
})
