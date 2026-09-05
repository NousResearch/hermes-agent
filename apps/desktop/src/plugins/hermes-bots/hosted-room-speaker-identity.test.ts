import { describe, expect, it, vi } from 'vitest'

import { speakerEvent, speakerReplay, speakerRoom } from './group-speaker-test-fixtures'
import type { GroupMessage } from './types'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

function oldMirror(message: GroupMessage): GroupMessage {
  return { id: message.id, at: message.at, from: { kind: 'member', name: 'Former Product' }, text: message.text }
}

describe('hosted speaker identity through replay and display mirrors', () => {
  it('retains canonical actor identity without changing model-facing author text', () => {
    const [message] = speakerReplay()

    expect(message.from).toEqual({
      kind: 'member',
      name: 'Product',
      source: 'old-device-label',
      hostedIdentity: { roomId: 'room-1', memberId: 'pm', profile: 't2oracle' }
    })
  })

  it.each(['', 'gateway'])('does not invent a member identity for actor kind %j', kind => {
    const event = speakerEvent()

    event.actor.kind = kind
    expect(speakerReplay([event])[0].from).not.toHaveProperty('hostedIdentity')
  })

  it('does not substitute the display label or profile for an absent member ID', () => {
    const event = speakerEvent()

    event.actor.id = ''
    expect(speakerReplay([event])[0].from).not.toHaveProperty('hostedIdentity')
  })

  it('keeps user events separate from member identity', () => {
    const event = speakerEvent()

    event.kind = 'message.user'
    expect(speakerReplay([event])[0].from).toEqual({ kind: 'user', name: 'You', source: 'old-device-label' })
  })

  it('survives JSON persistence, bounded projection, and a second Desktop merge', async () => {
    const chat = await import('./group-chat')
    const room = speakerRoom()
    const durable = JSON.parse(JSON.stringify(chat.durableGroupChatRooms({ Board: room })))
    const projected = chat.groupChatSyncSnapshot(durable)
    const restored = chat.mergeRemoteGroupChatSnapshotIntoRooms(projected, {})

    expect(restored.Board.log[0].from).toHaveProperty('hostedIdentity', {
      roomId: 'room-1',
      memberId: 'pm',
      profile: 't2oracle'
    })
  })

  it.each([false, true])('keeps identity when an old renamed mirror merges, reversed=%s', reverse => {
    return import('./group-chat').then(chat => {
      const replay = speakerReplay()
      const mirror = replay.map(oldMirror)
      const merged = chat.mergeGroupChatSyncEntries(...(reverse ? [mirror, replay] : [replay, mirror]))

      expect(merged).toHaveLength(1)
      expect(merged[0].from).toHaveProperty('hostedIdentity', {
        roomId: 'room-1',
        memberId: 'pm',
        profile: 't2oracle'
      })
      expect(merged[0].seq).toBe(1)
    })
  })

  it('does not collapse different canonical events with identical display text/time/profile', async () => {
    const chat = await import('./group-chat')

    const replay = speakerReplay([
      speakerEvent(1, 'ux', 'default', 'Team'),
      speakerEvent(2, 'reviewer', 'default', 'Team'),
      speakerEvent(3, 'reviewer', 'default', 'Team')
    ])

    const merged = chat.mergeGroupChatSyncEntries(replay, replay)

    expect(merged.map(message => message.eventId)).toEqual(['event-1', 'event-2', 'event-3'])
    expect(merged.map(message => message.from)).toEqual(replay.map(message => message.from))
  })

  it('does not bind an idless old label-only entry to a canonical author', async () => {
    const chat = await import('./group-chat')
    const [message] = speakerReplay()

    const legacy = {
      ...message,
      id: undefined,
      eventId: undefined,
      seq: undefined,
      from: { kind: 'member' as const, name: message.from.name, source: message.from.source }
    }

    const merged = chat.mergeGroupChatSyncEntries([message], [legacy])

    expect(merged).toHaveLength(2)
    expect(merged.find(entry => !entry.id)?.from).not.toHaveProperty('hostedIdentity')
  })

  it('keeps classic idless mirror deduplication unchanged', async () => {
    const chat = await import('./group-chat')
    const legacy: GroupMessage = { at: 1, from: { kind: 'member', name: 'default' }, text: 'Hello' }

    expect(chat.mergeGroupChatSyncEntries([legacy], [{ ...legacy, thread: 'legacy-4' }])).toHaveLength(1)
  })

  it.each([false, true])(
    'an unsequenced display mirror cannot replace a sequenced actor, reversed=%s',
    async reverse => {
      const chat = await import('./group-chat')
      const [message] = speakerReplay()

      const mirror = {
        ...oldMirror(message),
        from: { ...message.from, hostedIdentity: { roomId: 'room-1', memberId: 'other', profile: 'default' } }
      }

      const merged = chat.mergeGroupChatSyncEntries(...(reverse ? [[mirror], [message]] : [[message], [mirror]]))

      expect(merged).toHaveLength(1)
      expect(merged[0].from.hostedIdentity?.memberId).toBe('pm')
    }
  )

  it('degrades conflicting equal-rank identity instead of choosing by label', async () => {
    const chat = await import('./group-chat')
    const [message] = speakerReplay()

    const conflict = {
      ...message,
      from: { ...message.from, hostedIdentity: { roomId: 'room-1', memberId: 'other', profile: 'default' } }
    }

    expect(chat.mergeGroupChatSyncEntries([message], [conflict])[0].from).not.toHaveProperty('hostedIdentity')
  })

  it('recovers the author from a sequence-matched old mirror', async () => {
    const chat = await import('./group-chat')
    const [message] = speakerReplay()
    const mirror = { ...oldMirror(message), id: undefined, seq: message.seq }

    expect(chat.mergeGroupChatSyncEntries([message], [mirror])[0].from).toHaveProperty(
      'hostedIdentity',
      message.from.hostedIdentity
    )
  })

  it('leaves the existing model transcript formatter unchanged', async () => {
    const { formatGroupChatLine } = await import('./group-rounds')
    const [message] = speakerReplay()
    const { hostedIdentity: _identity, ...from } = message.from

    expect(formatGroupChatLine(message, 't2oracle')).toBe(formatGroupChatLine({ ...message, from }, 't2oracle'))
    expect(formatGroupChatLine(message, 't2oracle')).toContain('[old-device-label]: Decision ready.')
  })

  it('carries replay identity through the real runtime writer, storage, and cold display hydration', async () => {
    vi.resetModules()
    vi.useFakeTimers()
    const { scriptedStorage } = await import('./group-test-utils')
    const storage = new Map<string, unknown>()
    const ctx = scriptedStorage(storage)

    const room = {
      room_id: 'room-1',
      name: 'Board',
      authority_gateway_id: 'install:home',
      authority_epoch: 1,
      members: [{ member_id: 'pm', profile: 't2oracle', display_name: 'Product', handle: 'pm' }],
      disbanded_at: null
    }

    const request = vi.fn(async (method: string) => {
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
        return { events: [speakerEvent()], latest_seq: 1, has_more: false }
      }

      throw new Error(`Unexpected RPC: ${method}`)
    })

    Object.assign(host, {
      activeConnectionId: () => 'gateway-a',
      profileRoutes: async () => [
        { connectionId: 'gateway-a', mode: 'remote', profile: 'default', targetProfile: 'default' }
      ],
      request,
      requestProfile: (_route: unknown, method: string) => request(method),
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

    try {
      await runtime.startHostedRoomRuntime(ctx.storage)
      const live = chat.$groupChats.get().Board

      expect(live.log[0].from).toHaveProperty('hostedIdentity', {
        roomId: 'room-1',
        memberId: 'pm',
        profile: 't2oracle'
      })
      expect(live.members?.[0].hostedIdentity?.memberId).toBe('pm')
      const persisted = JSON.parse(JSON.stringify(storage.get('group-chats')))
      const cold = chat.mergeRemoteGroupChatSnapshotIntoRooms(chat.groupChatSyncSnapshot(persisted), {})

      expect(cold.Board.log[0].from).toEqual(live.log[0].from)
      expect(request.mock.calls.map(([method]) => method)).toEqual(
        expect.arrayContaining(['groups.state', 'groups.log'])
      )
    } finally {
      runtime.stopHostedRoomRuntime()
      setPluginCtx(null)
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })
})
