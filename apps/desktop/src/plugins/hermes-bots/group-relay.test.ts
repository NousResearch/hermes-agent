import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupChat from './group-chat'
import type * as groupRelay from './group-relay'
import { createGroupGateway, drain, runTimersInline, scriptedStorage } from './group-test-utils'
import type { GatewayOptions, ScriptedGateway } from './group-test-utils'
import type { RosterRow } from './types'

// The Desktop half of `hermes group send` for Desktop-coordinated rooms: a
// gateway envelope becomes a sendToGroupChat ON THE USER'S BEHALF, and the
// round's progress streams back as group_relay.reply lines the CLI tails.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

interface Room {
  chat: typeof groupChat
  gateway: ScriptedGateway
  relay: typeof groupRelay
}

const MEMBERS: RosterRow[] = [
  { name: 'scout', title: '' } as RosterRow,
  { name: 'helper', title: '' } as RosterRow
]

async function loadRoom(options: GatewayOptions = {}): Promise<Room> {
  vi.resetModules()
  const gateway = createGroupGateway(options)

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [chat, relay, shared, data, membership] = await Promise.all([
    import('./group-chat'),
    import('./group-relay'),
    import('./shared'),
    import('./data'),
    import('./group-membership')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  // Seat two local bots in "Launchpad": roster rows whose meta lists the
  // group, exactly what groupChatMemberBots reads.
  data.$lastRoster.set(MEMBERS)
  data.$botMeta.set({
    scout: { groups: ['Launchpad'] },
    helper: { groups: ['Launchpad'] }
  } as never)
  void membership
  chat.updateGroupChat('Launchpad', current => {
    current.roomId = 'rm-launch'

    return current
  })

  return { chat, gateway, relay }
}

const replyLines = (room: Room, id: string) =>
  room.gateway
    .rpcFor('group_relay.reply')
    .filter(call => call.params.id === id)
    .map(call => call.params.line as Record<string, unknown>)

const isRunning = (room: Room) => room.chat.$groupChats.get().Launchpad?.running === true

beforeEach(() => {
  runTimersInline()
})

describe('resolveRelayRoom', () => {
  it('prefers the durable roomId, falls back to the exact display name', async () => {
    const room = await loadRoom()
    const rooms = room.chat.$groupChats.get()

    expect(room.relay.resolveRelayRoom({ id: 'e', room_id: 'rm-launch', room_name: 'Wrong' }, rooms)?.name).toBe('Launchpad')
    expect(room.relay.resolveRelayRoom({ id: 'e', room_id: 'nope', room_name: 'Launchpad' }, rooms)?.name).toBe('Launchpad')
    expect(room.relay.resolveRelayRoom({ id: 'e', room_id: 'nope', room_name: 'launchpad' }, rooms)).toBeNull()
  })
})

describe('handleGroupRelayEnvelope', () => {
  it('sends on the user\'s behalf with via provenance, streams replies in the new thread, then done', async () => {
    const room = await loadRoom({
      turn: ({ profile }) => (profile === 'scout' ? 'scout says hi' : '(pass)')
    })

    const ok = await room.relay.handleGroupRelayEnvelope({
      id: 'e1',
      label: 'Ada via Discord',
      room_id: 'rm-launch',
      room_name: 'Launchpad',
      text: 'What is the plan?'
    })

    expect(ok).toBe(true)
    await drain(() => isRunning(room))
    await room.relay.tickGroupRelayWatchers()

    const log = room.chat.$groupChats.get().Launchpad.log
    expect(log[0].from).toEqual({ kind: 'user', name: 'You', via: 'Ada via Discord' })
    expect(log[0].text).toBe('What is the plan?')

    const lines = replyLines(room, 'e1')
    expect(lines[0]).toMatchObject({ kind: 'accepted', group: 'Launchpad' })
    expect(lines[0].thread).toBe(log[0].thread)
    expect(lines.filter(l => l.kind === 'reply')).toEqual([
      expect.objectContaining({ member: 'scout', text: 'scout says hi', thread: log[0].thread })
    ])
    expect(lines[lines.length - 1]).toMatchObject({ kind: 'done', status: 'settled', replies: 1 })
    expect(room.relay.groupRelayWatcherIds()).toEqual([])
  })

  it('never re-streams entries that predate the relay and ignores other threads', async () => {
    const room = await loadRoom({ turn: () => 'reply' })

    room.chat.appendGroupChatEntry('Launchpad', { kind: 'member', name: 'helper' }, 'old news', 'tmtm-old')
    room.chat.appendGroupChatEntry('Launchpad', { kind: 'member', name: 'scout' }, 'other thread', 'tmtm-other')

    await room.relay.handleGroupRelayEnvelope({ id: 'e2', room_id: 'rm-launch', text: 'go' })
    await drain(() => isRunning(room))
    await room.relay.tickGroupRelayWatchers()

    const texts = replyLines(room, 'e2')
      .filter(l => l.kind === 'reply')
      .map(l => l.text)

    expect(texts).not.toContain('old news')
    expect(texts).not.toContain('other thread')
    expect(texts.length).toBeGreaterThan(0)
  })

  it('continues an existing Desktop thread only when the id is known; otherwise starts a new one', async () => {
    const room = await loadRoom({ turn: () => '(pass)' })

    room.chat.appendGroupChatEntry('Launchpad', { kind: 'user', name: 'You' }, 'earlier', 'tmtm-known')

    await room.relay.handleGroupRelayEnvelope({ id: 'e3', room_id: 'rm-launch', text: 'follow-up', thread: 'tmtm-known' })
    await drain(() => isRunning(room))
    expect(replyLines(room, 'e3')[0].thread).toBe('tmtm-known')

    await room.relay.handleGroupRelayEnvelope({ id: 'e4', room_id: 'rm-launch', text: 'new topic', thread: 'discord-session-42' })
    await drain(() => isRunning(room))
    const minted = String(replyLines(room, 'e4')[0].thread)
    expect(minted).not.toBe('discord-session-42')
    expect(minted.length).toBeGreaterThan(0)
  })

  it('streams a reply that lands synchronously during the send, before the watcher exists', async () => {
    const room = await loadRoom({ turn: () => '(pass)' })

    // Simulate a member reply appended INSIDE the send's synchronous window
    // (the first store notification fires before handleGroupRelayEnvelope
    // has registered its watcher). A watcher seeded after the send would
    // treat this entry as pre-existing and never stream it.
    let injected = false

    const unsub = room.chat.$groupChats.listen(rooms => {
      const log = rooms.Launchpad?.log || []
      const user = log.find(entry => entry.text === 'race')

      if (user && !injected) {
        injected = true
        room.chat.appendGroupChatEntry('Launchpad', { kind: 'member', name: 'scout' }, 'raced you', String(user.thread))
      }
    })

    await room.relay.handleGroupRelayEnvelope({ id: 'e9', room_id: 'rm-launch', text: 'race' })
    unsub()
    await drain(() => isRunning(room))
    await room.relay.tickGroupRelayWatchers()

    const lines = replyLines(room, 'e9')
    expect(lines.filter(l => l.kind === 'reply').map(l => l.text)).toContain('raced you')
    expect(lines.pop()).toMatchObject({ kind: 'done' })
  })

  it('reports room_not_found and no_members without touching any room', async () => {
    const room = await loadRoom()

    expect(await room.relay.handleGroupRelayEnvelope({ id: 'e5', room_id: 'missing', room_name: 'Nope', text: 'x' })).toBe(false)
    expect(replyLines(room, 'e5')[0]).toMatchObject({ kind: 'error', reason: 'room_not_found' })

    room.chat.updateGroupChat('Empty', current => current)
    expect(await room.relay.handleGroupRelayEnvelope({ id: 'e6', room_name: 'Empty', text: 'x' })).toBe(false)
    expect(replyLines(room, 'e6')[0]).toMatchObject({ kind: 'error', reason: 'no_members' })
    expect(room.gateway.calls).toEqual([])
  })

  it('reports cancelled when a newer send supersedes the relay\'s round', async () => {
    const room = await loadRoom({ turn: () => '(pass)' })

    await room.relay.handleGroupRelayEnvelope({ id: 'e7', room_id: 'rm-launch', text: 'first' })
    // Composer send bumps the epoch before the watcher has observed the end.
    room.chat.updateGroupChat('Launchpad', current => {
      current.epoch = (current.epoch || 0) + 1

      return current
    })
    await room.relay.tickGroupRelayWatchers()

    const last = replyLines(room, 'e7').pop()
    expect(last).toMatchObject({ kind: 'done', status: 'cancelled' })
  })

  it('times out a watcher whose round never ends', async () => {
    const room = await loadRoom({ turn: () => '(pass)' })

    await room.relay.handleGroupRelayEnvelope({ id: 'e8', room_id: 'rm-launch', text: 'stuck' })
    // Hold the room "running" forever and jump the clock.
    room.chat.updateGroupChat('Launchpad', current => {
      current.running = true

      return current
    })
    await room.relay.tickGroupRelayWatchers(Date.now() + room.relay.GROUP_RELAY_WATCH_TIMEOUT_MS + 1)

    expect(replyLines(room, 'e8').pop()).toMatchObject({ kind: 'done', status: 'timeout' })
  })
})

describe('drainGroupRelayOutbox', () => {
  it('claims envelopes from the gateway and acts on each; tolerates a gateway without the method', async () => {
    const room = await loadRoom({ turn: () => 'ok' })
    const base = host.request as (method: string, params?: Record<string, unknown>) => Promise<unknown>
    let served = false

    host.request = async (method: string, params: Record<string, unknown> = {}) => {
      if (method === 'group_relay.outbox.drain') {
        if (served) {
          return { envelopes: [] }
        }

        served = true

        return { envelopes: [{ id: 'd1', room_id: 'rm-launch', text: 'via drain', label: 'CLI' }] }
      }

      return base(method, params)
    }

    room.relay.startGroupRelay()
    await room.relay.drainGroupRelayOutbox()
    await drain(() => isRunning(room))
    await room.relay.tickGroupRelayWatchers()
    room.relay.stopGroupRelay()

    const entry = room.chat.$groupChats.get().Launchpad.log.find(e => e.text === 'via drain')
    expect(entry).toMatchObject({ text: 'via drain', from: { kind: 'user', name: 'You', via: 'CLI' } })
    expect(replyLines(room, 'd1').pop()).toMatchObject({ kind: 'done' })

    host.request = async (method: string) => {
      if (method === 'group_relay.outbox.drain') {
        throw new Error('unknown method')
      }

      return {}
    }

    await expect(room.relay.drainGroupRelayOutbox()).resolves.toBeUndefined()
  })
})
