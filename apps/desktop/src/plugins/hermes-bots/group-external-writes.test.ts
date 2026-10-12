import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupChat from './group-chat'
import type * as groupMembership from './group-membership'
import type * as groupRounds from './group-rounds'
import { createGroupGateway, drain, runTimersInline, scriptedStorage } from './group-test-utils'
import type { GatewayOptions, ScriptedGateway } from './group-test-utils'
import type { GroupChat, GroupMember } from './types'

// #93813: a member's per-group session is a plain Hermes session, so the CLI
// (`hermes -p <bot> chat --resume "Group: …"`), cron and the agent's tools
// write to it too. Those rows must reach the room log — once — or the room
// silently diverges from what the member actually said.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

interface Room {
  chat: typeof groupChat
  gateway: ScriptedGateway
  membership: typeof groupMembership
  rounds: typeof groupRounds
}

/** Fresh plugin modules over `gateway` — the first call is a cold start, a
 *  second call with the same gateway is a window restart: the gateway keeps
 *  its sessions and the plugin storage, the renderer keeps nothing. */
async function loadRoom(gateway: ScriptedGateway): Promise<Room> {
  vi.resetModules()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [chat, membership, rounds, shared] = await Promise.all([
    import('./group-chat'),
    import('./group-membership'),
    import('./group-rounds'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, gateway, membership, rounds }
}

/** What plugin.tsx rebuilds `$groupChats` from after a restart. */
function hydrateFromStorage(room: Room) {
  const durable = (room.gateway.storage.get('group-chats') || {}) as Record<string, GroupChat>
  const rooms: Record<string, GroupChat> = {}

  for (const [name, stored] of Object.entries(durable)) {
    rooms[name] = { ...stored, epoch: 0, running: false }
  }

  room.chat.$groupChats.set(rooms)
}

const MEMBER: GroupMember = { name: 'research', title: '' }
const options: GatewayOptions = { turn: ({ n }) => `room reply ${n}` }

const texts = (room: Room) => (room.chat.$groupChats.get().Room?.log || []).map(entry => entry.text)
const settle = (room: Room) => drain(() => Boolean(room.chat.$groupChats.get().Room?.running))

async function drive(room: Room, text: string, thread?: string) {
  const id = room.rounds.sendToGroupChat('Room', [MEMBER], text, thread)!
  await settle(room)

  return id
}

beforeEach(() => {
  runTimersInline()
  // Every clock read ticks a millisecond: the gateway mirror merge orders
  // same-millisecond entries by id, and inline timers land a whole drive in
  // one tick, which would shuffle the log the assertions read.
  let now = 1_000_000
  vi.spyOn(Date, 'now').mockImplementation(() => (now += 1))
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('external writes into a member session', () => {
  it('reach the room log exactly once across two drives and a window restart', async () => {
    const gateway = createGroupGateway(options)
    let room = await loadRoom(gateway)

    const thread = await drive(room, 'hello room')
    const key = room.membership.groupSessionKey(thread, MEMBER)
    const session = gateway.sessions.get(String(room.chat.$groupChats.get().Room.sessions?.[key]))!

    // The user resumes the member's session from the CLI between rounds.
    session.messages.push({ content: 'cli question', role: 'user' }, { content: 'cli answer', role: 'assistant' })

    await drive(room, 'second', thread)

    expect(texts(room)).toEqual(['hello room', 'room reply 1', 'second', 'cli question', 'cli answer', 'room reply 2'])

    const mirrored = room.chat.$groupChats.get().Room.log.filter(entry => entry.text.startsWith('cli '))

    for (const entry of mirrored) {
      expect(entry.from).toMatchObject({ kind: 'member', name: 'research' })
      expect(entry.thread).toBe(thread)
    }

    // The cursor is keyed by the session, sits past the swept rows, and is durable.
    const durable = (gateway.storage.get('group-chats') as Record<string, GroupChat>).Room

    expect(durable.externalCursors).toEqual({ [key]: 4 })

    // Window restart: same gateway and storage, fresh renderer state.
    room = await loadRoom(gateway)
    hydrateFromStorage(room)

    await drive(room, 'third', thread)

    expect(texts(room)).toEqual([
      'hello room',
      'room reply 1',
      'second',
      'cli question',
      'cli answer',
      'room reply 2',
      'third',
      'room reply 3'
    ])
  })

  it('leave room-fed prompts alone and surface outside writes on room open, without a drive', async () => {
    const gateway = createGroupGateway(options)
    const room = await loadRoom(gateway)
    const view = await import('./group-chat-view')

    const thread = await drive(room, 'hello room')
    await drive(room, 'second', thread)

    expect(texts(room)).toEqual(['hello room', 'room reply 1', 'second', 'room reply 2'])
    expect(room.gateway.calls).toHaveLength(2)

    // Nobody drives the room: a peer asks the member something in its session,
    // the agent answers a compaction handoff and a cron delivery (plumbing rows
    // the room never speaks for), then the user merely opens the room.
    const key = room.membership.groupSessionKey(thread, MEMBER)
    const session = gateway.sessions.get(String(room.chat.$groupChats.get().Room.sessions?.[key]))!
    session.messages.push(
      { content: 'manager: status?', role: 'user' },
      { content: 'status report: all green', role: 'assistant' },
      { content: '[CONTEXT COMPACTION — REFERENCE ONLY] summary…', role: 'user' },
      { content: 'noted the summary', role: 'assistant' },
      { content: 'Cronjob Response: nightly\nran', role: 'user' },
      { content: 'cron acknowledged', role: 'assistant' },
      { content: 'continue', display_kind: 'auto_continue', role: 'user' } as never,
      { content: 'nudged answer', role: 'assistant' }
    )

    // The member sits on this Desktop's roster, as it does in production; the
    // stored descriptor alone would read as a remote seat.
    const data = await import('./data')
    data.$lastRoster.set([{ name: 'research' }] as never)
    data.$botMeta.set({ research: { groups: ['Room'] } } as never)

    view.openGroupChat('Room')
    await drain(() => texts(room).length < 6)

    expect(texts(room)).toEqual([
      'hello room',
      'room reply 1',
      'second',
      'room reply 2',
      'manager: status?',
      'status report: all green'
    ])
    expect(room.gateway.calls).toHaveLength(2)
  })
})
