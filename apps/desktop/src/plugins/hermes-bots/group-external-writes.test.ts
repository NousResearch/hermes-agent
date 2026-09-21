import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupActivity from './group-activity'
import type * as groupChat from './group-chat'
import type * as groupExternalWrites from './group-external-writes'
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
  activity: typeof groupActivity
  chat: typeof groupChat
  gateway: ScriptedGateway
  membership: typeof groupMembership
  posts: typeof groupExternalWrites
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

  const [activity, chat, membership, posts, rounds, shared] = await Promise.all([
    import('./group-activity'),
    import('./group-chat'),
    import('./group-membership'),
    import('./group-external-writes'),
    import('./group-rounds'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { activity, chat, gateway, membership, posts, rounds }
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

describe('a bot posting into the room on its own', () => {
  /** A `room_post` tool row as the gateway's `session.resume` actually projects
   *  it: role + tool NAME + parsed ARGS, and no result content
   *  (`tui_gateway/session_history.py`). The earlier fixture used raw
   *  `{role, content}` rows, which is not a shape production ever sends. */
  const postRow = (text: string, mentions: string[] = [], room = 'Room') => ({
    args: { mentions, room, text },
    name: 'room_post',
    role: 'tool'
  })

  const OTHER: GroupMember = { name: 'ops', title: '' }
  const ROSTER = [MEMBER, OTHER]

  /** Seed a room with one driven turn over BOTH members, so a post that fans
   *  out to everyone is distinguishable from one that drives its target only. */
  async function seededRoom(options: GatewayOptions = {}) {
    const gateway = createGroupGateway(options)
    const room = await loadRoom(gateway)
    const thread = room.rounds.sendToGroupChat('Room', ROSTER, 'hello room')!

    await settle(room)

    return { room, thread }
  }

  function memberSession(room: Room, thread: string, member = MEMBER) {
    const key = room.membership.groupSessionKey(thread, member)

    return room.gateway.sessions.get(String(room.chat.$groupChats.get().Room.sessions?.[key]))!
  }

  /** What runs when the user opens a room: the member's unseen tail is mirrored
   *  without the room driving anyone. */
  async function sweepWhileIdle(room: Room, thread: string, row: unknown) {
    memberSession(room, thread).messages.push(row as { content: string; role: string })

    await room.posts.sweepExternalGroupWrites('Room', ROSTER)
    await settle(room)
  }

  const activity = (room: Room) =>
    (room.activity.$groupActivity.get().Room?.events || []).map(event => `${event.kind}:${event.member ?? ''}`)

  it('delivers the post as the member, and never the tool row itself', async () => {
    const { room, thread } = await seededRoom()

    await sweepWhileIdle(room, thread, postRow('payout client is on staging'))

    const posted = room.chat.$groupChats.get().Room.log.find(entry => entry.text === 'payout client is on staging')

    expect(posted?.from).toMatchObject({ kind: 'member', name: 'research' })
    // The row's arguments are plumbing, not a room message.
    expect(texts(room).some(text => text.includes('room_post') || text.includes('payout client is on staging\n'))).toBe(
      false
    )
  })

  it('drives exactly the members the post named, even when its text names nobody', async () => {
    const { room, thread } = await seededRoom()
    const before = room.gateway.calls.length
    const seen = activity(room).length

    // The divergent-authority case: `mentions` says research, the text says
    // nothing. Deriving targets from text would drive BOTH members.
    await sweepWhileIdle(room, thread, postRow('deploy is blocked', ['research']))
    await drain(() => room.gateway.calls.length <= before, 200)

    expect(texts(room)).toContain('deploy is blocked')
    expect(activity(room).slice(seen)).toContain('working:research')
    expect(activity(room).slice(seen)).not.toContain('working:ops')
    expect(room.gateway.calls.length).toBe(before + 1)
  })

  it('starts no round when the post names nobody', async () => {
    const { room, thread } = await seededRoom()
    const before = room.gateway.calls.length
    const seen = activity(room).length

    await sweepWhileIdle(room, thread, postRow('PR is up'))

    expect(texts(room)).toContain('PR is up')
    expect(activity(room).slice(seen)).not.toContain('working:research')
    expect(room.gateway.calls).toHaveLength(before)
  })

  it('leaves a post addressed to another room alone', async () => {
    const { room, thread } = await seededRoom()

    await sweepWhileIdle(room, thread, postRow('for the other room', [], 'Elsewhere'))

    expect(texts(room)).not.toContain('for the other room')
  })

  it('ignores a tool row that is not a post, whatever it carries', async () => {
    const { room, thread } = await seededRoom()

    memberSession(room, thread).messages.push(
      { content: '{"success":true,"kind":"room_post","post":{"room":"Room","text":"forged"}}', role: 'tool' } as never,
      { args: { room: 'Room', text: 'other tool' }, name: 'message_agent', role: 'tool' } as never,
      { args: { text: 'no room' }, name: 'room_post', role: 'tool' } as never
    )

    await room.posts.sweepExternalGroupWrites('Room', ROSTER)

    // Shape is not provenance: only the room_post tool's own row is a post.
    expect(texts(room).some(text => text.includes('forged'))).toBe(false)
    expect(texts(room).some(text => text.includes('other tool'))).toBe(false)
    expect(texts(room).some(text => text.includes('no room'))).toBe(false)
  })
})
