import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as data from './data'
import type { GroupActivityEntry } from './group-activity'
import type * as groupActivity from './group-activity'
import type * as groupChat from './group-chat'
import type * as groupRounds from './group-rounds'
import { createGroupGateway, drain, runTimersInline, scriptedStorage } from './group-test-utils'
import type { GatewayOptions, ScriptedGateway } from './group-test-utils'
import type { GroupMember } from './types'

// Collapsible group Activity view: a runtime-only, bounded feed of truthful
// turn events (queued / working / replied / passed / timed-out / failed /
// cancelled / settled / delivered) that the room shows in a quiet disclosure.
// The transcript stays the only durable record — activity is never persisted.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

const MEMBERS: GroupMember[] = [
  { name: 'research', title: '' },
  { name: 'builder', title: '' },
  { name: 'ops', title: 'The Ops' }
]

/** A failed turn's activity row carries the typed reason the gateway sent. */
type ActivityRow = GroupActivityEntry

interface Room {
  activity: typeof groupActivity
  chat: typeof groupChat
  data: typeof data
  gateway: ScriptedGateway
  rounds: typeof groupRounds
}

async function loadRoom(options: GatewayOptions = {}): Promise<Room> {
  vi.resetModules()
  const gateway = createGroupGateway(options)

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [activity, chat, data, rounds, shared] = await Promise.all([
    import('./group-activity'),
    import('./group-chat'),
    import('./data'),
    import('./group-rounds'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { activity, chat, data, gateway, rounds }
}

function feed(room: Room, group: string): ActivityRow[] {
  return (room.activity.$groupActivity.get()[group]?.events || []) as ActivityRow[]
}

beforeEach(() => {
  runTimersInline()
})

describe('turn arc', () => {
  it('a settled turn records the full truthful arc: queued, working, replies and passes, settled', async () => {
    const room = await loadRoom({
      turn: ({ profile, prompt }) =>
        profile === 'research' && !prompt.includes('(you)') ? 'I looked into it — ship it.' : '(pass)'
    })

    room.rounds.sendToGroupChat('Room', MEMBERS, 'please review')
    await drain(() => Boolean(room.chat.$groupChats.get().Room?.running))

    const events = feed(room, 'Room')
    const kinds = events.map(event => event.kind)

    expect(kinds[0]).toBe('queued')
    expect(kinds[kinds.length - 1]).toBe('settled')
    expect(kinds.filter(kind => kind === 'working').length).toBeGreaterThanOrEqual(3)
    expect(kinds).toContain('replied')
    expect(kinds.filter(kind => kind === 'passed').length).toBeGreaterThanOrEqual(2)
    expect(events.find(event => event.kind === 'replied')?.member).toBe('research')
  })

  it('records source-scoped members by member key and labels them through their owner meta', async () => {
    // #94869 / #102294: bot meta lives under `connectionId::profile`, so an
    // activity row recorded by bare name could neither find the title nor
    // tell two same-named connections apart.
    const room = await loadRoom({ turn: () => 'on it' })

    const scoped: GroupMember = {
      connectionId: 'local',
      connectionKind: 'local',
      connectionLabel: 'This device',
      name: 'research',
      route: { connectionId: 'local', mode: 'local', profile: 'research', targetProfile: 'research' },
      sourceScoped: true,
      title: ''
    }

    room.data.$botMeta.set({ 'local::research': { title: 'Radar' } })
    room.data.$lastRoster.set([scoped])

    room.rounds.sendToGroupChat('Scoped', [scoped], 'status?')
    await drain(() => Boolean(room.chat.$groupChats.get().Scoped?.running))

    const events = feed(room, 'Scoped').filter(event => event.member && event.member !== 'You')

    expect(events.length).toBeGreaterThan(0)
    expect(events.every(event => event.member === 'local::research')).toBe(true)
    expect(events.map(event => room.activity.groupActivityLabel(event))).toContain('Radar replied')
  })

  it('a failed member turn records failed instead of a phantom reply', async () => {
    const room = await loadRoom({
      turn: ({ profile }) => {
        if (profile === 'builder') {
          throw new Error('gateway hiccup')
        }

        return '(pass)'
      }
    })

    room.rounds.sendToGroupChat('Flaky', MEMBERS, 'anyone around?')
    await drain(() => Boolean(room.chat.$groupChats.get().Flaky?.running))

    expect(feed(room, 'Flaky').some(event => event.kind === 'failed' && event.member === 'builder')).toBe(true)
  })

  it('a failed member turn preserves and prefers its typed reason', async () => {
    const room = await loadRoom({
      turn: ({ profile }) => {
        if (profile === 'builder') {
          throw Object.assign(new Error('generic gateway failure'), {
            data: { reason: 'provider_auth_or_access' }
          })
        }

        return '(pass)'
      }
    })

    room.rounds.sendToGroupChat('Typed failure', MEMBERS, 'anyone around?')
    await drain(() => Boolean(room.chat.$groupChats.get()['Typed failure']?.running))

    const failed = feed(room, 'Typed failure').find(event => event.kind === 'failed' && event.member === 'builder')

    expect(failed?.reason).toBe('provider_auth_or_access')
    expect(Object.values(room.data.$botAttention.get())[0]?.reason).toBe('provider_auth_or_access')
  })

  it('an untyped failed member turn keeps the message-classification fallback', async () => {
    const room = await loadRoom({
      turn: ({ profile }) => {
        if (profile === 'builder') {
          throw new Error('No LLM provider configured')
        }

        return '(pass)'
      }
    })

    room.rounds.sendToGroupChat('Untyped failure', MEMBERS, 'anyone around?')
    await drain(() => Boolean(room.chat.$groupChats.get()['Untyped failure']?.running))

    const failed = feed(room, 'Untyped failure').find(event => event.kind === 'failed' && event.member === 'builder')

    expect(failed?.reason).toBeUndefined()
    expect(Object.values(room.data.$botAttention.get())[0]?.reason).toBe('missing_config')
  })

  // #116458: a member whose backend never got a pool slot is healthy; the
  // room must not describe pool starvation as a bot crash (and never badge it).
  it('a slot-wait timeout reads as could-not-start, not as a bot error', async () => {
    const room = await loadRoom({
      turn: ({ profile }) => {
        if (profile === 'builder') {
          throw new Error(
            'Error invoking remote method \'hermes:api\': Local backend start for "builder" timed out while waiting for a free slot.'
          )
        }

        return '(pass)'
      }
    })

    room.rounds.sendToGroupChat('Slot wait', MEMBERS, 'anyone around?')
    await drain(() => Boolean(room.chat.$groupChats.get()['Slot wait']?.running))

    const failed = feed(room, 'Slot wait').find(event => event.kind === 'failed' && event.member === 'builder')

    expect(failed?.reason).toBe(room.activity.GROUP_SLOT_WAIT_REASON)
    expect(room.activity.groupActivityLabel(failed!, 'Slot wait')).toBe(
      "builder couldn't start — too many bots running"
    )
    expect(room.data.$botAttention.get()).toEqual({})
  })
})

describe('epoch scoping', () => {
  it('queues follow-ups without cancelling the active turn or losing its reply delta', async () => {
    let release!: (reply: string) => void
    const first = new Promise<string>(resolve => {
      release = resolve
    })
    const room = await loadRoom({ turn: ({ n }) => (n === 1 ? first : '(pass)') })
    const member: GroupMember[] = [{ name: 'research', title: '' }]
    const thread = room.rounds.sendToGroupChat('Busy', member, 'first ask')!
    await drain(() => room.gateway.calls.length < 1, 50)
    const epoch = room.chat.$groupChats.get().Busy.epoch
    room.rounds.sendToGroupChat('Busy', member, 'follow-up', thread)
    await drain(() => false)
    expect(room.gateway.calls).toHaveLength(1)
    expect(room.chat.$groupChats.get().Busy.epoch).toBe(epoch)
    release('first reply')
    await drain(() => room.gateway.calls.length < 2)
    await drain(() => Boolean(room.chat.$groupChats.get().Busy?.running))
    expect(room.gateway.calls).toHaveLength(2)
    expect(room.gateway.calls[1].prompt).toMatch(/follow-up[\s\S]*first reply/)
    expect(room.gateway.calls[1].prompt).not.toContain('first ask')
    expect(feed(room, 'Busy').some(event => event.kind === 'cancelled')).toBe(false)
  })

  it('epoch filtering drops events from a superseded run', async () => {
    const room = await loadRoom()
    room.chat.updateGroupChat('Epoch', current => {
      current.log = []
      current.epoch = 5

      return current
    })
    room.activity.recordGroupActivity('Epoch', { kind: 'queued', member: 'You' })
    room.activity.recordGroupActivity('Epoch', { kind: 'working', member: 'research' })

    room.chat.updateGroupChat('Epoch', current => {
      current.epoch = 6

      return current
    })

    expect(room.activity.currentGroupActivity('Epoch')).toHaveLength(0)

    room.activity.recordGroupActivity('Epoch', { kind: 'working', member: 'research' })
    const current = room.activity.currentGroupActivity('Epoch')

    expect(current).toHaveLength(1)
    expect(current[0].kind).toBe('working')
  })
})

describe('feed shape', () => {
  it('the feed is bounded: it stops growing and keeps the newest events', async () => {
    const room = await loadRoom()
    room.chat.updateGroupChat('Cap', current => {
      current.log = []

      return current
    })

    for (let i = 0; i < 500; i++) {
      room.activity.recordGroupActivity('Cap', { kind: 'working', member: `member-${i}` })
    }

    const bounded = feed(room, 'Cap')

    expect(bounded.length).toBeLessThan(500)
    expect(bounded[bounded.length - 1].member).toBe('member-499')
    expect(bounded.some(event => event.member === 'member-0')).toBe(false)

    room.activity.recordGroupActivity('Cap', { kind: 'working', member: 'one more' })

    expect(feed(room, 'Cap')).toHaveLength(bounded.length)
  })

  it('activity is runtime-only: never persisted, never hydrated', async () => {
    const room = await loadRoom()

    room.rounds.sendToGroupChat('Volatile', MEMBERS, 'hello')
    await drain(() => Boolean(room.chat.$groupChats.get().Volatile?.running))

    expect(feed(room, 'Volatile').length).toBeGreaterThan(0)
    expect([...room.gateway.storage.keys()]).not.toContain('group-activity')
  })

  it('labels read like a person wrote them, with settled/cancelled as room-level lines', async () => {
    const { activity } = await loadRoom()

    const label = (event: Omit<GroupActivityEntry, 'at' | 'epoch'>) =>
      activity.groupActivityLabel({ at: 0, epoch: 0, ...event })

    expect(label({ kind: 'queued', member: 'You' })).toBe('You sent a message')
    expect(label({ kind: 'replied', member: 'research' })).toBe('research replied')
    expect(label({ kind: 'timed-out', member: 'ops' })).toBe('ops took too long')
    expect(label({ kind: 'failed', member: 'ops', reason: 'slot wait timed out' })).toBe(
      'ops hit an error — slot wait timed out'
    )
    expect(label({ kind: 'cancelled', member: null })).toBe('turn interrupted by a newer message')
    expect(label({ kind: 'settled', member: null })).toBe('turn settled')
  })
})
