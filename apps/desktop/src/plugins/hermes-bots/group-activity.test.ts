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
const agentActivity = vi.hoisted(() => ({ clear: vi.fn(), update: vi.fn() }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

const MEMBERS: GroupMember[] = [
  { name: 'research', title: '' },
  { name: 'builder', title: '' },
  { name: 'ops', title: 'The Ops' }
]

/** A failed turn's activity row carries the typed reason the gateway sent.
 *  `recordGroupActivity` spreads it through, so it isn't on the entry type. */
type ActivityRow = GroupActivityEntry & { reason?: string }

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
  host.agentActivity = agentActivity

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
  agentActivity.clear.mockReset()
  agentActivity.update.mockReset()
})

describe('turn arc', () => {
  it('mirrors a bot-to-bot handoff into the shared Agents activity tree', async () => {
    const room = await loadRoom()
    room.chat.updateGroupChat('Leadership', current => {
      current.epoch = 1
      current.log = []
      current.roomId = 'leadership-room'
      current.running = true

      return current
    })

    room.activity.recordGroupActivity('Leadership', {
      kind: 'queued',
      member: 'You',
      thread: 'thread-1'
    })
    room.activity.recordGroupActivity('Leadership', {
      kind: 'working',
      member: 'cto',
      source: 'ceo',
      thread: 'thread-1'
    })

    expect(agentActivity.clear).toHaveBeenCalledWith('bot-group:leadership-room')
    expect(agentActivity.update).toHaveBeenLastCalledWith(
      'bot-group:leadership-room',
      expect.objectContaining({ goal: 'ceo → cto · Leadership', status: 'running' })
    )

    room.activity.recordGroupActivity('Leadership', {
      kind: 'replied',
      member: 'cto',
      source: 'ceo',
      thread: 'thread-1'
    })

    expect(agentActivity.update).toHaveBeenLastCalledWith(
      'bot-group:leadership-room',
      expect.objectContaining({ status: 'completed', summary: 'Replied in Leadership' })
    )
  })

  it('keeps a timed-out room turn active until its late reply is delivered', async () => {
    const room = await loadRoom()
    room.chat.updateGroupChat('Slow', current => {
      current.log = []
      current.running = true

      return current
    })
    room.activity.recordGroupActivity('Slow', { kind: 'working', member: 'cto', source: 'ceo', thread: 't' })
    room.activity.recordGroupActivity('Slow', { kind: 'timed-out', member: 'cto', source: 'ceo', thread: 't' })

    expect(agentActivity.update).toHaveBeenLastCalledWith(
      'bot-group:Slow',
      expect.objectContaining({ status: 'running', text: expect.stringContaining('still working') })
    )

    room.activity.recordGroupActivity('Slow', { kind: 'delivered', member: 'cto', source: 'ceo', thread: 't' })

    expect(agentActivity.update).toHaveBeenLastCalledWith(
      'bot-group:Slow',
      expect.objectContaining({ status: 'completed', summary: 'Delivered a late reply in Slow' })
    )
  })

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
})

describe('epoch scoping', () => {
  it('a newer send interrupts the previous run and records cancelled in the CURRENT epoch', async () => {
    const gates = new Map<number, { promise: Promise<string>; resolve: (value: string) => void }>()

    const gate = (n: number) => {
      const existing = gates.get(n)

      if (existing) {
        return existing.promise
      }

      let resolve!: (value: string) => void

      const promise = new Promise<string>(settle => {
        resolve = settle
      })

      gates.set(n, { promise, resolve })

      return promise
    }

    const room = await loadRoom({ turn: ({ n }) => gate(n) })
    const member: GroupMember[] = [{ name: 'research', title: '' }]

    room.rounds.sendToGroupChat('Busy', member, 'first ask')
    await drain(() => room.gateway.calls.length < 1, 50)
    room.rounds.sendToGroupChat('Busy', member, 'second ask, supersede')
    await drain(() => room.gateway.calls.length < 2, 50)

    gates.get(2)?.resolve('from the new run')
    await drain(() => Boolean(room.chat.$groupChats.get().Busy?.running))
    gates.get(1)?.resolve('late from the old run')
    await drain(() => false)

    const epoch = room.chat.$groupChats.get().Busy?.epoch || 0

    expect(feed(room, 'Busy').some(event => event.kind === 'cancelled')).toBe(true)
    // The view shows only the current run: every visible event is this epoch.
    expect(room.activity.currentGroupActivity('Busy').every(event => (event.epoch || 0) === epoch)).toBe(true)
    expect(room.activity.currentGroupActivity('Busy').some(event => event.kind === 'cancelled')).toBe(true)
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
    expect(label({ kind: 'cancelled', member: null })).toBe('turn interrupted by a newer message')
    expect(label({ kind: 'settled', member: null })).toBe('turn settled')
  })
})
