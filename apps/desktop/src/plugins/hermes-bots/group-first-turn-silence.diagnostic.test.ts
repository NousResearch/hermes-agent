import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { GroupChatRoom } from './group-chat'
import { createGroupGateway, drain, runTimersInline, scriptedStorage } from './group-test-utils'
import type { GatewayOptions } from './group-test-utils'
import type { GroupMember } from './types'

// Exercises production routing + round/session engines, with only the gateway
// boundary scripted. No live profiles or configuration are touched.
const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

async function loadRoom(options: GatewayOptions = {}) {
  vi.resetModules()
  const gateway = createGroupGateway({ turn: () => '1 2 3 4 5 6 7 8 9 10', ...options })

  for (const key of Object.keys(host)) {delete host[key]}
  Object.assign(host, gateway.host)

  const [chat, rounds, routing, turns, shared] = await Promise.all([
    import('./group-chat'),
    import('./group-rounds'),
    import('./routing'),
    import('./group-turns'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, rounds, routing, turns, gateway }
}

const members: GroupMember[] = [
  { name: 'default', title: 'Hermes' },
  { name: 'builder', title: 'Builder' }
]

beforeEach(() => runTimersInline())

// Model a plain JSON-RPC/IPC rejection rather than the Error instance the
// existing scripted gateway returns. Preserve successes and all poll behavior.
function usePlainRejections() {
  const request = host.request as (method: string, params: Record<string, unknown>) => Promise<unknown>

  host.request = async (method: string, params: Record<string, unknown>) => {
    try {
      return await request(method, params)
    } catch (error) {
      const rpc = error as Error & { code?: number }
      throw { code: rpc.code, message: rpc.message }
    }
  }
}

describe('first-turn silence diagnostics', () => {
  it('control: @hermes count-to-10 selects default and posts its first answer', async () => {
    const room = await loadRoom()
    room.rounds.sendToGroupChat('Count', members, '@hermes count to 10')
    await drain(() => Boolean(room.chat.$groupChats.get().Count?.running))
    expect(room.gateway.calls.map(call => call.profile)).toEqual(['default'])
    expect(room.chat.$groupChats.get().Count.log.map(entry => entry.from.kind)).toEqual(['user', 'member'])
  })

  it.each([false, true])('plain 4007 retains code/data through requestForBot (routed=%s)', async routed => {
    const room = await loadRoom()
    const failure = Object.freeze({ code: 4007, name: 4007, message: 'session not found', data: { reason: 'missing' } })
    host[routed ? 'requestProfile' : 'request'] = vi.fn().mockRejectedValue(failure)
    const member = routed ? { name: 'default', connectionId: 'mini', remoteSource: true } : members[0]
    const caught = await room.routing.requestForBot(member, 'session.resume').catch(error => error)
    expect(caught).toBeInstanceOf(Error)
    expect(caught).toMatchObject({ name: 'Error', code: 4007, data: failure.data, cause: failure })
  })

  it('plain missing-session rejection allows repeated sends to produce answers', async () => {
    const room = await loadRoom()
    usePlainRejections()

    for (let attempt = 0; attempt < 3; attempt++) {
      room.rounds.sendToGroupChat('Count', members, '@hermes count to 10')
      await drain(() => Boolean(room.chat.$groupChats.get().Count?.running))
    }

    expect(room.gateway.rpcFor('session.create')).toHaveLength(1)
    expect(room.gateway.calls).toHaveLength(3)
    expect(room.chat.$groupChats.get().Count.log.map(entry => entry.from.kind)).toEqual([
      'user',
      'member',
      'user',
      'member',
      'user',
      'member'
    ])
    expect(room.chat.$groupChats.get().Count.running).toBe(false)
  })

  it('a plain JSON-RPC 4007 allows first-session creation', async () => {
    const room = await loadRoom()
    usePlainRejections()
    await expect(room.turns.ensureGroupChatSession('Count', members[0])).resolves.toHaveProperty('runtime')
  })

  it('reused hidden sessions receive distinct thread identity for repeated requests', async () => {
    const room = await loadRoom()
    const first = room.rounds.sendToGroupChat('Count', members, '@hermes count to 10')
    await drain(() => Boolean(room.chat.$groupChats.get().Count?.running))
    const second = room.rounds.sendToGroupChat('Count', members, '@hermes count to 10')
    await drain(() => Boolean(room.chat.$groupChats.get().Count?.running))
    expect(first).not.toBe(second)
    expect(room.gateway.calls).toHaveLength(2)
    expect(room.gateway.calls[0].stored).toBe(room.gateway.calls[1].stored)
    expect(room.gateway.calls[0].prompt).toContain(`[Current thread: ${first}]`)
    expect(room.gateway.calls[1].prompt).toContain(`[Current thread: ${second}]`)
    expect(room.gateway.calls[1].prompt).not.toContain(`[Current thread: ${first}]`)
    expect(room.gateway.calls[1].prompt).not.toContain('1 2 3 4 5 6 7 8 9 10')
    expect(room.gateway.calls[1].prompt).toContain('Current thread transcript:')
    expect(room.gateway.calls[1].prompt).toContain('A new user message is a renewed request')
    expect(room.gateway.calls[1].prompt).toContain('unless the user asks again')
  })

  it('scoped default with a literal default handle still answers @hermes', async () => {
    const room = await loadRoom()

    const roster = [
      { name: 'default', handle: 'default', sourceScoped: true, connectionId: 'local' },
      { name: 'ux-director', handle: 'ux-director', sourceScoped: true, connectionId: 'local' }
    ]

    expect([...room.rounds.parseGroupChatMentions('@hermes next', roster).mentioned]).toEqual(['local::default'])
  })

  it('raw default handles do not strand the return handoff after count 2', async () => {
    const room = await loadRoom({ turn: ({ n }) => ['1 @ux-director', '2 @hermes', '3'][n - 1] || '(pass)' })

    const roster = [
      { name: 'default', handle: 'default', sourceScoped: true, connectionId: 'local' },
      { name: 'ux-director', handle: 'ux-director', sourceScoped: true, connectionId: 'local' }
    ]

    room.rounds.sendToGroupChat('Count', roster, '@hermes count to 10 with @ux-director')
    await drain(() => Boolean(room.chat.$groupChats.get().Count?.running))
    const posted = room.chat.$groupChats.get().Count.log.filter(entry => entry.from.kind === 'member')
    expect(posted.map(entry => entry.text)).toEqual(['1 @ux-director', '2 @hermes', '3'])
    expect(posted.map(entry => entry.from.name)).toEqual(['default', 'ux-director', 'default'])
  })

  it('gateway transition before the first turn cancels the drive without a replacement', async () => {
    const room = await loadRoom()
    room.rounds.sendToGroupChat('Count', members, '@hermes count to 10')
    room.chat.handleSessionsGatewayTransition()
    await drain(() => Boolean(room.chat.$groupChats.get().Count?.running))
    expect(room.gateway.calls).toHaveLength(0)
    expect(room.chat.$groupChats.get().Count.log.map(entry => entry.from.kind)).toEqual(['user'])
    expect(room.chat.$groupChats.get().Count.running).toBe(false)
  })

  it('display-name mentions resolve, and an unknown mention falls back to everyone', async () => {
    const room = await loadRoom()
    const roster = [{ name: 'default', display_name: 'Research Buddy' }, { name: 'builder' }]
    expect([...room.rounds.parseGroupChatMentions('@research-buddy count to 10', roster).mentioned]).toEqual([
      'default'
    ])
    expect(
      room.rounds.resolveGroupResponders(
        [{ at: 1, from: { kind: 'user', name: 'You' }, text: '@unknown count to 10' }],
        roster
      )
    ).toEqual(roster)
  })

  it('ordinary same-room sync preserves orchestration runtime fields', async () => {
    const room = await loadRoom()
    room.chat.updateGroupChat('Count', (current: GroupChatRoom) => {
      current.epoch = 7
      current.running = true
      current.turn = 'default'
      current.sessions = { default: 'stored-id' }
      current.watermarks = { 'thread::default': 1 }

      return current
    })

    const merged = room.chat.mergeRemoteGroupChatSnapshotIntoRooms({
      version: 3,
      rooms: { 'name:Count': { name: 'Count', revision: 1, log: [], members: [] } }
    })

    expect(merged.Count).toMatchObject({
      epoch: 7,
      running: true,
      turn: 'default',
      sessions: { default: 'stored-id' },
      watermarks: { 'thread::default': 1 }
    })
  })
})
