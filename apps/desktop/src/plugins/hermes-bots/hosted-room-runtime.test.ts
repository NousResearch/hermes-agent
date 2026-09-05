import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupChat from './group-chat'
import type * as groupRounds from './group-rounds'
import { pluginSdkMock, scriptedStorage } from './group-test-utils'
import type * as hostedRuntime from './hosted-room-runtime'
import type { GroupChat, GroupMember } from './types'

const { host } = vi.hoisted(() => ({
  host: {} as Record<string, unknown>
}))

vi.mock('@hermes/plugin-sdk', async () => pluginSdkMock(host))

interface RpcCall {
  connectionId?: string
  method: string
  params: Record<string, unknown>
}

interface RuntimeRoom {
  chat: typeof groupChat
  calls: RpcCall[]
  rounds: typeof groupRounds
  runtime: typeof hostedRuntime
  storage: Map<string, unknown>
}

const MEMBERS: GroupMember[] = [
  {
    name: 'research',
    connectionId: 'gateway-a',
    sourceScoped: true,
    targetProfile: 'research'
  },
  {
    name: 'builder',
    connectionId: 'gateway-a',
    sourceScoped: true,
    targetProfile: 'builder'
  }
]

function room(overrides: Partial<GroupChat> = {}): GroupChat {
  return {
    log: [],
    watermarks: {},
    members: MEMBERS,
    roomId: 'room-1',
    hosted: 'install:home',
    hostedEpoch: 1,
    hostedConnectionId: 'gateway-a',
    hostedSeq: 0,
    continuityMode: 'gateway',
    ...overrides
  }
}

function hostedEvent(
  seq: number,
  eventId: string,
  kind: string,
  payload: Record<string, unknown> = {},
  actor: Record<string, unknown> = {
    kind: 'gateway',
    id: 'install:home'
  }
) {
  return {
    room_id: 'room-1',
    seq,
    event_id: eventId,
    kind,
    actor,
    payload,
    created_at: seq
  }
}

async function loadRuntime(
  handler: (
    method: string,
    params: Record<string, unknown>,
    route?: Record<string, unknown>
  ) => Promise<unknown> | unknown,
  routes: Array<Record<string, unknown>> = [
    {
      connectionId: 'gateway-a',
      mode: 'remote' as const,
      profile: 'default',
      targetProfile: 'default'
    }
  ]
): Promise<RuntimeRoom> {
  vi.resetModules()
  const calls: RpcCall[] = []
  const storage = new Map<string, unknown>()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, {
    activeConnectionId: () => 'gateway-a',
    notify: vi.fn(),
    profileRoutes: async () => routes,
    request: async (method: string, params: Record<string, unknown>) => {
      calls.push({
        method,
        params
      })

      return handler(method, params)
    },
    requestProfile: async (route: Record<string, unknown>, method: string, params: Record<string, unknown>) => {
      calls.push({
        connectionId: String(route?.connectionId || ''),
        method,
        params
      })

      return handler(method, params, route)
    },
    state: {
      connectionId: {
        get: () => 'gateway-a',
        listen: () => () => undefined
      },
      gateway: {
        get: () => 'open',
        listen: () => () => undefined
      },
      profile: {
        get: () => 'default',
        listen: () => () => undefined
      }
    }
  })

  const [chat, rounds, runtime, shared] = await Promise.all([
    import('./group-chat'),
    import('./group-rounds'),
    import('./hosted-room-runtime'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(storage))

  return {
    chat,
    calls,
    rounds,
    runtime,
    storage
  }
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
})

describe('hosted Group Chat runtime', () => {
  it('hydrates after local state, reconciles optimistic ids, and replays one contiguous gateway log', async () => {
    const events = [
      hostedEvent(1, 'created-1', 'room.created', {
        name: 'Release',
        members: [
          {
            member_id: 'research',
            profile: 'research'
          },
          {
            member_id: 'builder',
            profile: 'builder'
          }
        ]
      }),
      hostedEvent(
        2,
        'user-1',
        'message.user',
        {
          text: 'Start',
          thread_id: 'thread-1'
        },
        {
          kind: 'user',
          id: 'desktop'
        }
      ),
      hostedEvent(
        3,
        'member-1',
        'message.member',
        {
          text: 'Done',
          thread_id: 'thread-1'
        },
        {
          kind: 'member',
          id: 'builder',
          display_name: 'Builder'
        }
      ),
      hostedEvent(4, 'settled-1', 'turn.settled')
    ]

    const loaded = await loadRuntime(method => {
      if (method === 'groups.capabilities') {
        return {
          driver: true,
          persistent_process: true,
          authority_gateway_id: 'install:home',
          max_log_limit: 100
        }
      }

      if (method === 'groups.list') {
        return {
          rooms: [
            {
              room_id: 'room-1',
              name: 'Release',
              members: MEMBERS,
              authority_gateway_id: 'install:home',
              authority_epoch: 1,
              disbanded_at: null
            }
          ]
        }
      }

      if (method === 'groups.state') {
        return {
          room: {
            room_id: 'room-1',
            name: 'Release',
            members: [
              {
                member_id: 'research',
                profile: 'research'
              },
              {
                member_id: 'builder',
                profile: 'builder'
              }
            ],
            authority_gateway_id: 'install:home',
            authority_epoch: 1,
            disbanded_at: null
          },
          driver_status: {
            working: false
          }
        }
      }

      if (method === 'groups.log') {
        return {
          events,
          latest_seq: 4,
          has_more: false
        }
      }

      throw new Error(`unexpected method: ${method}`)
    })

    loaded.chat.$groupChats.set({
      Release: room({
        log: [
          {
            at: 2,
            from: {
              kind: 'user',
              name: 'You'
            },
            id: 'user-1',
            text: 'Start',
            thread: 'thread-1'
          }
        ]
      })
    })
    await loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)

    const hydrated = loaded.chat.$groupChats.get().Release

    expect(hydrated.continuityMode).toBe('gateway')
    expect(hydrated.hostedSeq).toBe(4)
    expect(hydrated.log.map(entry => [entry.seq, entry.id, entry.text])).toEqual([
      [2, 'user-1', 'Start'],
      [3, 'member-1', 'Done']
    ])
    expect(hydrated.log.filter(entry => entry.id === 'user-1')).toHaveLength(1)
    expect(hydrated.running).toBe(false)
    expect(hydrated.hostedStatus).toMatchObject({
      state: 'ready',
      label: 'Ready'
    })
    expect(loaded.storage.get('group-chats')).toBeTruthy()

    loaded.runtime.stopHostedRoomRuntime()
  })

  it('replays the final events of a known room before marking a remote disband', async () => {
    const events = [
      hostedEvent(
        3,
        'member-1',
        'message.member',
        {
          text: 'Finished while Desktop was closed',
          thread_id: 'thread-1'
        },
        {
          kind: 'member',
          id: 'builder',
          display_name: 'Builder'
        }
      ),
      hostedEvent(4, 'disbanded-1', 'room.disbanded')
    ]

    const loaded = await loadRuntime((method, params) => {
      if (method === 'groups.capabilities') {
        return {
          authority_gateway_id: 'install:home',
          driver: true,
          max_log_limit: 1,
          persistent_process: true
        }
      }

      if (method === 'groups.list') {
        return {
          rooms: [
            {
              authority_epoch: 1,
              authority_gateway_id: 'install:home',
              disbanded_at: 4,
              latest_seq: 4,
              members: MEMBERS,
              name: 'Release',
              revision: 2,
              room_id: 'room-1'
            }
          ]
        }
      }

      if (method === 'groups.state') {
        expect(params.include_disbanded).toBe(true)

        return {
          driver_status: { working: false },
          room: {
            authority_epoch: 1,
            authority_gateway_id: 'install:home',
            disbanded_at: 4,
            latest_seq: 4,
            members: MEMBERS,
            name: 'Release',
            revision: 2,
            room_id: 'room-1'
          }
        }
      }

      if (method === 'groups.log') {
        expect(params.include_disbanded).toBe(true)
        expect(params.limit).toBe(1)
        const since = Number(params.since_seq)

        return {
          events: events.filter(event => event.seq > since).slice(0, 1),
          has_more: since < 3,
          latest_seq: 4
        }
      }

      throw new Error(`unexpected method: ${method}`)
    })

    loaded.chat.$groupChats.set({
      Release: room({
        hostedSeq: 2,
        log: [
          {
            at: 2,
            from: { kind: 'user', name: 'You' },
            id: 'user-1',
            seq: 2,
            text: 'Start',
            thread: 'thread-1'
          }
        ]
      })
    })
    loaded.chat.$groupHostedNeedsYou.set({ Release: true })
    loaded.chat.$groupClarify.set({
      'Release::builder': {
        at: 1,
        choices: ['once', 'deny'],
        group: 'Release',
        hostedApproval: {
          executionGeneration: 1,
          memberId: 'builder',
          roomId: 'room-1',
          taskId: 'task-1'
        },
        kind: 'approval',
        member: 'builder',
        memberKey: 'builder',
        multiSelect: false,
        question: 'Run command',
        requestId: 'approval-1'
      }
    })

    await loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)

    expect(loaded.chat.$groupChats.get().Release).toMatchObject({
      hostedSeq: 4,
      hostedStatus: { state: 'deleted' },
      continuityIssue: 'Delete it here to remove its local membership and history.'
    })
    expect(loaded.chat.$groupChats.get().Release.log.map(entry => entry.text)).toEqual([
      'Start',
      'Finished while Desktop was closed'
    ])
    expect(loaded.chat.$groupHostedNeedsYou.get().Release).toBeUndefined()
    expect(Object.values(loaded.chat.$groupClarify.get())).toEqual([])
    loaded.runtime.stopHostedRoomRuntime()
  })

  it.each(['state', 'log'] as const)(
    'does not paint a remote disband before terminal %s recovery succeeds',
    async failurePoint => {
      const loaded = await loadRuntime(method => {
        if (method === 'groups.capabilities') {
          return { authority_gateway_id: 'install:home', driver: true, persistent_process: true }
        }

        if (method === 'groups.list') {
          return {
            rooms: [
              {
                authority_epoch: 1,
                authority_gateway_id: 'install:home',
                disbanded_at: 4,
                latest_seq: 4,
                members: MEMBERS,
                name: 'Release',
                revision: 2,
                room_id: 'room-1'
              }
            ]
          }
        }

        if (method === 'groups.state') {
          if (failurePoint === 'state') {
            throw new Error('temporary state failure')
          }

          return {
            driver_status: { working: false },
            room: {
              authority_epoch: 1,
              authority_gateway_id: 'install:home',
              disbanded_at: 4,
              latest_seq: 4,
              members: MEMBERS,
              name: 'Release',
              revision: 2,
              room_id: 'room-1'
            }
          }
        }

        if (method === 'groups.log') {
          throw new Error('temporary log failure')
        }

        throw new Error(`unexpected method: ${method}`)
      })

      loaded.chat.$groupChats.set({
        Release: room({
          hostedSeq: 2,
          log: [
            {
              at: 2,
              from: { kind: 'user', name: 'You' },
              id: 'user-1',
              seq: 2,
              text: 'Start',
              thread: 'thread-1'
            }
          ]
        })
      })

      await loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)

      const unresolved = loaded.chat.$groupChats.get().Release

      expect(unresolved.hostedSeq).toBe(2)
      expect(unresolved.hostedStatus?.state).not.toBe('deleted')
      expect(unresolved.log.map(entry => entry.text)).toEqual(['Start'])
      loaded.runtime.stopHostedRoomRuntime()
    }
  )

  it('does not materialize a remotely disbanded room that this client never joined', async () => {
    const loaded = await loadRuntime(method => {
      if (method === 'groups.capabilities') {
        return { authority_gateway_id: 'install:home', driver: true, persistent_process: true }
      }

      if (method === 'groups.list') {
        return {
          rooms: [
            {
              authority_epoch: 1,
              authority_gateway_id: 'install:home',
              disbanded_at: 4,
              latest_seq: 4,
              members: MEMBERS,
              name: 'Release',
              revision: 2,
              room_id: 'room-1'
            }
          ]
        }
      }

      throw new Error(`unexpected method: ${method}`)
    })

    await loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)

    expect(loaded.chat.$groupChats.get()).toEqual({})
    expect(loaded.calls.some(call => call.method === 'groups.state')).toBe(false)
    expect(loaded.calls.some(call => call.method === 'groups.log')).toBe(false)
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('reconciles peer members without rewriting them onto the home gateway', async () => {
    const routes = [
      { connectionId: 'gateway-a', mode: 'remote' as const, profile: 'default', targetProfile: 'default' },
      { connectionId: 'gateway-b', mode: 'remote' as const, profile: 'default', targetProfile: 'default' }
    ]

    const serverMembers = [
      {
        handle: 'research',
        member_id: 'member-1-research',
        profile: 'research',
        target: { kind: 'local', profile: 'research' }
      },
      {
        handle: 'builder',
        member_id: 'member-2-builder',
        profile: 'builder',
        target: {
          installation_id: 'install:gateway-b',
          kind: 'peer',
          peer_id: 'install:gateway-b',
          profile: 'builder'
        }
      }
    ]

    const loaded = await loadRuntime((method, _params, route) => {
      const connectionId = String(route?.connectionId || '')

      if (method === 'groups.capabilities') {
        return {
          authority_gateway_id: `install:${connectionId}`,
          driver: true,
          persistent_process: true
        }
      }

      if (method === 'groups.list') {
        return {
          rooms:
            connectionId === 'gateway-a'
              ? [
                  {
                    authority_epoch: 1,
                    authority_gateway_id: 'install:gateway-a',
                    disbanded_at: null,
                    members: serverMembers,
                    name: 'Distributed',
                    room_id: 'room-1'
                  }
                ]
              : []
        }
      }

      if (method === 'groups.state') {
        return {
          driver_status: { working: false },
          room: {
            authority_epoch: 1,
            authority_gateway_id: 'install:gateway-a',
            disbanded_at: null,
            members: serverMembers,
            name: 'Distributed',
            room_id: 'room-1'
          }
        }
      }

      if (method === 'groups.log') {
        return { events: [], has_more: false, latest_seq: 0 }
      }

      throw new Error(`unexpected method: ${method}`)
    }, routes)

    loaded.chat.$groupChats.set({
      Distributed: room({
        continuityMode: 'distributed',
        members: [
          { connectionId: 'gateway-a', handle: 'research', name: 'research', sourceScoped: true },
          { connectionId: 'gateway-b', handle: 'builder', name: 'builder', sourceScoped: true }
        ]
      })
    })

    await loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)

    const reconciled = loaded.chat.$groupChats.get().Distributed

    expect(reconciled.continuityMode).toBe('distributed')
    expect(reconciled.members).toEqual([
      expect.objectContaining({ connectionId: 'gateway-a', name: 'research' }),
      expect.objectContaining({ connectionId: 'gateway-b', name: 'builder' })
    ])

    loaded.runtime.stopHostedRoomRuntime()
  })

  it('degrades an old gateway without starting the classic Desktop round driver', async () => {
    const missing = Object.assign(new Error('method not found'), {
      code: -32601
    })

    const loaded = await loadRuntime(method => {
      if (method === 'groups.capabilities' || method === 'groups.send') {
        throw missing
      }

      throw new Error(`unexpected method: ${method}`)
    })

    loaded.chat.$groupChats.set({
      Legacy: room({
        log: [
          {
            at: 1,
            from: {
              kind: 'user',
              name: 'You'
            },
            id: 'legacy-1',
            text: 'Keep going',
            thread: 'thread-1'
          }
        ]
      })
    })

    const localProjection = loaded.chat.groupChatSyncSnapshot(loaded.chat.$groupChats.get())

    const mergedProjection = loaded.chat.mergeGroupChatSyncSnapshots(
      {
        version: 3,
        rooms: {
          'id:room-1': {
            name: 'Legacy',
            roomId: 'room-1',
            log: [],
            revision: 9,
            hosted: 'install:untrusted-projection',
            hostedEpoch: 9,
            continuityMode: 'gateway'
          }
        }
      },
      localProjection
    )

    expect(mergedProjection.rooms['id:room-1']).toMatchObject({
      hosted: 'install:home',
      hostedEpoch: 1,
      continuityMode: 'gateway'
    })

    await loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)

    expect(loaded.chat.$groupChats.get().Legacy).toMatchObject({
      hosted: 'install:home',
      continuityMode: 'gateway',
      running: false,
      hostedStatus: {
        state: 'unsupported',
        label: 'Update this device to keep this Group Chat running.'
      }
    })

    const thread = loaded.rounds.sendToGroupChat('Legacy', MEMBERS, 'Continue', null, [])

    expect(thread).toBeNull()
    await Promise.resolve()
    await Promise.resolve()
    expect(loaded.calls.some(call => call.method === 'session.create' || call.method === 'prompt.submit')).toBe(false)
    expect(loaded.chat.$groupChats.get().Legacy.hosted).toBe('install:home')
    expect(loaded.chat.$groupChats.get().Legacy.continuityIssue).toBe(
      'Update this device to keep this Group Chat running.'
    )

    loaded.runtime.stopHostedRoomRuntime()
  })

  it('persists send, stop, and disband commands before dispatch and acknowledges them idempotently', async () => {
    const loaded = await loadRuntime(method => {
      if (method === 'groups.capabilities') {
        return {
          driver: true,
          persistent_process: true,
          authority_gateway_id: 'install:home'
        }
      }

      if (method === 'groups.list') {
        return {
          rooms: []
        }
      }

      if (method === 'groups.create') {
        return {
          room: {
            room_id: 'room-new',
            authority_gateway_id: 'install:home',
            authority_epoch: 1
          }
        }
      }

      if (method === 'groups.send' || method === 'groups.stop' || method === 'groups.disband') {
        return {
          ok: true
        }
      }

      throw new Error(`unexpected method: ${method}`)
    })

    loaded.chat.$groupChats.set({
      Release: room()
    })
    await loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)

    const probe = await loaded.runtime.probeHostedRoomMembers(MEMBERS)

    await expect(
      loaded.runtime.createHostedGroupChat({
        route: probe.route,
        roomId: 'room-new',
        name: 'New Group',
        members: MEMBERS.map(member => ({
          member_id: member.name,
          profile: member.name,
          handle: member.name
        }))
      })
    ).resolves.toEqual({
      authorityId: 'install:home',
      authorityEpoch: 1,
      connectionId: 'gateway-a'
    })

    await expect(
      loaded.runtime.sendHostedGroupChat(
        'Release',
        {
          at: 1,
          from: {
            kind: 'user',
            name: 'You'
          },
          id: 'send-1',
          text: 'Ship it',
          thread: 'thread-1'
        },
        'thread-1'
      )
    ).resolves.toBe(true)
    await expect(loaded.runtime.stopHostedGroupChat('Release')).resolves.toBe(true)
    await expect(loaded.runtime.disbandHostedGroupChat('Release')).resolves.toBe(true)

    expect(loaded.calls.map(call => call.method)).toEqual(
      expect.arrayContaining(['groups.create', 'groups.send', 'groups.stop', 'groups.disband'])
    )
    expect((loaded.storage.get('hosted-room-outbox-v1') as { commands: unknown[] }).commands).toEqual([])

    loaded.runtime.stopHostedRoomRuntime()
  })

  it('confirms an unknown create outcome before allowing Desktop fallback', async () => {
    const loaded = await loadRuntime(method => {
      if (method === 'groups.capabilities') {
        return {
          driver: true,
          persistent_process: true,
          authority_gateway_id: 'install:home'
        }
      }

      if (method === 'groups.list') {
        return {
          rooms: []
        }
      }

      if (method === 'groups.create') {
        throw new Error('response lost')
      }

      if (method === 'groups.state') {
        return {
          room: {
            room_id: 'room-new',
            name: 'New Group',
            authority_gateway_id: 'install:home',
            authority_epoch: 1
          }
        }
      }

      throw new Error(`unexpected method: ${method}`)
    })

    await loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)
    const probe = await loaded.runtime.probeHostedRoomMembers(MEMBERS)

    await expect(
      loaded.runtime.createHostedGroupChat({
        route: probe.route,
        roomId: 'room-new',
        name: 'New Group',
        members: MEMBERS.map(member => ({
          member_id: member.name,
          profile: member.name,
          handle: member.name
        }))
      })
    ).resolves.toEqual({
      authorityId: 'install:home',
      authorityEpoch: 1,
      connectionId: 'gateway-a'
    })
    expect(loaded.calls.map(call => call.method)).toEqual(expect.arrayContaining(['groups.create', 'groups.state']))

    loaded.runtime.stopHostedRoomRuntime()
  })

  it('retries a hosted rename with the same idempotency key', async () => {
    let renameAttempts = 0

    const loaded = await loadRuntime(method => {
      if (method === 'groups.capabilities') {
        return {
          driver: true,
          persistent_process: true,
          authority_gateway_id: 'install:home'
        }
      }

      if (method === 'groups.list') {
        return {
          rooms: []
        }
      }

      if (method === 'groups.rename') {
        renameAttempts += 1

        if (renameAttempts === 1) {
          throw new Error('connection closed')
        }

        return {
          room: {
            room_id: 'room-1',
            name: 'Renamed'
          }
        }
      }

      throw new Error(`unexpected method: ${method}`)
    })

    loaded.chat.$groupChats.set({
      Release: room()
    })
    await loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)

    const staleRefreshGeneration = loaded.runtime.beginHostedRoomMutation('room-1')

    await expect(loaded.runtime.renameHostedGroupChat('Release', 'Renamed')).resolves.toBe(false)

    expect(loaded.runtime.hostedRoomMutationIsCurrent('room-1', staleRefreshGeneration)).toBe(false)

    const pending = loaded.storage.get('hosted-room-outbox-v1') as {
      commands: Array<{ commandId: string; kind: string; status: string }>
    }

    expect(pending.commands).toEqual([
      expect.objectContaining({
        kind: 'rename',
        status: 'pending'
      })
    ])

    await loaded.runtime.dispatchHostedRoomOutbox()

    const calls = loaded.calls.filter(call => call.method === 'groups.rename')

    expect(calls).toHaveLength(2)
    expect(calls[0].params.event_id).toBe(calls[1].params.event_id)
    expect(calls[1].params).toMatchObject({
      room_id: 'room-1',
      name: 'Renamed'
    })
    expect(loaded.storage.get('hosted-room-outbox-v1')).toMatchObject({
      commands: []
    })

    loaded.runtime.stopHostedRoomRuntime()
  })

  it('replays an unknown in-flight send after Desktop closes with the same command id', async () => {
    let releaseFirstSend: () => void = () => undefined
    let firstSend = true

    const loaded = await loadRuntime(method => {
      if (method === 'groups.capabilities') {
        return {
          driver: true,
          persistent_process: true,
          authority_gateway_id: 'install:home'
        }
      }

      if (method === 'groups.list') {
        return {
          rooms: []
        }
      }

      if (method === 'groups.send' && firstSend) {
        firstSend = false

        return new Promise(resolve => {
          releaseFirstSend = () =>
            resolve({
              ok: true
            })
        })
      }

      if (method === 'groups.send') {
        return {
          ok: true
        }
      }

      throw new Error(`unexpected method: ${method}`)
    })

    loaded.chat.$groupChats.set({
      Release: room()
    })
    const storage = scriptedStorage(loaded.storage).storage

    await loaded.runtime.startHostedRoomRuntime(storage)

    const delivery = loaded.runtime.sendHostedGroupChat(
      'Release',
      {
        at: 1,
        from: {
          kind: 'user',
          name: 'You'
        },
        id: 'send-after-close',
        text: 'Keep working',
        thread: 'thread-1'
      },
      'thread-1'
    )

    for (let attempt = 0; attempt < 1000; attempt++) {
      const persisted = loaded.storage.get('hosted-room-outbox-v1') as {
        commands?: Array<{ status?: string }>
      }

      if (
        persisted?.commands?.[0]?.status === 'in-flight' &&
        loaded.calls.some(call => call.method === 'groups.send')
      ) {
        break
      }

      await Promise.resolve()
    }

    expect(loaded.storage.get('hosted-room-outbox-v1')).toMatchObject({
      commands: [
        {
          commandId: 'send-after-close',
          status: 'in-flight'
        }
      ]
    })
    expect(loaded.calls.filter(call => call.method === 'groups.send')).toHaveLength(1)

    loaded.runtime.stopHostedRoomRuntime()
    releaseFirstSend()
    await expect(delivery).resolves.toBe(false)

    await loaded.runtime.startHostedRoomRuntime(storage)

    expect(loaded.calls.filter(call => call.method === 'groups.send')).toHaveLength(2)
    expect(loaded.storage.get('hosted-room-outbox-v1')).toMatchObject({
      commands: []
    })

    loaded.runtime.stopHostedRoomRuntime()
  })

  it.each(['send', 'disband'] as const)(
    'replays a persisted %s after the hosted worker recovers from 4123',
    async kind => {
      let available = false
      let accepted = 0

      const loaded = await loadRuntime(method => {
        if (method === 'groups.capabilities') {
          return {
            driver: true,
            persistent_process: true,
            authority_gateway_id: 'install:home'
          }
        }

        if (method === 'groups.list') {
          return { rooms: [] }
        }

        if (method === `groups.${kind}`) {
          if (!available) {
            throw Object.assign(new Error('Group Chat worker is unavailable'), {
              code: 4123
            })
          }

          accepted += 1

          return { ok: true }
        }

        throw new Error(`unexpected method: ${method}`)
      })

      loaded.chat.$groupChats.set({ Release: room() })
      const storage = scriptedStorage(loaded.storage).storage

      await loaded.runtime.startHostedRoomRuntime(storage)

      const submitted =
        kind === 'send'
          ? await loaded.runtime.sendHostedGroupChat(
              'Release',
              {
                at: 1,
                from: { kind: 'user', name: 'You' },
                id: 'worker-restart-send',
                text: 'Keep working',
                thread: 'thread-1'
              },
              'thread-1'
            )
          : await loaded.runtime.disbandHostedGroupChat('Release')

      expect(submitted).toBe(false)

      const persisted = loaded.storage.get('hosted-room-outbox-v1') as {
        commands: Array<{ commandId: string; status: string }>
      }

      expect(persisted.commands).toHaveLength(1)
      expect(persisted.commands[0].status).toBe('pending')

      const commandId = persisted.commands[0].commandId

      loaded.runtime.stopHostedRoomRuntime()
      available = true
      await loaded.runtime.startHostedRoomRuntime(storage)

      const calls = loaded.calls.filter(call => call.method === `groups.${kind}`)

      expect(calls).toHaveLength(2)
      expect(calls.map(call => String(call.params.event_id || call.params.cancel_id || ''))).toEqual([
        commandId,
        commandId
      ])
      expect(accepted).toBe(1)
      expect(loaded.storage.get('hosted-room-outbox-v1')).toMatchObject({
        commands: []
      })

      loaded.runtime.stopHostedRoomRuntime()
    }
  )

  it('creates a multi-host Group Chat with target-issued scoped grants', async () => {
    const routes = [
      { connectionId: 'host-a', mode: 'remote' as const, profile: 'default', targetProfile: 'default' },
      { connectionId: 'host-b', mode: 'remote' as const, profile: 'default', targetProfile: 'default' },
      { connectionId: 'host-b', mode: 'remote' as const, profile: 'builder', targetProfile: 'builder' }
    ]

    const loaded = await loadRuntime((method, _params, route) => {
      const connectionId = String(route?.connectionId || '')

      if (method === 'groups.capabilities') {
        return {
          authority_gateway_id: `install:${connectionId}`,
          driver: true,
          persistent_process: true,
          room_link: {
            enabled: true,
            endpoint: {
              available: true,
              url: `https://${connectionId}.example.test:19445`
            },
            catalog: {
              attachments: false,
              catalog_digest: `digest:${connectionId}`,
              installation_id: `install:${connectionId}`,
              link_modes: ['direct'],
              persistent_process: true,
              protocol_versions: [2],
              text: true
            }
          }
        }
      }

      if (method === 'groups.list') {
        return { rooms: [] }
      }

      if (method === 'groups.peer.invite') {
        return {
          grant: 'grant:builder',
          target_profile: 'builder',
          catalog: {
            attachments: false,
            catalog_digest: 'digest:host-b',
            installation_id: 'install:host-b',
            link_modes: ['direct'],
            persistent_process: true,
            protocol_versions: [2],
            text: true
          }
        }
      }

      if (method === 'groups.create') {
        return {
          room: {
            authority_epoch: 1,
            authority_gateway_id: 'install:host-a',
            room_id: 'room-multi'
          }
        }
      }

      if (method === 'groups.peer.register') {
        return { registered: true }
      }

      throw new Error(`unexpected method: ${method}`)
    }, routes)

    const storage = scriptedStorage(loaded.storage).storage

    await loaded.runtime.startHostedRoomRuntime(storage)

    const members: GroupMember[] = [
      {
        connectionId: 'host-a',
        name: 'research',
        route: routes[0],
        sourceScoped: true,
        targetProfile: 'research'
      },
      {
        connectionId: 'host-b',
        name: 'builder',
        route: routes[2],
        sourceScoped: true,
        targetProfile: 'builder'
      }
    ]

    const probe = await loaded.runtime.probeHostedRoomMembers(members)

    expect(probe.route).toMatchObject({
      homeConnectionId: 'host-a',
      kind: 'multi-gateway',
      remoteConnectionIds: ['host-b']
    })
    await expect(
      loaded.runtime.createAutonomousHostedGroupChat({
        members: [
          { handle: 'research', member: members[0], profile: 'research' },
          { handle: 'builder', member: members[1], profile: 'builder' }
        ],
        name: 'Multi',
        probe,
        roomId: 'room-multi'
      })
    ).resolves.toMatchObject({
      authorityId: 'install:host-a',
      connectionId: 'host-a',
      continuityMode: 'distributed'
    })

    expect(loaded.calls.find(call => call.method === 'groups.peer.invite')?.connectionId).toBe('host-b')
    expect(loaded.calls.find(call => call.method === 'groups.create')?.connectionId).toBe('host-a')
    expect(loaded.calls.find(call => call.method === 'groups.peer.register')?.params).toMatchObject({
      grant: 'grant:builder',
      member_id: 'member-2-builder',
      room_id: 'room-multi',
      target_profile: 'builder',
      target_url: 'https://host-b.example.test:19445/p/builder'
    })
    expect((loaded.storage.get('hosted-room-cleanup-v1') as { operations: unknown[] }).operations).toEqual([])

    loaded.runtime.stopHostedRoomRuntime()
  })

  it('durably disbands and revokes a partial multi-host setup', async () => {
    const routes = [
      { connectionId: 'host-a', mode: 'remote' as const, profile: 'default', targetProfile: 'default' },
      { connectionId: 'host-b', mode: 'remote' as const, profile: 'default', targetProfile: 'default' },
      { connectionId: 'host-b', mode: 'remote' as const, profile: 'builder', targetProfile: 'builder' }
    ]

    let cleanupAvailable = false

    const loaded = await loadRuntime((method, _params, route) => {
      const connectionId = String(route?.connectionId || '')

      if (method === 'groups.capabilities') {
        return {
          authority_gateway_id: `install:${connectionId}`,
          driver: true,
          persistent_process: true,
          room_link: {
            enabled: true,
            endpoint: { available: true, url: `https://${connectionId}.example.test:19445` },
            catalog: {
              attachments: false,
              catalog_digest: `digest:${connectionId}`,
              installation_id: `install:${connectionId}`,
              link_modes: ['direct'],
              persistent_process: true,
              protocol_versions: [2],
              text: true
            }
          }
        }
      }

      if (method === 'groups.list') {
        return { rooms: [] }
      }

      if (method === 'groups.peer.invite') {
        return {
          grant: 'grant:builder',
          target_profile: 'builder',
          catalog: {
            attachments: false,
            catalog_digest: 'digest:host-b',
            installation_id: 'install:host-b',
            link_modes: ['direct'],
            persistent_process: true,
            protocol_versions: [2],
            text: true
          }
        }
      }

      if (method === 'groups.create' || method === 'groups.state') {
        throw new Error('create failed')
      }

      if (method === 'groups.disband' || method === 'groups.peer.revoke') {
        if (!cleanupAvailable) {
          throw new Error('device offline')
        }

        return { ok: true }
      }

      throw new Error(`unexpected method: ${method}`)
    }, routes)

    await loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)

    const members: GroupMember[] = [
      { connectionId: 'host-a', name: 'research', route: routes[0], sourceScoped: true, targetProfile: 'research' },
      { connectionId: 'host-b', name: 'builder', route: routes[2], sourceScoped: true, targetProfile: 'builder' }
    ]

    const probe = await loaded.runtime.probeHostedRoomMembers(members)

    const failure = await loaded.runtime
      .createAutonomousHostedGroupChat({
        members: [
          { handle: 'research', member: members[0], profile: 'research' },
          { handle: 'builder', member: members[1], profile: 'builder' }
        ],
        name: 'Partial',
        probe,
        roomId: 'room-partial'
      })
      .catch(error => error as Error & { fallbackSafe?: boolean })

    expect(failure).toMatchObject({
      fallbackSafe: false,
      message: expect.stringContaining('could not finish cleanup')
    })
    expect((loaded.storage.get('hosted-room-cleanup-v1') as { operations: unknown[] }).operations).not.toEqual([])

    cleanupAvailable = true
    loaded.runtime.stopHostedRoomRuntime()
    await loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)

    expect(loaded.calls.map(call => call.method)).toEqual(
      expect.arrayContaining(['groups.disband', 'groups.peer.revoke'])
    )
    expect((loaded.storage.get('hosted-room-cleanup-v1') as { operations: unknown[] }).operations).toEqual([])

    loaded.runtime.stopHostedRoomRuntime()
  })

  it('reprobes deterministic unsupported gateways only after the 30-second cache expires', async () => {
    vi.setSystemTime(100)
    let probes = 0
    const missing = Object.assign(new Error('method not found'), { code: -32601 })

    const loaded = await loadRuntime(method => {
      if (method === 'groups.capabilities') {
        probes += 1
        throw missing
      }

      throw new Error(`unexpected method: ${method}`)
    })

    await loaded.runtime.probeHostedRoomMembers(MEMBERS)
    await loaded.runtime.probeHostedRoomMembers(MEMBERS)
    expect(probes).toBe(1)

    vi.setSystemTime(30_101)
    await loaded.runtime.probeHostedRoomMembers(MEMBERS)
    expect(probes).toBe(2)
  })

  it('keeps a projection-only room read-only until its member gateway inventory settles', async () => {
    const loaded = await loadRuntime(method => {
      if (method === 'groups.capabilities') {
        return {
          authority_gateway_id: 'install:home',
          driver: true,
          persistent_process: true
        }
      }

      if (method === 'groups.list') {
        return { rooms: [] }
      }

      throw new Error(`unexpected method: ${method}`)
    })

    const projected = room({
      hosted: null,
      hostedConnectionId: null,
      hostedEpoch: null,
      continuityMode: 'desktop',
      members: MEMBERS.map(member => ({ ...member, remoteSource: true }))
    })

    loaded.chat.$groupChats.set({ Projected: projected })
    expect(loaded.runtime.groupChatContinuityReady(projected)).toBe(false)
    expect(loaded.rounds.sendToGroupChat('Projected', projected.members || [], 'Do not double-drive')).toBeNull()
    expect(loaded.chat.$groupChats.get().Projected.continuityIssue).toBe('Syncing recent activity…')
    expect(loaded.calls.some(call => call.method === 'session.create' || call.method === 'prompt.submit')).toBe(false)
    expect(
      loaded.runtime.groupChatContinuityReady({
        ...projected,
        members: MEMBERS
      })
    ).toBe(true)

    await loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)

    expect(loaded.runtime.groupChatContinuityReady(projected)).toBe(true)
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('keeps a remotely deleted room read-only without painting a local send', async () => {
    const loaded = await loadRuntime(method => {
      throw new Error(`deleted room must not dispatch: ${method}`)
    })

    const deleted = room({
      hostedStatus: { label: 'Deleted', state: 'deleted' },
      running: false
    })

    loaded.chat.$groupChats.set({ Deleted: deleted })
    expect(loaded.runtime.groupChatContinuityReady(deleted)).toBe(false)
    expect(loaded.rounds.sendToGroupChat('Deleted', MEMBERS, 'must not paint')).toBeNull()
    expect(loaded.chat.$groupChats.get().Deleted.log).toEqual([])
    expect(loaded.calls.some(call => call.method === 'groups.send')).toBe(false)
  })

  it('does not enqueue another Stop while a hosted Stop is already pending', async () => {
    const loaded = await loadRuntime(method => {
      throw new Error(`stopping room must not dispatch: ${method}`)
    })

    loaded.chat.$groupChats.set({
      Stopping: room({
        hostedStatus: { canStop: false, label: 'Stopping…', state: 'stopping' },
        running: true
      })
    })
    await loaded.rounds.stopGroupThread('Stopping', null, MEMBERS)

    expect(loaded.calls.some(call => call.method === 'groups.stop')).toBe(false)
    expect(loaded.chat.$groupChats.get().Stopping.hostedStatus?.state).toBe('stopping')
  })

  it.each(['send', 'stop'] as const)('does not let a stale replay overwrite a newer hosted %s', async action => {
    let exposeRoom = false
    let releaseLog: () => void = () => undefined
    let logStarted: () => void = () => undefined

    const logRequested = new Promise<void>(resolve => {
      logStarted = resolve
    })

    const heldLog = new Promise<Record<string, unknown>>(resolve => {
      releaseLog = () => resolve({ events: [], has_more: false, latest_seq: 0 })
    })

    const loaded = await loadRuntime(method => {
      if (method === 'groups.capabilities') {
        return {
          authority_gateway_id: 'install:home',
          driver: true,
          persistent_process: true
        }
      }

      if (method === 'groups.list') {
        return {
          rooms: exposeRoom
            ? [
                {
                  authority_epoch: 1,
                  authority_gateway_id: 'install:home',
                  disbanded_at: null,
                  latest_seq: 0,
                  members: MEMBERS,
                  name: 'Release',
                  revision: 1,
                  room_id: 'room-1'
                }
              ]
            : []
        }
      }

      if (method === 'groups.state') {
        return {
          driver_status: { working: false },
          room: {
            authority_epoch: 1,
            authority_gateway_id: 'install:home',
            disbanded_at: null,
            members: MEMBERS,
            name: 'Release',
            room_id: 'room-1'
          }
        }
      }

      if (method === 'groups.log') {
        logStarted()

        return heldLog
      }

      if (method === 'groups.send' || method === 'groups.stop') {
        return { ok: true }
      }

      throw new Error(`unexpected method: ${method}`)
    })

    loaded.chat.$groupChats.set({
      Release: room({
        hostedStatus: { label: 'Working', state: 'working' },
        running: true
      })
    })
    await loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)
    exposeRoom = true

    const refresh = loaded.runtime.refreshHostedRooms()
    await logRequested

    if (action === 'send') {
      await expect(loaded.rounds.sendToGroupChatDurably('Release', MEMBERS, 'Keep going')).resolves.toBeTruthy()
    } else {
      await loaded.rounds.stopGroupThread('Release', null, MEMBERS)
    }

    for (let attempt = 0; attempt < 10; attempt += 1) {
      await Promise.resolve()
    }

    const expectedState = loaded.chat.$groupChats.get().Release.hostedStatus?.state

    if (action === 'send') {
      expect(['queued', 'working']).toContain(expectedState)
    } else {
      expect(expectedState).toBe('stopped')
    }

    releaseLog()
    await refresh

    expect(loaded.chat.$groupChats.get().Release.hostedStatus?.state).toBe(expectedState)
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('continues a bounded partial replay on the next refresh and offers Retry while incomplete', async () => {
    const events = Array.from({ length: 21 }, (_, index) =>
      hostedEvent(index + 1, `message-${index + 1}`, 'message.user', {
        text: `Message ${index + 1}`,
        thread_id: 'thread-1'
      })
    )

    const loaded = await loadRuntime((method, params) => {
      if (method === 'groups.capabilities') {
        return {
          authority_gateway_id: 'install:home',
          driver: true,
          max_log_limit: 1,
          persistent_process: true
        }
      }

      if (method === 'groups.list') {
        return {
          rooms: [
            {
              authority_epoch: 1,
              authority_gateway_id: 'install:home',
              disbanded_at: null,
              latest_seq: events.length,
              members: MEMBERS,
              name: 'Release',
              revision: 1,
              room_id: 'room-1'
            }
          ]
        }
      }

      if (method === 'groups.state') {
        return {
          driver_status: { working: false },
          room: {
            authority_epoch: 1,
            authority_gateway_id: 'install:home',
            disbanded_at: null,
            members: MEMBERS,
            name: 'Release',
            room_id: 'room-1'
          }
        }
      }

      if (method === 'groups.log') {
        const since = Number(params.since_seq || 0)

        return {
          events: events.slice(since, since + 1),
          has_more: since + 1 < events.length,
          latest_seq: events.length
        }
      }

      throw new Error(`unexpected method: ${method}`)
    })

    loaded.chat.$groupChats.set({ Release: room() })
    await loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)

    expect(loaded.chat.$groupChats.get().Release).toMatchObject({
      hostedSeq: 20,
      hostedStatus: { canRetry: true },
      continuityIssue: 'Syncing recent activity…'
    })

    await loaded.runtime.refreshHostedRooms()

    expect(loaded.chat.$groupChats.get().Release).toMatchObject({
      hostedSeq: 21,
      continuityIssue: null
    })
    expect(loaded.calls.filter(call => call.method === 'groups.log')).toHaveLength(21)
    expect(loaded.calls.filter(call => call.method === 'groups.log').every(call => call.params.limit === 1)).toBe(true)
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('does not resurrect an idle room deleted while replay is in flight', async () => {
    let releaseLog: () => void = () => undefined
    let logStarted: () => void = () => undefined

    const logRequested = new Promise<void>(resolve => {
      logStarted = resolve
    })

    const heldLog = new Promise<Record<string, unknown>>(resolve => {
      releaseLog = () => resolve({ events: [], has_more: false, latest_seq: 0 })
    })

    const loaded = await loadRuntime(method => {
      if (method === 'groups.capabilities') {
        return { authority_gateway_id: 'install:home', driver: true, persistent_process: true }
      }

      if (method === 'groups.list') {
        return {
          rooms: [
            {
              authority_epoch: 1,
              authority_gateway_id: 'install:home',
              disbanded_at: null,
              members: MEMBERS,
              name: 'Release',
              room_id: 'room-1'
            }
          ]
        }
      }

      if (method === 'groups.state') {
        return {
          driver_status: { working: false },
          room: {
            authority_epoch: 1,
            authority_gateway_id: 'install:home',
            disbanded_at: null,
            members: MEMBERS,
            name: 'Release',
            room_id: 'room-1'
          }
        }
      }

      if (method === 'groups.log') {
        logStarted()

        return heldLog
      }

      throw new Error(`unexpected method: ${method}`)
    })

    loaded.chat.$groupChats.set({ Release: room({ running: false }) })
    const refresh = loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)
    await logRequested

    loaded.runtime.markHostedRoomLocallyDeleted('room-1')
    loaded.chat.$groupChats.set({})
    releaseLog()
    await refresh

    expect(loaded.chat.$groupChats.get().Release).toBeUndefined()
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('separates queued work from active work and fingerprints idle room state', async () => {
    const loaded = await loadRuntime(() => ({}))

    expect(
      loaded.runtime.hostedRoomDriverDisplayStatus({ kind: 'ready' }, { counts: { queued: 1 }, working: false })
    ).toMatchObject({ kind: 'queued', canStop: true })
    expect(loaded.runtime.hostedRoomPollFingerprint({ revision: 4, latest_seq: 9 })).toBe('4:9')
  })
})
