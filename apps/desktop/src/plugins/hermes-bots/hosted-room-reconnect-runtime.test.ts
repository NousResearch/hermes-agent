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
  it('does not restart cleanup after stop wins a pending storage load', async () => {
    let releaseLoad: (value: unknown) => void = () => undefined
    let loadStarted: () => void = () => undefined
    const started = new Promise<void>(resolve => {
      loadStarted = resolve
    })
    const pending = new Promise<unknown>(resolve => {
      releaseLoad = resolve
    })
    const loaded = await loadRuntime(method => {
      throw new Error(`unexpected method after stop: ${method}`)
    })
    const get = vi.fn(async (key: string) => {
      if (key === 'hosted-room-outbox-v1') {
        loadStarted()

        return pending
      }

      return null
    })
    const storage = {
      get,
      set: vi.fn()
    }

    const start = loaded.runtime.startHostedRoomRuntime(storage as never)
    await started
    loaded.runtime.stopHostedRoomRuntime()
    releaseLoad(null)
    await start

    expect(get).toHaveBeenCalledTimes(1)
    expect(get).toHaveBeenCalledWith('hosted-room-outbox-v1', null)
    expect(loaded.calls).toEqual([])
  })

  it('does not let a pre-stop refresh rejection mark a restarted runtime offline', async () => {
    let releaseState: () => void = () => undefined
    let stateStarted: () => void = () => undefined
    const stateRequested = new Promise<void>(resolve => {
      stateStarted = resolve
    })
    const staleState = new Promise<Record<string, unknown>>((_resolve, reject) => {
      releaseState = () => reject(new Error('old connection closed'))
    })
    let stateCalls = 0
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
              latest_seq: 0,
              members: MEMBERS,
              name: 'Release',
              revision: 1,
              room_id: 'room-1'
            }
          ]
        }
      }

      if (method === 'groups.state') {
        stateCalls += 1
        if (stateCalls === 1) {
          stateStarted()

          return staleState
        }

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
        return { events: [], has_more: false, latest_seq: 0 }
      }

      throw new Error(`unexpected method: ${method}`)
    })
    const storage = scriptedStorage(loaded.storage).storage

    loaded.chat.$groupChats.set({ Release: room() })
    const firstStart = loaded.runtime.startHostedRoomRuntime(storage)
    await stateRequested
    loaded.runtime.stopHostedRoomRuntime()
    const secondStart = loaded.runtime.startHostedRoomRuntime(storage)
    releaseState()
    await Promise.all([firstStart, secondStart])

    expect(stateCalls).toBe(2)
    expect(loaded.chat.$groupChats.get().Release.hostedStatus?.state).toBe('ready')
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('surfaces an explicit reconnect action when a peer route needs reauthorization', async () => {
    const serverMembers = [
      {
        member_id: 'research',
        profile: 'research'
      },
      {
        display_name: 'Remote Builder',
        handle: 'builder',
        member_id: 'builder',
        profile: 'builder',
        target: {
          installation_id: 'install:peer',
          kind: 'peer',
          peer_id: 'install:peer'
        }
      }
    ]
    const loaded = await loadRuntime(
      (method, _params, route) => {
        const connectionId = String(route?.connectionId || '')

        if (method === 'groups.capabilities') {
          return {
            authority_gateway_id: connectionId === 'gateway-b' ? 'install:peer' : 'install:home',
            driver: true,
            features: connectionId === 'gateway-a' ? ['peer_route_grant_fingerprint'] : [],
            max_log_limit: 100,
            methods: connectionId === 'gateway-b' ? ['groups.peer.revoke_exact'] : [],
            persistent_process: true
          }
        }

        if (method === 'groups.list') {
          if (connectionId === 'gateway-b') {
            return { rooms: [] }
          }

          return {
            rooms: [
              {
                authority_epoch: 1,
                authority_gateway_id: 'install:home',
                disbanded_at: null,
                latest_seq: 0,
                members: serverMembers,
                name: 'Release',
                revision: 1,
                room_id: 'room-1'
              }
            ]
          }
        }

        if (method === 'groups.state') {
          return {
            driver_status: {
              blocked: true,
              peer_routes: [
                {
                  member_id: 'builder',
                  status: 'needs_reauthorization'
                }
              ],
              working: false
            },
            room: {
              authority_epoch: 1,
              authority_gateway_id: 'install:home',
              disbanded_at: null,
              members: serverMembers,
              name: 'Release',
              room_id: 'room-1'
            }
          }
        }

        if (method === 'groups.log') {
          return { events: [], has_more: false, latest_seq: 0 }
        }

        throw new Error(`unexpected method: ${method}`)
      },
      [
        { connectionId: 'gateway-a', mode: 'remote', profile: 'default', targetProfile: 'default' },
        { connectionId: 'gateway-b', mode: 'remote', profile: 'default', targetProfile: 'default' }
      ]
    )

    loaded.chat.$groupChats.set({ Release: room() })
    await loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)
    await loaded.runtime.refreshHostedRooms()

    expect(loaded.chat.$groupChats.get().Release).toMatchObject({
      continuityIssue: 'Reconnect Remote Builder to continue this Group Chat.',
      hostedStatus: {
        canReconnect: true,
        canRetry: false,
        canStop: false,
        label: 'Remote Builder needs your attention.',
        reconnectMemberId: 'builder',
        state: 'needs-attention'
      },
      running: false
    })
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('uses the stored member route to explain an older peer gateway without polling forever', async () => {
    let peerUpgraded = false
    let stateCalls = 0
    const serverMembers = [
      { member_id: 'research', profile: 'research' },
      {
        display_name: 'Remote Builder',
        handle: 'builder',
        member_id: 'builder',
        profile: 'builder',
        target: {
          installation_id: 'install:peer',
          kind: 'peer',
          peer_id: 'install:peer'
        }
      }
    ]
    const loaded = await loadRuntime(
      (method, _params, route) => {
        const connectionId = String(route?.connectionId || '')

        if (method === 'groups.capabilities') {
          if (connectionId === 'gateway-b') {
            if (!peerUpgraded) {
              throw Object.assign(new Error('Method not found'), { code: -32601 })
            }

            return {
              authority_gateway_id: 'install:peer',
              driver: true,
              methods: ['groups.peer.revoke_exact'],
              persistent_process: true
            }
          }

          return {
            authority_gateway_id: 'install:home',
            driver: true,
            features: ['peer_route_grant_fingerprint'],
            persistent_process: true
          }
        }

        if (method === 'groups.list') {
          return connectionId === 'gateway-b'
            ? { rooms: [] }
            : {
                rooms: [
                  {
                    authority_epoch: 1,
                    authority_gateway_id: 'install:home',
                    disbanded_at: null,
                    latest_seq: 0,
                    members: serverMembers,
                    name: 'Release',
                    revision: 1,
                    room_id: 'room-1'
                  }
                ]
              }
        }

        if (method === 'groups.state') {
          stateCalls += 1

          return {
            driver_status: {
              peer_routes: [{ member_id: 'builder', status: 'needs_reauthorization' }],
              working: false
            },
            room: {
              authority_epoch: 1,
              authority_gateway_id: 'install:home',
              members: serverMembers,
              name: 'Release',
              room_id: 'room-1'
            }
          }
        }

        if (method === 'groups.log') {
          return { events: [], has_more: false, latest_seq: 0 }
        }

        throw new Error(`unexpected method: ${method}`)
      },
      [
        { connectionId: 'gateway-a', mode: 'remote', profile: 'default', targetProfile: 'default' },
        { connectionId: 'gateway-b', mode: 'remote', profile: 'default', targetProfile: 'default' }
      ]
    )

    loaded.chat.$groupChats.set({
      Release: room({
        members: [
          MEMBERS[0],
          {
            connectionId: 'gateway-b',
            handle: 'builder',
            name: 'builder',
            route: {
              connectionId: 'gateway-b',
              mode: 'remote',
              profile: 'builder',
              targetProfile: 'builder'
            },
            sourceScoped: true,
            targetProfile: 'builder'
          }
        ]
      })
    })
    await loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)

    expect(loaded.chat.$groupChats.get().Release.continuityIssue).toMatch(/^Update /)
    expect(stateCalls).toBe(1)

    await loaded.runtime.refreshHostedRooms()
    expect(stateCalls).toBe(1)

    peerUpgraded = true
    vi.setSystemTime(new Date(Date.now() + 31_000))
    await loaded.runtime.refreshHostedRooms()

    expect(stateCalls).toBe(2)
    expect(loaded.chat.$groupChats.get().Release.continuityIssue).toBe(
      'Reconnect Remote Builder to continue this Group Chat.'
    )
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('does not let an in-flight poll restore a cache entry after invalidation', async () => {
    let releaseState: () => void = () => undefined
    let stateStarted: () => void = () => undefined
    const stateRequested = new Promise<void>(resolve => {
      stateStarted = resolve
    })
    const heldState = new Promise<Record<string, unknown>>(resolve => {
      releaseState = () =>
        resolve({
          driver_status: { working: true },
          room: {
            authority_epoch: 1,
            authority_gateway_id: 'install:home',
            disbanded_at: null,
            members: MEMBERS,
            name: 'Release',
            room_id: 'room-1'
          }
        })
    })
    let stateCalls = 0
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
              latest_seq: 0,
              members: MEMBERS,
              name: 'Release',
              revision: 1,
              room_id: 'room-1'
            }
          ]
        }
      }

      if (method === 'groups.state') {
        stateCalls += 1
        if (stateCalls === 1) {
          stateStarted()

          return heldState
        }

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
        return { events: [], has_more: false, latest_seq: 0 }
      }

      throw new Error(`unexpected method: ${method}`)
    })

    loaded.chat.$groupChats.set({ Release: room() })
    const start = loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)
    await stateRequested
    loaded.runtime.invalidateHostedRoomPoll('room-1')
    releaseState()
    await start

    expect(loaded.chat.$groupChats.get().Release.hostedStatus?.state).toBe('working')
    await loaded.runtime.refreshHostedRooms()
    expect(stateCalls).toBe(2)
    expect(loaded.chat.$groupChats.get().Release.hostedStatus?.state).toBe('ready')
    loaded.runtime.stopHostedRoomRuntime()
  })
})
