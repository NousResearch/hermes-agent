import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupChat from './group-chat'
import type * as groupRounds from './group-rounds'
import { pluginSdkMock, scriptedStorage } from './group-test-utils'
import type * as hostedRuntime from './hosted-room-runtime'
import type { GroupChat, GroupMember } from './types'

const { activation, host } = vi.hoisted(() => ({
  activation: { epoch: 0 },
  host: {} as Record<string, unknown>
}))

vi.mock('@hermes/plugin-sdk', async () => ({
  ...await pluginSdkMock(host),
  gatewayActivationEpoch: () => activation.epoch
}))

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
  source: { connectionId: string; gateway: string; profile: string }
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
  const source = { connectionId: 'gateway-a', gateway: 'open', profile: 'default' }
  activation.epoch = 0

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, {
    activeConnectionId: () => source.connectionId,
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
        get: () => source.connectionId,
        listen: () => () => undefined
      },
      gateway: {
        get: () => source.gateway,
        listen: () => () => undefined
      },
      profile: {
        get: () => source.profile,
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
    source,
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

  it('offers a retry when the peer gateway needed for reauthorization is unavailable', async () => {
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

    const loaded = await loadRuntime(method => {
      if (method === 'groups.capabilities') {
        return {
          authority_gateway_id: 'install:home',
          driver: true,
          features: ['peer_route_grant_fingerprint'],
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
            pending_actions: [{ kind: 'retry', task_id: 'uncertain-task' }],
            peer_routes: [{ member_id: 'builder', status: 'needs_reauthorization' }],
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
    })

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

    expect(loaded.chat.$groupChats.get().Release).toMatchObject({
      continuityIssue: 'Could not reconnect this Bot. Check that its device is online, then try again.',
      hostedStatus: {
        canRetry: true,
        canStop: false,
        label: 'Remote Builder needs your attention.',
        state: 'needs-attention'
      },
      running: false
    })
    expect(loaded.chat.$groupChats.get().Release.hostedStatus?.taskId).toBeUndefined()
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('keeps a shipped room across second launch and recovers an unsupported peer only after explicit check', async () => {
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

    const shippedMembers: GroupMember[] = [
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

    loaded.chat.$groupChats.set({
      Release: room({
        log: [
          {
            at: 1,
            from: { kind: 'user', name: 'You' },
            id: 'shipped-history-1',
            text: 'History survives the upgrade',
            thread: 'thread-1'
          }
        ],
        members: shippedMembers
      })
    })
    const storage = scriptedStorage(loaded.storage).storage

    await loaded.runtime.startHostedRoomRuntime(storage)
    expect(loaded.chat.$groupChats.get().Release).toMatchObject({
      continuityIssue: 'Update this device to keep this Group Chat running.',
      hostedStatus: { checkConnectionId: 'gateway-b', state: 'needs-attention' }
    })
    expect(stateCalls).toBe(1)

    const peerCapabilityCalls = () =>
      loaded.calls.filter(call => call.connectionId === 'gateway-b' && call.method === 'groups.capabilities').length

    expect(peerCapabilityCalls()).toBe(1)
    await loaded.runtime.refreshHostedRooms()
    expect(peerCapabilityCalls()).toBe(1)
    expect(stateCalls).toBe(1)

    await loaded.chat.persistGroupChatRooms()
    const serialized = structuredClone(await storage.get('group-chats', null)) as unknown as Record<string, GroupChat>
    expect(serialized.Release.peerProbeHint).toEqual({
      connectionId: 'gateway-b',
      installationId: 'install:peer',
      memberId: 'builder'
    })

    loaded.runtime.stopHostedRoomRuntime()
    loaded.chat.$groupChats.set({})
    loaded.runtime.$hostedRoomCapabilities.set({})
    loaded.chat.$groupChats.set(loaded.chat.hydrateGroupChatRooms(serialized))
    await loaded.runtime.startHostedRoomRuntime(storage)
    expect(loaded.chat.$groupChats.get().Release).toMatchObject({
      continuityIssue: 'Update this device to keep this Group Chat running.',
      hostedStatus: { checkConnectionId: 'gateway-b', state: 'needs-attention' }
    })
    expect(Object.keys(loaded.chat.$groupChats.get())).toEqual(['Release'])
    expect(loaded.chat.$groupChats.get().Release.log.map(entry => entry.text)).toContain('History survives the upgrade')
    expect(loaded.chat.$groupChats.get().Release.members).toHaveLength(2)
    expect(stateCalls).toBe(2)

    loaded.runtime.$hostedRoomOutbox.set({
      version: 1,
      commands: [{
        attempts: 0,
        authorityId: 'install:home',
        commandId: 'pending-other-room',
        connectionId: 'gateway-a',
        failureCode: null,
        kind: 'send',
        payload: { text: 'must stay pending' },
        roomId: 'room-other',
        status: 'pending'
      }]
    })
    loaded.runtime.$hostedRoomCleanup.set({
      version: 1,
      operations: [{
        armed: true,
        cancelId: 'pending-cleanup',
        connectionId: 'gateway-a',
        kind: 'home-disband',
        operationId: 'cleanup-other-room',
        ownerId: '',
        ownerLeaseUntil: 0,
        roomId: 'room-cleanup',
        setupId: 'setup-other-room'
      }]
    })
    const callsBeforeCheck = loaded.calls.length
    peerUpgraded = true
    const firstCheck = loaded.runtime.checkHostedRoomGateway('Release')
    const coalescedCheck = loaded.runtime.checkHostedRoomGateway('Release')
    expect(coalescedCheck).toBe(firstCheck)
    await expect(firstCheck).resolves.toBe(true)
    await vi.advanceTimersByTimeAsync(0)

    expect(stateCalls).toBe(3)
    expect(loaded.chat.$groupChats.get().Release).toMatchObject({
      continuityIssue: 'Reconnect Remote Builder to continue this Group Chat.',
      hostedStatus: {
        canReconnect: true,
        reconnectMemberId: 'builder',
        state: 'needs-attention'
      }
    })
    expect(loaded.chat.$groupChats.get().Release.hostedStatus?.checkConnectionId).toBeUndefined()
    expect(loaded.chat.$groupChats.get().Release.log.map(entry => entry.text)).toContain('History survives the upgrade')
    expect(loaded.chat.$groupChats.get().Release.members).toHaveLength(2)
    expect(loaded.calls.slice(callsBeforeCheck).filter(call => [
      'groups.create',
      'groups.disband',
      'groups.peer.revoke',
      'groups.peer.revoke_exact',
      'groups.rename',
      'groups.retry',
      'groups.send',
      'groups.stop',
      'prompt.submit',
      'session.create'
    ].includes(call.method))).toEqual([])
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('never authorizes a peer connection bound to the wrong installation and gives recovery guidance', async () => {
    const members = [
      { member_id: 'research', profile: 'research' },
      {
        display_name: 'Remote Builder',
        handle: 'builder',
        member_id: 'builder',
        profile: 'builder',
        target: { installation_id: 'install:peer', kind: 'peer', peer_id: 'install:peer' }
      }
    ]
    const loaded = await loadRuntime((method, _params, route) => {
      const connectionId = String(route?.connectionId || '')

      if (method === 'groups.capabilities') {
        return connectionId === 'gateway-b'
          ? {
              authority_gateway_id: 'install:other',
              driver: true,
              methods: ['groups.peer.revoke_exact'],
              persistent_process: true
            }
          : {
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
              rooms: [{
                authority_epoch: 1,
                authority_gateway_id: 'install:home',
                latest_seq: 0,
                members,
                name: 'Release',
                room_id: 'room-1'
              }]
            }
      }

      if (method === 'groups.state') {
        return {
          driver_status: { peer_routes: [{ member_id: 'builder', status: 'needs_reauthorization' }] },
          room: {
            authority_epoch: 1,
            authority_gateway_id: 'install:home',
            members,
            name: 'Release',
            room_id: 'room-1'
          }
        }
      }

      if (method === 'groups.log') {return { events: [], has_more: false, latest_seq: 0 }}
      throw new Error(`unexpected method: ${method}`)
    }, [
      { connectionId: 'gateway-a', mode: 'remote', profile: 'default', targetProfile: 'default' },
      { connectionId: 'gateway-b', mode: 'remote', profile: 'default', targetProfile: 'default' }
    ])

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

    expect(loaded.chat.$groupChats.get().Release.hostedStatus).toMatchObject({
      canReconnect: false,
      checkConnectionId: 'gateway-b',
      state: 'needs-attention'
    })
    expect(loaded.chat.$groupChats.get().Release.continuityIssue).toBe(
      'Reconnect Remote Builder from the device where this Bot is installed, then check again.'
    )
    expect(loaded.chat.$groupChats.get().Release.continuityIssue).not.toMatch(/update/i)
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('fences a manual check to its exact room, source activation, and runtime lifetime', async () => {
    const routes = [
      { connectionId: 'gateway-a', mode: 'remote', profile: 'default', targetProfile: 'default' },
      { connectionId: 'gateway-b', mode: 'remote', profile: 'default', targetProfile: 'default' }
    ]
    const members = [
      { member_id: 'research', profile: 'research' },
      {
        display_name: 'Remote Builder', handle: 'builder', member_id: 'builder', profile: 'builder',
        target: { installation_id: 'install:peer', kind: 'peer', peer_id: 'install:peer' }
      }
    ]
    const loaded = await loadRuntime((method, _params, route) => {
      const connectionId = String(route?.connectionId || '')

      if (method === 'groups.capabilities') {
        if (connectionId === 'gateway-b') {throw Object.assign(new Error('Method not found'), { code: -32601 })}

        return { authority_gateway_id: 'install:home', driver: true, features: ['peer_route_grant_fingerprint'], persistent_process: true }
      }

      if (method === 'groups.list') {return connectionId === 'gateway-b' ? { rooms: [] } : { rooms: [{ authority_epoch: 1, authority_gateway_id: 'install:home', latest_seq: 0, members, name: 'Release', room_id: 'room-1' }] }}
      if (method === 'groups.state') {return { driver_status: { peer_routes: [{ member_id: 'builder', status: 'needs_reauthorization' }] }, room: { authority_epoch: 1, authority_gateway_id: 'install:home', members, name: 'Release', room_id: 'room-1' } }}
      if (method === 'groups.log') {return { events: [], has_more: false, latest_seq: 0 }}
      throw new Error(`unexpected method: ${method}`)
    }, routes)
    const storage = scriptedStorage(loaded.storage).storage
    const original = room({
      members: [
        MEMBERS[0],
        { connectionId: 'gateway-b', handle: 'builder', name: 'builder', route: { connectionId: 'gateway-b', mode: 'remote', profile: 'builder', targetProfile: 'builder' }, sourceScoped: true, targetProfile: 'builder' }
      ]
    })
    loaded.chat.$groupChats.set({ Release: original })
    await loaded.runtime.startHostedRoomRuntime(storage)
    const checked = structuredClone(loaded.chat.$groupChats.get().Release)

    const heldCheck = async (move: () => void, restart = false) => {
      let entered!: () => void
      let release!: () => void
      const started = new Promise<void>(resolve => { entered = resolve })
      const held = new Promise<typeof routes>(resolve => { release = () => resolve(routes) })
      let calls = 0
      host.profileRoutes = () => {
        calls += 1

        if (calls === 1) {entered(); return held}

        return Promise.resolve(routes)
      }
      let before = loaded.calls.length
      const pending = loaded.runtime.checkHostedRoomGateway('Release')
      await started
      move()

      if (restart) {
        await loaded.runtime.startHostedRoomRuntime(storage)
        before = loaded.calls.length
      }

      release()
      await expect(pending).resolves.toBe(false)
      expect(loaded.calls.length).toBe(before)
    }

    const replacement = room({ roomId: 'room-replacement', hosted: 'install:replacement', hostedStatus: { checkConnectionId: 'gateway-b', label: 'Replacement', state: 'needs-attention' } })
    await heldCheck(() => loaded.chat.$groupChats.set({ Release: replacement }))
    expect(loaded.chat.$groupChats.get()).toEqual({ Release: replacement })

    loaded.chat.$groupChats.set({ Release: structuredClone(checked) })
    await heldCheck(() => { loaded.source.profile = 'other' })
    loaded.source.profile = 'default'

    await heldCheck(() => { loaded.source.connectionId = 'gateway-other' })
    loaded.source.connectionId = 'gateway-a'

    await heldCheck(() => { activation.epoch += 1 })

    await heldCheck(() => loaded.runtime.stopHostedRoomRuntime(), true)
    loaded.runtime.stopHostedRoomRuntime()
  })

  it('awaits one refresh-only successor when a manual check interrupts an in-flight poll', async () => {
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

    loaded.chat.$groupChats.set({
      Release: room({ hostedStatus: { checkConnectionId: 'gateway-a', label: 'Update required', state: 'unsupported' } })
    })
    const start = loaded.runtime.startHostedRoomRuntime(scriptedStorage(loaded.storage).storage)
    await stateRequested
    let settled = false
    const check = loaded.runtime.checkHostedRoomGateway('Release').finally(() => {settled = true})
    await Promise.resolve()
    expect(settled).toBe(false)
    releaseState()
    await Promise.all([start, check])

    expect(stateCalls).toBe(2)
    expect(loaded.chat.$groupChats.get().Release.hostedStatus?.state).toBe('ready')
    expect(loaded.chat.$groupChats.get().Release.hostedStatus?.checkConnectionId).toBeUndefined()
    loaded.runtime.stopHostedRoomRuntime()
  })
})
