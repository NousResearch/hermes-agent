import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { pluginSdkMock, scriptedStorage } from './group-test-utils'

const { host } = vi.hoisted(() => ({
  host: {} as Record<string, unknown>
}))

const groupRounds = vi.hoisted(() => ({
  cancelGroupThreadForLeaseLoss: vi.fn(async (..._args: unknown[]) => undefined),
  sendToGroupChat: vi.fn((..._args: unknown[]): unknown => null),
  stopGroupThread: vi.fn(async (..._args: unknown[]) => undefined)
}))

vi.mock('@hermes/plugin-sdk', async () => pluginSdkMock(host))
vi.mock('./group-rounds', () => groupRounds)

async function loadRuntime() {
  vi.resetModules()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, {
    activeConnectionId: () => 'gateway-a',
    onEvent: vi.fn(() => () => undefined),
    profileRoutes: async () => [],
    request: vi.fn(async () => ({})),
    requestProfile: vi.fn(async () => ({})),
    retainProfileSocket: vi.fn(() => () => undefined),
    state: {
      connectionId: {
        get: () => 'gateway-a',
        listen: () => () => undefined
      }
    }
  })

  const [chat, data, runtime] = await Promise.all([
    import('./group-chat'),
    import('./data'),
    import('./desktop-room-command-runtime')
  ])

  return {
    chat,
    data,
    runtime
  }
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.clearAllMocks()
  vi.clearAllTimers()
  vi.useRealTimers()
})

describe('classic Group Chat command runtime', () => {
  it('mints one private authority and publishes only its hash', async () => {
    const loaded = await loadRuntime()
    const stored = new Map<string, unknown>()

    await loaded.runtime.startDesktopRoomCommandRuntime(scriptedStorage(stored).storage)
    const authority = await loaded.runtime.prepareDesktopRoomAuthority()

    expect(authority.desktopCoordinatorId).toMatch(/^desktop:/)
    expect(authority.desktopAuthorityToken).toMatch(/^authority:/)
    expect(authority.desktopAuthorityHash).toMatch(/^[a-f0-9]{64}$/)
    expect(authority.desktopAuthorityHash).not.toContain(authority.desktopAuthorityToken)
    expect(stored.get('desktop-room-command-consumer-v1')).toBe(authority.desktopCoordinatorId)

    loaded.runtime.stopDesktopRoomCommandRuntime()
  })

  it('does not cache a coordinator identity until storage confirms it', async () => {
    const loaded = await loadRuntime()
    const stored = new Map<string, unknown>()
    let failRead = true
    let failWrite = true
    const storage = {
      get: vi.fn(async (key: string) => {
        if (failRead) {
          throw new Error('read unavailable')
        }
        return stored.get(key) ?? null
      }),
      set: vi.fn(async (key: string, value: unknown) => {
        if (failWrite) {
          throw new Error('disk unavailable')
        }
        stored.set(key, structuredClone(value))
      })
    }

    await loaded.runtime.startDesktopRoomCommandRuntime(storage as never)
    await expect(loaded.runtime.prepareDesktopRoomAuthority()).rejects.toThrow('could not read')
    expect(storage.set).not.toHaveBeenCalled()

    failRead = false
    await expect(loaded.runtime.prepareDesktopRoomAuthority()).rejects.toThrow('disk unavailable')

    failWrite = false
    const authority = await loaded.runtime.prepareDesktopRoomAuthority()
    expect(stored.get('desktop-room-command-consumer-v1')).toBe(authority.desktopCoordinatorId)
    loaded.runtime.stopDesktopRoomCommandRuntime()
  })

  it('adopts only a classic room with local execution evidence', async () => {
    const loaded = await loadRuntime()
    const stored = new Map<string, unknown>()

    loaded.chat.$groupChats.set({
      Active: {
        log: [],
        sessions: {
          research: 'session-1'
        },
        watermarks: {}
      },
      Silent: {
        log: [],
        watermarks: {}
      },
      Hosted: {
        hosted: 'install:home',
        log: [],
        sessions: {
          research: 'session-2'
        },
        watermarks: {}
      }
    })

    await loaded.runtime.startDesktopRoomCommandRuntime(scriptedStorage(stored).storage)

    expect(loaded.chat.$groupChats.get().Active).toMatchObject({
      desktopAuthorityHash: expect.stringMatching(/^[a-f0-9]{64}$/),
      desktopAuthorityToken: expect.stringMatching(/^authority:/),
      desktopCoordinatorId: expect.stringMatching(/^desktop:/),
      roomId: expect.stringMatching(/^[a-z0-9-]+$/)
    })
    expect(loaded.chat.$groupChats.get().Silent.desktopCoordinatorId).toBeUndefined()
    expect(loaded.chat.$groupChats.get().Hosted.desktopCoordinatorId).toBeUndefined()

    loaded.runtime.stopDesktopRoomCommandRuntime()
  })

  it('lets healthy Bots continue when another Group Chat member is offline', async () => {
    const loaded = await loadRuntime()
    const stored = new Map<string, unknown>()

    const members = [
      { connectionId: 'gateway-a', name: 'online' },
      { connectionId: 'gateway-b', name: 'offline', sourceMissing: true }
    ]

    loaded.data.$lastRoster.set(members)
    loaded.chat.$groupChats.set({
      Planning: {
        log: [],
        members,
        roomId: 'room-1',
        sessions: {},
        watermarks: {}
      }
    })
    groupRounds.sendToGroupChat.mockImplementation((...args: unknown[]) => {
      const options = (args[5] || {}) as { entryId?: unknown }
      const room = loaded.chat.$groupChats.get().Planning
      loaded.chat.$groupChats.set({
        Planning: {
          ...room,
          desktopCommandSettled: {
            [String(options.entryId)]: Date.now()
          }
        }
      })

      return 'thread-1'
    })
    await loaded.runtime.startDesktopRoomCommandRuntime(scriptedStorage(stored).storage)

    const result = await loaded.runtime.executeDesktopRoomCommand(
      {
        action: 'send',
        command_id: 'messaging:send-1',
        payload: {
          message: 'Review the plan',
          recipients: members
        },
        room_id: 'room-1'
      },
      [{ authorityToken: 'authority:test', name: 'Planning', roomId: 'room-1' }],
      {
        consumerId: 'desktop:test',
        request: vi.fn(async () => ({})),
        route: { connectionId: 'gateway-a', mode: 'remote', profile: 'default', targetProfile: 'default' },
        signal: null
      }
    )

    expect(result).toEqual({ room_name: 'Planning', thread_id: 'thread-1' })
    expect(groupRounds.sendToGroupChat).toHaveBeenCalledWith(
      'Planning',
      expect.arrayContaining([
        expect.objectContaining({ name: 'online' }),
        expect.objectContaining({ name: 'offline' })
      ]),
      'Review the plan',
      null,
      undefined,
      expect.objectContaining({ entryId: 'messaging:send-1' })
    )
    loaded.runtime.stopDesktopRoomCommandRuntime()
  })

  it('settles a durable Stop after restart when its send was already superseded', async () => {
    const loaded = await loadRuntime()
    const stored = new Map<string, unknown>()
    const members = [{ connectionId: 'gateway-a', name: 'online' }]

    loaded.data.$lastRoster.set(members)
    loaded.chat.$groupChats.set({
      Planning: {
        log: [],
        members,
        roomId: 'room-1',
        sessions: {},
        watermarks: {}
      }
    })
    await loaded.runtime.startDesktopRoomCommandRuntime(scriptedStorage(stored).storage)

    const result = await loaded.runtime.executeDesktopRoomCommand(
      {
        action: 'stop',
        command_id: 'messaging:stop-1',
        payload: { target_command_id: 'messaging:send-1' },
        room_id: 'room-1',
        target_command_state: 'failed',
        target_result_code: 'superseded_by_stop'
      },
      [{ authorityToken: 'authority:test', name: 'Planning', roomId: 'room-1' }],
      {
        consumerId: 'desktop:test',
        request: vi.fn(async () => ({})),
        route: { connectionId: 'gateway-a', mode: 'remote', profile: 'default', targetProfile: 'default' },
        signal: null
      }
    )

    expect(result).toEqual({ room_name: 'Planning', stale: true, stopped: true })
    expect(groupRounds.stopGroupThread).not.toHaveBeenCalled()
    loaded.runtime.stopDesktopRoomCommandRuntime()
  })

  it('still aborts live work when the mailbox already marked its send superseded', async () => {
    const loaded = await loadRuntime()
    const stored = new Map<string, unknown>()
    const members = [{ connectionId: 'gateway-a', name: 'online' }]

    loaded.data.$lastRoster.set(members)
    loaded.chat.$groupChats.set({
      Planning: {
        log: [],
        members,
        roomId: 'room-1',
        sessions: {},
        watermarks: {}
      }
    })
    groupRounds.sendToGroupChat.mockImplementation((...args: unknown[]) => {
      const group = String(args[0] || '')
      const room = loaded.chat.$groupChats.get()[group]
      loaded.chat.$groupChats.set({
        [group]: {
          ...room,
          running: true
        }
      })

      return 'thread-1'
    })
    groupRounds.stopGroupThread.mockImplementation(async (...args: unknown[]) => {
      const group = String(args[0] || '')
      const room = loaded.chat.$groupChats.get()[group]
      loaded.chat.$groupChats.set({
        [group]: {
          ...room,
          running: false
        }
      })
    })
    await loaded.runtime.startDesktopRoomCommandRuntime(scriptedStorage(stored).storage)

    const context = {
      consumerId: 'desktop:test',
      request: vi.fn(async () => ({})),
      route: { connectionId: 'gateway-a', mode: 'remote' as const, profile: 'default', targetProfile: 'default' },
      signal: null
    }

    const descriptors = [{ authorityToken: 'authority:test', name: 'Planning', roomId: 'room-1' }]

    const send = loaded.runtime.executeDesktopRoomCommand(
      {
        action: 'send',
        command_id: 'messaging:send-1',
        payload: { message: 'Review the plan', recipients: members },
        room_id: 'room-1'
      },
      descriptors,
      context
    )

    await Promise.resolve()
    await Promise.resolve()

    const stopped = await loaded.runtime.executeDesktopRoomCommand(
      {
        action: 'stop',
        command_id: 'messaging:stop-1',
        payload: { target_command_id: 'messaging:send-1' },
        room_id: 'room-1',
        target_command_state: 'failed',
        target_result_code: 'superseded_by_stop'
      },
      descriptors,
      context
    )

    expect(stopped).toEqual({ room_name: 'Planning', stopped: true })
    expect(groupRounds.stopGroupThread).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(250)
    await expect(send).resolves.toEqual({ room_name: 'Planning', stopped: true })

    groupRounds.stopGroupThread.mockClear()
    loaded.chat.$groupChats.set({
      Planning: {
        ...loaded.chat.$groupChats.get().Planning,
        desktopCommandSettled: {},
        log: [
          {
            at: 1,
            from: { kind: 'user', name: 'You' },
            id: 'old-message',
            text: 'Earlier work',
            thread: 'thread-old'
          }
        ],
        running: false
      }
    })
    groupRounds.sendToGroupChat.mockImplementation(() => {
      const room = loaded.chat.$groupChats.get().Planning
      loaded.chat.$groupChats.set({ Planning: { ...room, running: true } })
      return 'thread-new'
    })
    groupRounds.stopGroupThread.mockResolvedValue(undefined)

    const laterSend = loaded.runtime.executeDesktopRoomCommand(
      {
        action: 'send',
        command_id: 'messaging:send-later',
        payload: { message: 'New work', recipients: members },
        room_id: 'room-1'
      },
      descriptors,
      context
    )
    await Promise.resolve()
    await Promise.resolve()
    const earlierStop = await loaded.runtime.executeDesktopRoomCommand(
      {
        action: 'stop',
        command_id: 'messaging:stop-earlier',
        payload: { target_message_id: 'old-message', target_thread_id: 'thread-old' },
        room_id: 'room-1'
      },
      descriptors,
      context
    )

    expect(earlierStop).toEqual({ room_name: 'Planning', stopped: true })
    expect(groupRounds.stopGroupThread).toHaveBeenCalledWith('Planning', 'thread-old', expect.any(Array))
    loaded.chat.updateGroupChat('Planning', current => ({
      ...current,
      desktopCommandSettled: { 'messaging:send-later': Date.now() },
      running: false
    }))
    await vi.advanceTimersByTimeAsync(250)
    await expect(laterSend).resolves.toEqual({ room_name: 'Planning', thread_id: 'thread-new' })

    groupRounds.cancelGroupThreadForLeaseLoss.mockClear()
    const abandonedSend = loaded.runtime.executeDesktopRoomCommand(
      {
        action: 'send',
        command_id: 'messaging:send-disposed',
        payload: { message: 'Work during reload', recipients: members },
        room_id: 'room-1'
      },
      descriptors,
      context
    )
    const abandonedExpectation = expect(abandonedSend).rejects.toThrow('moved to another Desktop')
    await Promise.resolve()
    await Promise.resolve()
    loaded.runtime.stopDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(250)
    await abandonedExpectation
    expect(groupRounds.cancelGroupThreadForLeaseLoss).toHaveBeenCalledWith('Planning', members)
  })

  it('does not re-drive terminal Stops or stop newer same-thread work', async () => {
    const loaded = await loadRuntime()
    const stored = new Map<string, unknown>()
    const members = [{ connectionId: 'gateway-a', name: 'online' }]

    loaded.data.$lastRoster.set(members)
    loaded.chat.$groupChats.set({
      Planning: {
        log: [
          {
            at: 2,
            from: { kind: 'user', name: 'You' },
            id: 'new-message',
            text: 'Newer work in the same thread',
            thread: 'thread-1'
          }
        ],
        members,
        roomId: 'room-1',
        sessions: {},
        watermarks: {}
      }
    })
    await loaded.runtime.startDesktopRoomCommandRuntime(scriptedStorage(stored).storage)
    const descriptors = [{ authorityToken: 'authority:test', name: 'Planning', roomId: 'room-1' }]
    const context = {
      consumerId: 'desktop:test',
      request: vi.fn(async () => ({})),
      route: {
        connectionId: 'gateway-a',
        mode: 'remote' as const,
        profile: 'default',
        targetProfile: 'default'
      },
      signal: null
    }

    const terminal = await loaded.runtime.executeDesktopRoomCommand(
      {
        action: 'stop',
        command_id: 'messaging:stop-terminal',
        payload: { target_command_id: 'messaging:send-complete' },
        room_id: 'room-1',
        target_command_state: 'completed'
      },
      descriptors,
      context
    )
    const stale = await loaded.runtime.executeDesktopRoomCommand(
      {
        action: 'stop',
        command_id: 'messaging:stop-stale',
        payload: { target_message_id: 'old-message', target_thread_id: 'thread-1' },
        room_id: 'room-1'
      },
      descriptors,
      context
    )

    expect(terminal).toEqual({ room_name: 'Planning', stale: true, stopped: false })
    expect(stale).toEqual({ room_name: 'Planning', stale: true, stopped: false })
    expect(groupRounds.stopGroupThread).not.toHaveBeenCalled()
    loaded.runtime.stopDesktopRoomCommandRuntime()
  })

  it('bounds repeated classic room execution after the final mailbox claim', async () => {
    const loaded = await loadRuntime()
    const stored = new Map<string, unknown>()
    const members = [{ connectionId: 'gateway-a', name: 'online' }]

    loaded.data.$lastRoster.set(members)
    loaded.chat.$groupChats.set({
      Planning: {
        log: [],
        members,
        roomId: 'room-1',
        sessions: {},
        watermarks: {}
      }
    })
    groupRounds.sendToGroupChat.mockImplementation(() => {
      const room = loaded.chat.$groupChats.get().Planning
      loaded.chat.$groupChats.set({ Planning: { ...room, running: false } })

      return 'thread-rejected'
    })
    await loaded.runtime.startDesktopRoomCommandRuntime(scriptedStorage(stored).storage)

    const execution = loaded.runtime.executeDesktopRoomCommand(
      {
        action: 'send',
        attempts: 2,
        command_id: 'messaging:send-rejected',
        payload: { message: 'Review the plan', recipients: members },
        room_id: 'room-1'
      },
      [{ authorityToken: 'authority:test', name: 'Planning', roomId: 'room-1' }],
      {
        consumerId: 'desktop:test',
        request: vi.fn(async () => ({})),
        route: {
          connectionId: 'gateway-a',
          mode: 'remote',
          profile: 'default',
          targetProfile: 'default'
        },
        signal: null
      }
    )
    const outcome = execution.then(
      () => null,
      error => error as Error & { retryable?: boolean }
    )

    await vi.advanceTimersByTimeAsync(250)
    const error = await outcome
    expect(error?.message).toContain('after repeated attempts')
    expect(error?.retryable).not.toBe(true)
    expect(groupRounds.sendToGroupChat).toHaveBeenCalledTimes(2)
    loaded.runtime.stopDesktopRoomCommandRuntime()
  })
})
