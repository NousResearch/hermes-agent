import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { desktopRoomDescriptors } from './desktop-room-command-client'
import type { DesktopRoomCommand } from './desktop-room-command-client'
import { executeDesktopRoomCommand, startDesktopRoomCommandRuntime, stopDesktopRoomCommandRuntime } from './desktop-room-command-runtime'
import { settleDesktopCommand } from './group-command-receipts'
import { classicAuthorityHash } from './group-desktop-authority'
import type { GroupChat, ProfileRoute } from './types'

// Only the runtime, classifier, command client and receipt/fence helpers are real.
// No room engine, backend, socket, model, filesystem or Stop implementation loads.
const mocks = vi.hoisted(() => ({
  state: { connectionId: 'gateway-a', profile: 'default', gateway: 'open', epoch: 1 },
  rooms: {} as Record<string, GroupChat>,
  request: vi.fn(),
  routes: vi.fn(),
  persist: vi.fn(),
  retain: vi.fn(() => () => undefined),
  onEvent: vi.fn(),
  send: vi.fn(),
  stop: vi.fn(),
  cancel: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', () => ({
  gatewayActivationEpoch: () => mocks.state.epoch,
  host: {
    state: {
      connectionId: { get: () => mocks.state.connectionId },
      profile: { get: () => mocks.state.profile },
      gateway: { get: () => mocks.state.gateway }
    },
    activeConnectionId: () => mocks.state.connectionId,
    profileRoutes: mocks.routes,
    requestProfile: mocks.request,
    retainProfileSocket: mocks.retain,
    onEvent: mocks.onEvent
  }
}))
vi.mock('./group-chat', () => ({
  $groupChats: { get: () => mocks.rooms },
  backfillClassicGroupAuthorities: vi.fn(),
  GROUP_CHAT_MAX_MEMBERS: 8,
  groupChatHostedGateway: (room: GroupChat) => room.hosted,
  persistGroupChatRoomsRequired: mocks.persist,
  scheduleGroupChatServerSync: vi.fn(),
  updateGroupChat: (name: string, update: (room: GroupChat) => GroupChat) => { mocks.rooms[name] = update(mocks.rooms[name]) }
}))
vi.mock('./data', () => ({
  $botMeta: { get: () => ({}) },
  $lastRoster: { get: () => [{ name: 'reviewer' }] },
  cachedUnionRoster: () => null
}))
vi.mock('./group-membership', () => ({
  groupChatMemberBots: () => [{ name: 'reviewer' }],
  groupChatBotsFromDescriptors: () => [{ name: 'reviewer' }]
}))
vi.mock('./group-rounds', () => ({
  sendToGroupChat: mocks.send,
  stopGroupThread: mocks.stop,
  cancelGroupThreadForLeaseLoss: mocks.cancel
}))
vi.mock('./hosted-room-runtime', () => ({ groupChatContinuityReady: () => true }))

const legacy = { driver: false, persistent_process: false }
const canonical = { driver: true, persistent_process: true }
const route: ProfileRoute = { connectionId: 'gateway-a', profile: 'default', targetProfile: 'default', mode: 'remote' }

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(res => { resolve = res })

  return { promise, resolve }
}

function command(action: 'send' | 'stop'): DesktopRoomCommand {
  return {
    action,
    command_id: `command-${action}`,
    lease_token: `lease-${action}`,
    room_id: 'room-1',
    payload: action === 'send'
      ? { message: 'Only an inert dispatch', recipients: [{ name: 'reviewer' }] }
      : { target_thread_id: 'thread-old', target_message_id: 'message-old' }
  }
}

function execute(action: 'send' | 'stop', sourceRoute = { ...route }) {
  return executeDesktopRoomCommand(command(action), desktopRoomDescriptors(mocks.rooms), {
    consumerId: 'inert-consumer', request: vi.fn(), route: sourceRoute, signal: null
  })
}

function serve(capabilities: unknown = legacy) {
  const claimed = new Set<string>()
  mocks.request.mockImplementation(async (_route: ProfileRoute, method: string, params: Record<string, unknown>) => {
    if (method === 'groups.capabilities') {
      if (capabilities instanceof Error) { throw capabilities }

      return capabilities
    }

    if (method === 'groups.desktop.claim') {
      const action = (params.actions as Array<'send' | 'stop'>)[0]
      const key = `${_route.connectionId}:${action}`

      if (claimed.has(key)) { return { commands: [] } }
      claimed.add(key)

      return { commands: [command(action)] }
    }

    return {}
  })
}

function expectNoRendererControl() {
  expect(mocks.send).not.toHaveBeenCalled()
  expect(mocks.stop).not.toHaveBeenCalled()
  expect(mocks.cancel).not.toHaveBeenCalled()
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.resetAllMocks()
  Object.assign(mocks.state, { connectionId: 'gateway-a', profile: 'default', gateway: 'open', epoch: 1 })
  mocks.rooms = {
    Planning: {
      desktopAuthorityHash: classicAuthorityHash('authority:inert'),
      desktopAuthorityToken: 'authority:inert',
      roomId: 'room-1',
      members: [{ name: 'reviewer' }],
      log: [{ id: 'message-old', at: 1, from: { kind: 'user', name: 'You' }, text: 'Retained history', thread: 'thread-old' }],
      watermarks: {}
    }
  }
  mocks.routes.mockResolvedValue([{ ...route }])
  mocks.persist.mockResolvedValue(undefined)
  mocks.retain.mockReturnValue(() => undefined)
  mocks.onEvent.mockReturnValue(() => undefined)
  mocks.stop.mockResolvedValue(undefined)
  mocks.send.mockImplementation((_group, _members, _message, _thread, _reply, options) => {
    const room = mocks.rooms.Planning
    room.desktopCommandSettled = settleDesktopCommand('Planning', room, options.entryId, 'send', {
      room_name: 'Planning', thread_id: 'thread-inert'
    })

    return 'thread-inert'
  })
  serve()
})

afterEach(() => {
  stopDesktopRoomCommandRuntime()
  vi.clearAllTimers()
  vi.useRealTimers()
})

describe('non-UI legacy command capability gate (inert)', () => {
  it.each([
    ['canonical', canonical],
    ['unavailable', { driver: false, persistent_process: true }],
    ['driver false alone', { driver: false }],
    ['canonical ownership while stopped', { ...legacy, authority_gateway_id: 'owner' }],
    ['canonical method while stopped', { ...legacy, methods: ['groups.create'] }],
    ['malformed', { ...legacy, features: [false] }],
    ['empty response', null],
    ['failed read', new Error('offline')],
    ['TypeError is not legacy', new TypeError('unsupported')]
  ])('plugin-start, push and periodic discovery refuse %s', async (_label, capability) => {
    serve(capability)
    const history = structuredClone(mocks.rooms.Planning.log)
    await startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(0)
    const pending = mocks.onEvent.mock.calls.find(([name]) => name === 'desktop_rooms.commands.pending')?.[1]
    pending?.({ connectionId: 'gateway-a' })
    await vi.advanceTimersByTimeAsync(60_000)

    expect(mocks.request.mock.calls.filter(([, method]) => String(method).startsWith('groups.desktop.'))).toEqual([])
    expectNoRendererControl()
    expect(mocks.retain).not.toHaveBeenCalled()
    expect(mocks.rooms.Planning.log).toEqual(history)
  })

  it('retains positive legacy dispatch without borrowing evidence from another connection', async () => {
    const underlying = mocks.request.getMockImplementation()!
    mocks.routes.mockResolvedValue([{ ...route }, { ...route, connectionId: 'gateway-b' }])
    mocks.request.mockImplementation((r, method, params) => method === 'groups.capabilities' && r.connectionId === 'gateway-a'
      ? Promise.resolve(canonical) : underlying(r, method, params))
    await startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(0)

    expect(mocks.send).toHaveBeenCalledTimes(1)
    expect(mocks.stop).toHaveBeenCalledTimes(1)
    const control = mocks.request.mock.calls.filter(([, method]) => String(method).startsWith('groups.desktop.'))
    expect(control.length).toBeGreaterThan(0)
    expect(control.every(([r]) => r.connectionId === 'gateway-b')).toBe(true)
    expect(mocks.request).toHaveBeenCalledWith({ ...route, connectionId: 'gateway-b' }, 'groups.capabilities', { profile: 'default' })
  })

  it.each(['connection', 'profile', 'activation', 'socket'] as const)('discards late discovery evidence after %s changes', async change => {
    const pending = deferred<unknown>()
    const underlying = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation((r, method, params) => method === 'groups.capabilities'
      ? pending.promise : underlying(r, method, params))
    await startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(0)

    if (change === 'connection') { mocks.state.connectionId = 'gateway-b' }

    if (change === 'profile') { mocks.state.profile = 'other' }

    if (change === 'activation') { mocks.state.epoch += 1 }

    if (change === 'socket') { mocks.state.gateway = 'closed' }
    pending.resolve(legacy)
    await vi.advanceTimersByTimeAsync(0)

    expect(mocks.request.mock.calls.filter(([, method]) => String(method).startsWith('groups.desktop.'))).toEqual([])
    expectNoRendererControl()
  })

  it('rechecks source after persistence before presence or claim', async () => {
    // Initial startup persistence is allowed; the next room readback invalidates
    // the capability epoch before either publication or claim can be sent.
    let writes = 0
    mocks.persist.mockImplementation(async () => {
      writes += 1

      if (writes > 1) { mocks.state.epoch += 1 }
    })
    await startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(0)

    expect(mocks.request.mock.calls.filter(([, method]) => String(method).startsWith('groups.desktop.'))).toEqual([])
    expectNoRendererControl()
  })

  it('does not use a late claim after activation switches away and back', async () => {
    const pending = deferred<unknown>()
    const underlying = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation((r, method, params) => method === 'groups.desktop.claim'
      ? pending.promise : underlying(r, method, params))
    await startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(0)
    mocks.state.epoch += 2
    pending.resolve({ commands: [command('send'), command('stop')] })
    await vi.advanceTimersByTimeAsync(0)

    expectNoRendererControl()
    expect(mocks.request.mock.calls.filter(([, method]) => method === 'groups.desktop.complete')).toEqual([])
  })

  it.each(['send', 'stop'] as const)('checks current capabilities at direct %s execution, not only discovery', async action => {
    serve(canonical)
    await expect(execute(action)).rejects.toThrow()
    expectNoRendererControl()
  })

  it.each(['send', 'stop'] as const)('refuses stale direct %s execution after its awaited readback', async action => {
    mocks.persist.mockImplementation(async () => { mocks.state.epoch += 1 })
    await expect(execute(action)).rejects.toThrow()
    expectNoRendererControl()
  })

  it('does not use legacy discovery as permission after capabilities become canonical before execution', async () => {
    const underlying = mocks.request.getMockImplementation()!
    let claimed = false
    mocks.request.mockImplementation((r, method, params) => {
      if (method === 'groups.capabilities' && claimed) { return Promise.resolve(canonical) }

      if (method === 'groups.desktop.claim') { claimed = true }

      return underlying(r, method, params)
    })
    await startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(0)
    expectNoRendererControl()
  })

  it('reclassifies after persistence instead of publishing or claiming with old legacy evidence', async () => {
    let writes = 0
    mocks.persist.mockImplementation(async () => {
      writes += 1

      if (writes > 1) { serve(canonical) }
    })
    await startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(0)

    expect(mocks.request.mock.calls.filter(([, method]) => String(method).startsWith('groups.desktop.'))).toHaveLength(0)
    expectNoRendererControl()
  })

  it.each(['send', 'stop'] as const)('rechecks at the last %s boundary after an earlier legacy read', async action => {
    const pending = deferred<unknown>()
    let reads = 0
    mocks.request.mockImplementation(async (_r, method) => {
      if (method !== 'groups.capabilities') { return {} }
      reads += 1

      return reads === 1 ? legacy : pending.promise
    })
    const result = execute(action)
    const rejection = expect(result).rejects.toThrow()
    await vi.advanceTimersByTimeAsync(0)
    expectNoRendererControl()
    pending.resolve(canonical)
    await rejection
    expectNoRendererControl()
  })

  it('does not carry a late route-inventory read across an activation', async () => {
    const pending = deferred<ProfileRoute[]>()
    mocks.routes.mockReturnValue(pending.promise)
    await startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(0)
    mocks.state.epoch += 1
    pending.resolve([{ ...route }])
    await vi.advanceTimersByTimeAsync(0)
    expect(mocks.request).not.toHaveBeenCalled()
    expectNoRendererControl()
  })

  it.each(['connectionId', 'profile', 'targetProfile'] as const)('binds direct execution to the exact %s read', async field => {
    const pending = deferred<unknown>()
    mocks.request.mockReturnValue(pending.promise)
    const mutableRoute = { ...route }
    const result = execute('send', mutableRoute)
    const rejection = expect(result).rejects.toThrow()
    await vi.advanceTimersByTimeAsync(0)
    mutableRoute[field] = 'replaced'
    pending.resolve(legacy)
    await rejection
    expectNoRendererControl()
  })

  it('keeps exact retained receipts readable without replaying unknown settled commands', async () => {
    const room = mocks.rooms.Planning
    const result = { room_name: 'Planning', thread_id: 'accepted-thread' }
    room.desktopCommandSettled = settleDesktopCommand('Planning', room, 'command-send', 'send', result)
    serve(canonical)
    await expect(execute('send')).resolves.toEqual(result)
    room.desktopCommandSettled = { 'command-send': 1 }
    await expect(execute('send')).rejects.toThrow('saved result cannot be recovered')
    expectNoRendererControl()
  })
})
