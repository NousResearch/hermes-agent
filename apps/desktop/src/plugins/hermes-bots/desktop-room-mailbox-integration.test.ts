import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as clientModule from './desktop-room-command-client'
import type * as runtimeModule from './desktop-room-command-runtime'
import type * as groupChat from './group-chat'
import type * as groupRounds from './group-rounds'
import { createGroupGateway, scriptedStorage } from './group-test-utils'
import type { GatewayOptions, ScriptedGateway } from './group-test-utils'
import type { GroupChat, GroupMember, ProfileRoute } from './types'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

interface QueuedCommand extends clientModule.DesktopRoomCommand {
  action: 'send' | 'stop'
  command_id: string
  created: number
  payload: Record<string, unknown>
  room_id: string
  state: 'claimed' | 'completed' | 'failed' | 'pending'
  leaseOwner?: string
  leaseUntil?: number
}

class ControlledMailbox {
  clock = 1_000
  completeFailures = 0
  failClaims = 0
  readonly calls: Array<{ method: string; params: Record<string, unknown> }> = []
  readonly commands = new Map<string, QueuedCommand>()
  readonly owners = new Map<string, { consumer: string; until: number }>()
  readonly tokens = new Map<string, string>()

  queue(command: Omit<QueuedCommand, 'created' | 'state'>) {
    this.commands.set(command.command_id, { ...command, created: this.clock, state: 'pending' })
  }

  advance(milliseconds: number) {
    this.clock += milliseconds
  }

  expire(commandId: string) {
    const command = this.commands.get(commandId)

    if (command) {
      command.leaseUntil = this.clock - 1
    }
  }

  async request(method: string, params: Record<string, unknown>) {
    this.calls.push({ method, params: structuredClone(params) })

    if (method === 'groups.desktop.presence') {
      return { room_ids: this.own(params) }
    }

    if (method === 'groups.desktop.claim') {
      if (this.failClaims-- > 0) {
        throw new Error('temporary route failure')
      }

      const owned = new Set(this.own(params))
      const actions = new Set(Array.isArray(params.actions) ? params.actions.map(String) : ['send', 'stop'])
      const limit = Math.max(1, Math.min(8, Number(params.limit || 8)))

      const available = [...this.commands.values()]
        .filter(
          command =>
            owned.has(command.room_id) &&
            actions.has(command.action) &&
            (command.state === 'pending' ||
              (command.state === 'claimed' && Number(command.leaseUntil || 0) <= this.clock))
        )
        .sort(
          (left, right) =>
            Number(right.action === 'stop') - Number(left.action === 'stop') || left.created - right.created
        )
        .slice(0, limit)

      for (const command of available) {
        command.state = 'claimed'
        command.leaseOwner = String(params.consumer_id)
        command.lease_token = `lease:${command.command_id}:${this.clock}`
        command.leaseUntil = this.clock + 45_000
        command.attempts = Number(command.attempts || 0) + 1
      }

      return { commands: structuredClone(available) }
    }

    const command = this.commands.get(String(params.command_id || ''))

    if (!command || command.leaseOwner !== params.consumer_id || command.lease_token !== params.lease_token) {
      throw new Error('command lease is no longer owned by this Desktop')
    }

    if (method === 'groups.desktop.renew') {
      if (Number(command.leaseUntil || 0) <= this.clock) {
        throw new Error('command lease expired')
      }

      command.leaseUntil = this.clock + 45_000

      return { command: structuredClone(command) }
    }

    if (method === 'groups.desktop.complete') {
      if (this.completeFailures-- > 0) {
        throw new Error('completion transport failed')
      }

      if (Number(command.leaseUntil || 0) <= this.clock) {
        throw new Error('command lease expired')
      }

      command.state = params.success === true ? 'completed' : 'failed'
      command.leaseOwner = undefined
      command.lease_token = undefined
      command.leaseUntil = undefined

      return { command: structuredClone(command) }
    }

    throw new Error(`unexpected RPC ${method}`)
  }

  private own(params: Record<string, unknown>) {
    const consumer = String(params.consumer_id || '')
    const owned: string[] = []

    for (const raw of Array.isArray(params.room_authorities) ? params.room_authorities : []) {
      const authority = raw as Record<string, unknown>
      const roomId = String(authority.room_id || '')
      const token = String(authority.authority_token || '')
      const expected = this.tokens.get(roomId)
      const current = this.owners.get(roomId)

      if (!expected || expected !== token) {
        continue
      }

      if (current && current.consumer !== consumer && current.until > this.clock) {
        continue
      }

      this.owners.set(roomId, { consumer, until: this.clock + 90_000 })
      owned.push(roomId)
    }

    return owned
  }
}

interface Loaded {
  chat: typeof groupChat
  client: typeof clientModule
  gateway: ScriptedGateway
  mailbox: ControlledMailbox
  rounds: typeof groupRounds
  runtime: typeof runtimeModule
}

const route: ProfileRoute = {
  connectionId: 'gateway-a',
  mode: 'remote',
  profile: 'default',
  targetProfile: 'default'
}

const members: GroupMember[] = [{ name: 'reviewer', title: 'Reviewer' }]

async function load(options: GatewayOptions = {}): Promise<Loaded> {
  vi.resetModules()
  const gateway = createGroupGateway(options)
  const mailbox = new ControlledMailbox()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const gatewayRequest = host.requestProfile as (
    route: ProfileRoute,
    method: string,
    params: Record<string, unknown>
  ) => Promise<unknown>

  host.profileRoutes = async () => [route]
  host.requestProfile = async (target: ProfileRoute, method: string, params: Record<string, unknown>) =>
    method.startsWith('groups.desktop.') ? mailbox.request(method, params) : gatewayRequest(target, method, params)
  host.onEvent = vi.fn(() => () => undefined)
  host.retainProfileSocket = vi.fn(() => () => undefined)

  const [chat, client, rounds, runtime, data, shared] = await Promise.all([
    import('./group-chat'),
    import('./desktop-room-command-client'),
    import('./group-rounds'),
    import('./desktop-room-command-runtime'),
    import('./data'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))
  data.$lastRoster.set(members)

  return { chat, client, gateway, mailbox, rounds, runtime }
}

function seed(loaded: Loaded, name = 'Workshop', roomId = 'room-1') {
  loaded.chat.$groupChats.set({
    [name]: {
      log: [],
      members,
      roomId,
      sessions: {},
      watermarks: {}
    }
  })
  loaded.chat.backfillClassicGroupAuthorities([name])
  const room = loaded.chat.$groupChats.get()[name]
  loaded.mailbox.tokens.set(roomId, String(room.desktopAuthorityToken))

  return room
}

function runCycle(loaded: Loaded, consumerId: string, actions?: string[]) {
  return loaded.client.runDesktopRoomCommandCycle({
    actions,
    consumerId,
    execute: loaded.runtime.executeDesktopRoomCommand,
    request: (_route, method, params) => loaded.mailbox.request(method, params),
    rooms: loaded.chat.$groupChats.get(),
    routes: [route]
  })
}

async function settle<T>(promise: Promise<T>, limit = 120) {
  let settled = false
  void promise.finally(() => {
    settled = true
  })

  for (let index = 0; index < limit && !settled; index += 1) {
    await vi.advanceTimersByTimeAsync(250)
  }

  return promise
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(async () => {
  const runtime = await import('./desktop-room-command-runtime')
  runtime.stopDesktopRoomCommandRuntime()
  vi.clearAllTimers()
  vi.useRealTimers()
})

describe('classic Desktop mailbox with the real room engine', () => {
  it('keeps cold-start sends pending until discovery completes while Stop remains available', async () => {
    const loaded = await load({ turn: () => 'Ready after discovery' })
    seed(loaded)
    const remoteMembers: GroupMember[] = [{ name: 'reviewer', connectionId: 'gateway-a', remoteSource: true }]
    loaded.chat.updateGroupChat('Workshop', current => ({ ...current, members: remoteMembers }))
    const data = await import('./data')
    data.$lastRoster.set(remoteMembers)
    loaded.mailbox.queue({
      action: 'send', command_id: 'cold:send', room_id: 'room-1',
      payload: { message: '@reviewer Reply once', recipients: remoteMembers }
    })
    const request = host.requestProfile as (route: ProfileRoute, method: string, params: Record<string, unknown>) => Promise<unknown>

    host.requestProfile = (route: ProfileRoute, method: string, params: Record<string, unknown>) => {
      if (method === 'groups.capabilities') {
        return Promise.reject(Object.assign(new Error('older gateway'), { code: -32601 }))
      }

      return request(route, method, params)
    }

    const storage = scriptedStorage(loaded.gateway.storage).storage
    await loaded.runtime.startDesktopRoomCommandRuntime(storage)
    await vi.advanceTimersByTimeAsync(150_000)
    expect(loaded.mailbox.commands.get('cold:send')).toMatchObject({ state: 'pending' })
    expect(loaded.mailbox.commands.get('cold:send')?.attempts || 0).toBe(0)
    expect(loaded.gateway.rpcFor('prompt.submit')).toHaveLength(0)
    expect(loaded.mailbox.calls.some(call => call.method === 'groups.desktop.claim' &&
      (call.params.actions as string[])?.includes('stop'))).toBe(true)

    const hosted = await import('./hosted-room-runtime')

    try {
      await hosted.startHostedRoomRuntime(storage)
      expect(hosted.groupChatContinuityReady(loaded.chat.$groupChats.get().Workshop)).toBe(true)
      await loaded.runtime.startDesktopRoomCommandRuntime(storage)
      await vi.advanceTimersByTimeAsync(10_000)
      expect(loaded.mailbox.commands.get('cold:send')).toMatchObject({ state: 'completed', attempts: 1 })
      expect(loaded.gateway.rpcFor('prompt.submit')).toHaveLength(1)
    } finally {
      hosted.stopHostedRoomRuntime()
      loaded.runtime.stopDesktopRoomCommandRuntime()
    }
  })

  const reviewState = (loaded: Loaded) => structuredClone(Object.fromEntries(
    Object.entries(loaded.chat.$groupChats.get()).map(([name, room]) => [name, {
      log: room.log, epoch: room.epoch, running: room.running, turn: room.turn,
      sessions: room.sessions, holds: room.holds, settled: room.desktopCommandSettled,
      tombstone: room.tombstone, token: room.desktopAuthorityToken
    }])
  ))

  it.each(['disband', 'removed', 'replacement'] as const)(
    'delayed session creation cannot mutate %s room',
    async kind => {
      const loaded = await load()
      loaded.chat.updateGroupChat('Legacy', () => ({ log: [], members, sessions: {}, watermarks: {} }), { sync: false })
      loaded.mailbox.tokens.set('name:Legacy', String(loaded.chat.$groupChats.get().Legacy.desktopAuthorityToken))
      const originalRequest = host.requestProfile as (target: ProfileRoute, method: string, params: Record<string, unknown>) => Promise<unknown>
      const originalLocalRequest = host.request as (method: string, params: Record<string, unknown>) => Promise<unknown>
      let release!: () => void
      let created = false
      const blocked = new Promise<void>(resolve => { release = resolve })

      host.request = async (method: string, params: Record<string, unknown>) => {
        const result = await originalLocalRequest(method, params)

        if (method === 'session.create') { created = true; await blocked }

        return result
      }

      host.requestProfile = async (target: ProfileRoute, method: string, params: Record<string, unknown>) => {
        const result = await originalRequest(target, method, params)

        if (method === 'session.create') {
          created = true
          await blocked
        }

        return result
      }

      loaded.mailbox.queue({ action: 'send', command_id: 'review:delayed-create', room_id: 'name:Legacy',
        payload: { message: 'Obsolete input', recipients: members } })
      const work = runCycle(loaded, 'desktop:one', ['send'])
      await vi.advanceTimersByTimeAsync(250)
      expect(created).toBe(true)
      expect(loaded.gateway.rpcFor('prompt.submit')).toHaveLength(0)

      if (kind === 'disband') {
        await (await import('./group-chat-view')).disbandGroupChat('Legacy', members)
      } else if (kind === 'removed') {
        loaded.chat.$groupChats.set({})
      } else {
        const { classicAuthorityHash } = await import('./group-desktop-authority')
        loaded.chat.$groupChats.set({ Legacy: {
          ...loaded.chat.$groupChats.get().Legacy, log: [], sessions: { reviewer: 'replacement-session' },
          desktopAuthorityToken: 'authority:replacement', desktopAuthorityHash: classicAuthorityHash('authority:replacement')
        } })
      }

      await vi.advanceTimersByTimeAsync(500)
      const snapshot = reviewState(loaded)
      release()
      await settle(work)
      await vi.advanceTimersByTimeAsync(100)
      expect(loaded.gateway.rpcFor('prompt.submit')).toHaveLength(0)
      expect(reviewState(loaded)).toEqual(snapshot)
    }
  )

  it.each(['removed', 'hosted', 'replacement', 'conflict', 'duplicate'] as const)(
    'Stop waiting for an earlier send cannot mutate %s authority',
    async kind => {
      const loaded = await load()
      const original = seed(loaded)
      loaded.mailbox.queue({ action: 'stop', command_id: 'review:waiting-stop', room_id: 'room-1',
        payload: { target_command_id: 'review:pending-send' } })
      const work = runCycle(loaded, 'desktop:one', ['stop'])
      await vi.advanceTimersByTimeAsync(100)
      const { classicAuthorityHash } = await import('./group-desktop-authority')

      const next = kind === 'removed' ? {} : { Workshop: {
        ...original,
        ...(kind === 'hosted' ? { hosted: 'gateway-new' } : {}),
        ...(kind === 'conflict' ? { desktopAuthorityConflict: true } : {}),
        ...(kind === 'replacement' ? {
          desktopAuthorityToken: 'authority:replacement', desktopAuthorityHash: classicAuthorityHash('authority:replacement')
        } : {})
      }, ...(kind === 'duplicate' ? { Duplicate: { ...original } } : {}) }

      loaded.chat.$groupChats.set(next)
      const snapshot = reviewState(loaded)
      const outcome = await settle(work)
      expect(outcome[0]?.success).toBe(false)
      expect(loaded.gateway.rpcFor('session.interrupt')).toHaveLength(0)
      expect(reviewState(loaded)).toEqual(snapshot)
    }
  )

  it('actual disband and same-name recreation retain the new session binding', async () => {
    const loaded = await load()
    seed(loaded, 'Workshop', 'old-room')
    const originalRequest = host.request as (method: string, params: Record<string, unknown>) => Promise<unknown>
    let release!: () => void
    let created = false
    const blocked = new Promise<void>(resolve => { release = resolve })

    host.request = async (method: string, params: Record<string, unknown>) => {
      const result = await originalRequest(method, params)

      if (method === 'session.create' && params.title === 'Group: old-room') {
        created = true
        await blocked
      }

      return result
    }

    loaded.mailbox.queue({ action: 'send', command_id: 'review:recreated', room_id: 'old-room',
      payload: { message: 'Obsolete input', recipients: members } })
    const work = runCycle(loaded, 'desktop:one', ['send'])
    await vi.advanceTimersByTimeAsync(250)
    expect(created).toBe(true)
    await (await import('./group-chat-view')).disbandGroupChat('Workshop', members)
    await vi.advanceTimersByTimeAsync(1_000)

    const replacementSession = await originalRequest('session.create', {
      profile: 'reviewer', title: 'Group: new-room', hidden: true
    }) as { stored_session_id: string }

    loaded.chat.updateGroupChat('Workshop', () => ({
      roomId: 'new-room', log: [], members, sessions: { reviewer: replacementSession.stored_session_id },
      watermarks: {}, running: false, epoch: 0
    }))
    const replacementToken = loaded.chat.$groupChats.get().Workshop.desktopAuthorityToken
    release()
    await settle(work)
    await vi.advanceTimersByTimeAsync(100)
    expect(loaded.gateway.rpcFor('prompt.submit')).toHaveLength(0)
    expect(loaded.chat.$groupChats.get().Workshop.desktopAuthorityToken).toBe(replacementToken)
    const bound = loaded.chat.$groupChats.get().Workshop.sessions?.reviewer
    const persisted = (loaded.gateway.storage.get('group-chats') as Record<string, GroupChat>).Workshop.sessions?.reviewer
    loaded.rounds.sendToGroupChat('Workshop', members, 'New room input')
    await vi.advanceTimersByTimeAsync(5_000)
    expect({ bound, persisted, submittedTitle: loaded.gateway.calls[0]?.title }).toEqual({
      bound: replacementSession.stored_session_id, persisted: replacementSession.stored_session_id,
      submittedTitle: 'Group: new-room'
    })
  })

  it('lease abort cannot mutate or interrupt a same-epoch replacement', async () => {
    const loaded = await load({ pollsBusy: 8 })
    seed(loaded)
    const controller = new AbortController()

    const work = loaded.runtime.executeDesktopRoomCommand({ action: 'send', command_id: 'review:rotation',
      room_id: 'room-1', payload: { message: 'Original input', recipients: members } },
      loaded.client.desktopRoomDescriptors(loaded.chat.$groupChats.get()),
      { consumerId: 'desktop:one', request: async () => ({}), route, signal: controller.signal })
      .then(result => ({ result }), error => ({ error }))

    await vi.advanceTimersByTimeAsync(250)
    expect(loaded.gateway.rpcFor('prompt.submit')).toHaveLength(1)
    const { classicAuthorityHash } = await import('./group-desktop-authority')
    loaded.chat.$groupChats.set({ Workshop: { ...loaded.chat.$groupChats.get().Workshop,
      desktopAuthorityToken: 'authority:replacement', desktopAuthorityHash: classicAuthorityHash('authority:replacement') } })
    const snapshot = reviewState(loaded)
    controller.abort('lease-lost')
    await settle(work)
    await vi.advanceTimersByTimeAsync(10_000)
    expect(loaded.gateway.rpcFor('session.interrupt')).toHaveLength(0)
    expect(reviewState(loaded)).toEqual(snapshot)
  })

  it('replacement during Stop interrupt cannot receive old receipts', async () => {
    const loaded = await load({ pollsBusy: 8 })
    seed(loaded)
    loaded.mailbox.queue({ action: 'send', command_id: 'review:send-stop', room_id: 'room-1',
      payload: { message: 'Original input', recipients: members } })
    const send = runCycle(loaded, 'desktop:one', ['send'])
    await vi.advanceTimersByTimeAsync(250)
    const originalRequest = host.requestProfile as (target: ProfileRoute, method: string, params: Record<string, unknown>) => Promise<unknown>
    const originalLocalRequest = host.request as (method: string, params: Record<string, unknown>) => Promise<unknown>
    let release!: () => void
    let interrupted = false
    const blocked = new Promise<void>(resolve => { release = resolve })

    host.request = async (method: string, params: Record<string, unknown>) => {
      const result = await originalLocalRequest(method, params)

      if (method === 'session.interrupt') { interrupted = true; await blocked }

      return result
    }

    host.requestProfile = async (target: ProfileRoute, method: string, params: Record<string, unknown>) => {
      const result = await originalRequest(target, method, params)

      if (method === 'session.interrupt') { interrupted = true; await blocked }

      return result
    }

    loaded.mailbox.queue({ action: 'stop', command_id: 'review:stop', room_id: 'room-1',
      payload: { target_command_id: 'review:send-stop' } })
    const stop = runCycle(loaded, 'desktop:one', ['stop'])
    await vi.advanceTimersByTimeAsync(100)
    expect(interrupted).toBe(true)
    const { classicAuthorityHash } = await import('./group-desktop-authority')
    loaded.chat.$groupChats.set({ Workshop: { ...loaded.chat.$groupChats.get().Workshop,
      desktopAuthorityToken: 'authority:replacement', desktopAuthorityHash: classicAuthorityHash('authority:replacement'),
      desktopCommandSettled: {}, holds: {}, running: true, epoch: 20, log: [] } })
    const snapshot = reviewState(loaded)
    release()
    await settle(Promise.all([send, stop]))
    await vi.advanceTimersByTimeAsync(10_000)
    expect(reviewState(loaded)).toEqual(snapshot)
    expect(loaded.mailbox.commands.get('review:stop')?.state).toBe('failed')
  })

  it.each([true, false])('does not resume disbanded in-flight work, legacy=%s', async legacy => {
    const loaded = await load({ pollsBusy: 8, turn: () => 'late legacy reply' })
    loaded.chat.updateGroupChat('Legacy', () => ({
      log: [], members, sessions: {}, watermarks: {},
      ...(legacy ? {} : { roomId: 'room-stable' })
    }), { sync: false })
    const roomId = legacy ? 'name:Legacy' : 'room-stable'
    loaded.mailbox.tokens.set(roomId, String(loaded.chat.$groupChats.get().Legacy.desktopAuthorityToken))
    loaded.mailbox.queue({
      action: 'send', command_id: 'legacy:inflight', room_id: roomId,
      payload: { message: 'Original legacy work', recipients: members }
    })
    const work = runCycle(loaded, 'desktop:one', ['send'])
    await vi.advanceTimersByTimeAsync(250)
    const before = loaded.gateway.rpcFor('prompt.submit').length
    expect(before).toBe(1)
    const view = await import('./group-chat-view')
    await view.disbandGroupChat('Legacy', members)

    for (let index = 0; index < 12; index += 1) {
      await vi.advanceTimersByTimeAsync(5_000)
    }

    const after = loaded.chat.$groupChats.get().Legacy
    expect(loaded.gateway.rpcFor('prompt.submit')).toHaveLength(before)
    expect(after?.log || []).toEqual([])
    expect(Boolean(after?.running)).toBe(false)
    expect((await work)[0]?.success).toBe(false)
  })

  it.each(['removed', 'hosted', 'replacement', 'conflict', 'duplicate'] as const)(
    'invalidates an active legacy command when its room becomes %s',
    async kind => {
      const loaded = await load({ pollsBusy: 8, turn: () => 'obsolete reply' })
      loaded.chat.updateGroupChat('Legacy', () => ({ log: [], members, sessions: {}, watermarks: {} }), { sync: false })
      loaded.mailbox.tokens.set('name:Legacy', String(loaded.chat.$groupChats.get().Legacy.desktopAuthorityToken))
      loaded.mailbox.queue({
        action: 'send', command_id: 'legacy:obsolete', room_id: 'name:Legacy',
        payload: { message: 'Original work', recipients: members }
      })
      const work = runCycle(loaded, 'desktop:one', ['send'])
      await vi.advanceTimersByTimeAsync(250)
      expect(loaded.gateway.rpcFor('prompt.submit')).toHaveLength(1)
      const original = loaded.chat.$groupChats.get().Legacy
      const { classicAuthorityHash } = await import('./group-desktop-authority')

      const next: Record<string, GroupChat> = kind === 'removed' ? {} : {
        Legacy: {
          ...original,
          ...(kind === 'hosted' ? { hosted: 'gateway-new' } : {}),
          ...(kind === 'conflict' ? { desktopAuthorityConflict: true } : {}),
          ...(kind === 'replacement' ? {
            log: [],
            desktopAuthorityToken: 'authority:replacement',
            desktopAuthorityHash: classicAuthorityHash('authority:replacement')
          } : {})
        }
      }

      if (kind === 'duplicate') {
        next.Duplicate = { ...original, roomId: 'name:Legacy' }
      }

      loaded.chat.$groupChats.set(next)

      const executionState = (rooms: Record<string, groupChat.GroupChatRoom>) => Object.fromEntries(
        Object.entries(rooms).map(([name, room]) => [name, {
          log: room.log, epoch: room.epoch, running: room.running, turn: room.turn,
          sessions: room.sessions, holds: room.holds, settled: room.desktopCommandSettled
        }])
      )

      const snapshot = structuredClone(executionState(next))

      for (let index = 0; index < 12; index += 1) {
        await vi.advanceTimersByTimeAsync(5_000)
      }

      expect(loaded.gateway.rpcFor('prompt.submit')).toHaveLength(1)
      expect(executionState(loaded.chat.$groupChats.get())).toEqual(snapshot)
      expect((await work)[0]?.success).toBe(false)
    }
  )

  it.each(['removed', 'token', 'duplicate'] as const)(
    'filters a %s descriptor without suppressing healthy and legacy rooms',
    async kind => {
      const loaded = await load({ turn: () => '(pass)' })
      seed(loaded)

      for (const name of ['Legacy', 'Stale']) {
        loaded.chat.updateGroupChat(name, () => ({ log: [], members, watermarks: {} }), { sync: false })
        loaded.mailbox.tokens.set(`name:${name}`, String(loaded.chat.$groupChats.get()[name].desktopAuthorityToken))
      }

      // The send pump captures rooms before retaining sockets; mutate one advertised descriptor here.
      host.retainProfileSocket = () => {
        const current = loaded.chat.$groupChats.get()

        if (kind === 'removed') {
          const { Stale: _stale, ...rest } = current
          loaded.chat.$groupChats.set(rest)
        } else if (kind === 'token') {
          loaded.chat.$groupChats.set({
            ...current,
            Stale: { ...current.Stale, desktopAuthorityToken: 'authority:changed' }
          })
        } else {
          loaded.chat.$groupChats.set({ ...current, Duplicate: { ...current.Stale, roomId: 'name:Stale' } })
        }

        return () => undefined
      }

      for (const roomId of ['room-1', 'name:Legacy', 'name:Stale']) {
        loaded.mailbox.queue({
          action: 'send',
          command_id: `${roomId}:send`,
          room_id: roomId,
          payload: { message: 'Only valid rooms run', recipients: members }
        })
      }

      await loaded.runtime.startDesktopRoomCommandRuntime()
      await vi.advanceTimersByTimeAsync(10_000)
      expect(loaded.mailbox.commands.get('room-1:send')?.state).toBe('completed')
      expect(loaded.mailbox.commands.get('name:Legacy:send')?.state).toBe('completed')
      expect(loaded.mailbox.commands.get('name:Stale:send')?.state).toBe('pending')

      const advertised = loaded.mailbox.calls
        .filter(call => ['groups.desktop.claim', 'groups.desktop.presence'].includes(call.method))
        .flatMap(call => call.params.room_authorities as Array<{ room_id: string }>)

      expect(advertised.some(item => item.room_id === 'name:Stale')).toBe(false)
    }
  )

  it('does not request a claim when every captured descriptor becomes invalid', async () => {
    const loaded = await load()
    seed(loaded)

    host.retainProfileSocket = () => {
      loaded.chat.$groupChats.set({})

      return () => undefined
    }

    await loaded.runtime.startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(1_000)
    expect(loaded.mailbox.calls.filter(call => call.method === 'groups.desktop.claim')).toEqual([])
  })

  // Ported from the independent re-review patch, with legacy execution checked too.
  it('a legacy name-keyed room cannot suppress healthy room claims', async () => {
    const loaded = await load({ turn: () => '(pass)' })
    seed(loaded)
    loaded.chat.updateGroupChat('Legacy', () => ({ log: [], members, sessions: {}, watermarks: {} }), { sync: false })
    const legacy = loaded.chat.$groupChats.get().Legacy
    expect(legacy.roomId).toBeUndefined()
    loaded.mailbox.tokens.set('name:Legacy', String(legacy.desktopAuthorityToken))

    for (const roomId of ['room-1', 'name:Legacy']) {
      loaded.mailbox.queue({
        action: 'send',
        command_id: `${roomId}:send`,
        room_id: roomId,
        payload: { message: 'Healthy room work', recipients: members }
      })
    }

    await loaded.runtime.startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(10_000)
    expect(loaded.mailbox.commands.get('room-1:send')?.state).toBe('completed')
    expect(loaded.mailbox.commands.get('name:Legacy:send')?.state).toBe('completed')
    expect(loaded.gateway.rpcFor('prompt.submit')).toHaveLength(2)
  })

  // Real completion, lost ACK, bounded log eviction, and reclaimed delivery.
  it('a retained receipt prevents replay after the input leaves the bounded log', async () => {
    const loaded = await load({ turn: () => '(pass)' })
    seed(loaded)
    loaded.mailbox.completeFailures = 2
    loaded.mailbox.queue({
      action: 'send',
      command_id: 'telegram:already-settled',
      room_id: 'room-1',
      payload: { actor_display_name: 'David via Telegram', message: 'Old completed work', recipients: members }
    })
    await settle(runCycle(loaded, 'desktop:one', ['send']))
    const submissions = loaded.gateway.rpcFor('prompt.submit').length
    expect(loaded.mailbox.commands.get('telegram:already-settled')?.state).toBe('claimed')
    const originalResult = loaded.mailbox.calls.find(call => call.method === 'groups.desktop.complete')!.params.result

    for (let index = 0; index < 96; index += 1) {
      loaded.chat.appendGroupChatEntry(
        'Workshop',
        { kind: 'user', name: 'You' },
        `Later conversation ${index}`,
        'later-thread',
        undefined,
        `later:${String(index).padStart(3, '0')}`
      )
    }

    const before = structuredClone(loaded.chat.$groupChats.get().Workshop)
    expect(before.desktopCommandSettled?.['telegram:already-settled']).toBeTruthy()
    expect(before.log.some(entry => entry.id === 'telegram:already-settled')).toBe(false)
    await loaded.chat.persistGroupChatRoomsRequired()

    const replayThread = loaded.rounds.sendToGroupChat('Workshop', members, 'Old completed work', null, undefined, {
      entryId: 'telegram:already-settled',
      userName: 'David via Telegram'
    })

    expect(replayThread).toBe((originalResult as { thread_id: string }).thread_id)
    expect(loaded.chat.$groupChats.get().Workshop.log).toEqual(before.log)
    const projection = JSON.stringify(loaded.chat.groupChatSyncSnapshot())
    expect(projection).not.toContain(String(before.desktopAuthorityToken))
    expect(projection).not.toContain('desktopCommandSettled')
    loaded.mailbox.advance(46_000)
    loaded.mailbox.expire('telegram:already-settled')
    const result = await settle(runCycle(loaded, 'desktop:one', ['send']))
    await vi.advanceTimersByTimeAsync(10_000)
    const current = loaded.chat.$groupChats.get().Workshop
    expect(result[0].success).toBe(true)
    expect(current.log).toEqual(before.log)
    expect({
      running: Boolean(current.running),
      epoch: current.epoch,
      submissions: loaded.gateway.rpcFor('prompt.submit').length - submissions
    }).toEqual({ running: false, epoch: before.epoch, submissions: 0 })
    expect(
      loaded.mailbox.calls.filter(call => call.method === 'groups.desktop.complete').at(-1)!.params.result
    ).toEqual(originalResult)
  })

  it.each(['send', 'stop'] as const)(
    'replays the exact %s completion after restart, rename and trimming',
    async action => {
      let loaded = await load({ turn: () => '(pass)' })
      seed(loaded)
      loaded.chat.appendGroupChatEntry(
        'Workshop',
        { kind: 'user', name: 'You' },
        'Existing',
        'thread-existing',
        undefined,
        'existing'
      )

      const command = {
        action,
        command_id: `restart:${action}`,
        room_id: 'room-1',
        payload:
          action === 'send'
            ? { message: 'Do this once', recipients: members }
            : { target_message_id: 'existing', target_thread_id: 'thread-existing' }
      }

      loaded.mailbox.completeFailures = 2
      loaded.mailbox.queue(command)
      await settle(runCycle(loaded, 'desktop:before', [action]))
      const result = loaded.mailbox.calls.find(call => call.method === 'groups.desktop.complete')!.params.result
      expect(result).toBeTruthy()
      loaded.chat.updateGroupChat('Workshop', room => ({ ...room, log: [] }), { sync: false })
      await loaded.chat.persistGroupChatRoomsRequired()
      const saved = JSON.parse(JSON.stringify(loaded.gateway.storage.get('group-chats'))) as Record<string, GroupChat>
      loaded.runtime.stopDesktopRoomCommandRuntime()
      vi.clearAllTimers()
      loaded = await load({
        turn: () => {
          throw new Error('A replay must not dispatch')
        }
      })
      loaded.chat.$groupChats.set({ Renamed: saved.Workshop })
      loaded.mailbox.tokens.set('room-1', String(saved.Workshop.desktopAuthorityToken))
      loaded.mailbox.queue(command)
      const replay = await settle(runCycle(loaded, 'desktop:after', [action]))
      expect(replay[0].success).toBe(true)
      expect(loaded.mailbox.calls.find(call => call.method === 'groups.desktop.complete')!.params.result).toEqual(
        result
      )
      expect(loaded.chat.$groupChats.get().Renamed.log).toEqual([])
      expect(loaded.chat.$groupChats.get().Renamed.running).toBeFalsy()
      expect(loaded.gateway.rpcFor('prompt.submit')).toEqual([])
    }
  )

  it('replays a stale Stop result exactly even after the target becomes current', async () => {
    const loaded = await load()
    seed(loaded)
    loaded.mailbox.queue({
      action: 'stop',
      command_id: 'stop:stale',
      room_id: 'room-1',
      payload: { target_thread_id: 'old-thread', target_message_id: 'old-message' }
    })
    loaded.mailbox.completeFailures = 2
    await settle(runCycle(loaded, 'desktop:one', ['stop']))
    const originalResult = loaded.mailbox.calls.find(call => call.method === 'groups.desktop.complete')!.params.result
    expect(originalResult).toEqual({ room_name: 'Workshop', stale: true, stopped: false })
    loaded.chat.appendGroupChatEntry(
      'Workshop',
      { kind: 'user', name: 'You' },
      'Now matches',
      'old-thread',
      undefined,
      'old-message'
    )
    loaded.chat.updateGroupChat('Workshop', room => ({ ...room, running: true }), { sync: false })
    const before = structuredClone(loaded.chat.$groupChats.get().Workshop)
    loaded.mailbox.advance(46_000)
    loaded.mailbox.expire('stop:stale')
    await settle(runCycle(loaded, 'desktop:one', ['stop']))
    expect(
      loaded.mailbox.calls.filter(call => call.method === 'groups.desktop.complete').at(-1)!.params.result
    ).toEqual(originalResult)
    expect(loaded.chat.$groupChats.get().Workshop.running).toBe(true)
    expect(loaded.chat.$groupChats.get().Workshop.epoch).toBe(before.epoch)
    expect(loaded.chat.$groupChats.get().Workshop.holds).toEqual(before.holds)
  })

  it('fails closed on old timestamp-only settlements without reinserting trimmed input', async () => {
    const loaded = await load()
    seed(loaded)
    loaded.chat.updateGroupChat(
      'Workshop',
      room => ({ ...room, desktopCommandSettled: { 'old:send': 123 }, log: [] }),
      { sync: false }
    )
    loaded.mailbox.queue({
      action: 'send',
      command_id: 'old:send',
      room_id: 'room-1',
      payload: { message: 'Old completed input', recipients: members }
    })
    const outcomes = await settle(runCycle(loaded, 'desktop:one', ['send']))
    expect(outcomes[0].success).toBe(false)
    expect(loaded.mailbox.commands.get('old:send')?.state).toBe('failed')
    expect(loaded.chat.$groupChats.get().Workshop.log).toEqual([])
    expect(loaded.chat.$groupChats.get().Workshop.running).toBeFalsy()
    expect(loaded.gateway.rpcFor('prompt.submit')).toEqual([])
  })

  it('cannot revive an old in-flight claim after a stop and restart', async () => {
    const loaded = await load()
    seed(loaded)

    const normalRequest = host.requestProfile as (
      target: ProfileRoute,
      method: string,
      params: Record<string, unknown>
    ) => Promise<unknown>

    let release!: (value: unknown) => void

    const delayed = new Promise(resolve => {
      release = resolve
    })

    let delayedOnce = false
    const consumers: string[] = []

    host.requestProfile = (target: ProfileRoute, method: string, params: Record<string, unknown>) => {
      if (method === 'groups.desktop.claim' && (params.actions as string[]).includes('send')) {
        consumers.push(String(params.consumer_id))

        if (!delayedOnce) {
          delayedOnce = true

          return delayed
        }
      }

      return normalRequest(target, method, params)
    }

    await loaded.runtime.startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(0)
    expect(consumers).toHaveLength(1)
    loaded.runtime.stopDesktopRoomCommandRuntime()
    await loaded.runtime.startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(0)
    release({
      commands: [
        {
          action: 'send',
          command_id: 'telegram:stale-runtime',
          room_id: 'room-1',
          lease_token: 'lease:stale',
          payload: { message: 'Old claim', recipients: members }
        }
      ]
    })
    await vi.advanceTimersByTimeAsync(10_000)
    expect(loaded.gateway.rpcFor('prompt.submit')).toHaveLength(0)
    expect(new Set(consumers).size).toBe(2)
    expect(loaded.chat.$groupChats.get().Workshop.log).toEqual([])
  })

  it('does not start a delayed mailbox drive under a newer user thread epoch', async () => {
    const loaded = await load()
    seed(loaded)
    loaded.chat.updateGroupChat('Workshop', room => ({ ...room, running: true }), { sync: false })
    const { beginGroupCommandFence, releaseGroupCommandFence } = await import('./group-command-fence')
    const fence = beginGroupCommandFence('room-1', 'telegram:delayed')

    try {
      loaded.rounds.sendToGroupChat('Workshop', members, 'Old mailbox input', null, undefined, {
        entryId: 'telegram:delayed',
        commandFence: fence
      })
      loaded.rounds.sendToGroupChat('Workshop', members, 'New user thread')
      await vi.advanceTimersByTimeAsync(10_000)
      expect(loaded.gateway.calls).toHaveLength(1)
      expect(loaded.gateway.calls[0].prompt).toContain('New user thread')
      expect(loaded.gateway.calls[0].prompt).not.toContain('Old mailbox input')
      expect(loaded.chat.$groupChats.get().Workshop.desktopCommandSettled?.['telegram:delayed']).toBeUndefined()
    } finally {
      releaseGroupCommandFence(fence)
    }
  })

  it.each(['cancel', 'expire'])(
    'fences the post-turn commit after %s even if the member returns a reply',
    async how => {
      const loaded = await load()
      seed(loaded)
      const turns = await import('./group-turns')

      const turn = vi
        .spyOn(turns, 'runGroupChatMemberTurn')
        .mockImplementation(async (_group, _member, _prompt, _thread, _images, fence) => {
          if (how === 'cancel') {
            loaded.rounds.cancelGroupThreadForLeaseLoss('Workshop', members, fence!)
          } else {
            // A sleeping renderer may resume promise continuations before timers.
            vi.setSystemTime(Date.now() + 46_000)
          }

          return 'POST_TURN_LATE_REPLY'
        })

      loaded.mailbox.queue({
        action: 'send',
        command_id: 'telegram:post-turn',
        room_id: 'room-1',
        payload: { message: 'Review', recipients: members }
      })

      try {
        expect(await settle(runCycle(loaded, 'desktop:one', ['send']))).toEqual([
          expect.objectContaining({ retryable: true })
        ])
        const room = loaded.chat.$groupChats.get().Workshop
        expect(room.log.map(entry => entry.text)).not.toContain('POST_TURN_LATE_REPLY')
        expect(room.watermarks).toEqual({})
        expect(room.desktopCommandSettled?.['telegram:post-turn']).toBeUndefined()
        expect(room.holds || {}).toEqual({})
      } finally {
        turn.mockRestore()
      }
    }
  )

  it('does not stop or overwrite a newer user thread when the older mailbox lease is lost', async () => {
    const loaded = await load()
    const room = seed(loaded)
    const controller = new AbortController()
    let release!: (value: string) => void

    const late = new Promise<string>(resolve => {
      release = resolve
    })

    const turns = await import('./group-turns')

    const turn = vi
      .spyOn(turns, 'runGroupChatMemberTurn')
      .mockImplementation(async (_group, _member, _prompt, _thread, _images, fence) => (fence ? late : '(pass)'))

    const result = loaded.runtime.executeDesktopRoomCommand(
      {
        action: 'send',
        command_id: 'telegram:old',
        room_id: 'room-1',
        payload: { message: 'Old work', recipients: members }
      },
      loaded.client.desktopRoomDescriptors({ Workshop: room }),
      { consumerId: 'desktop:one', request: async () => ({}), route, signal: controller.signal }
    )

    const outcome = result.catch(error => error as { retryable: boolean })

    try {
      await vi.advanceTimersByTimeAsync(0)
      expect(turn).toHaveBeenCalledTimes(1)
      const thread = loaded.rounds.sendToGroupChat('Workshop', members, 'New unrelated work')
      const newEpoch = loaded.chat.$groupChats.get().Workshop.epoch
      controller.abort('lease-lost')
      release('OLD_REPLY')
      await vi.advanceTimersByTimeAsync(1_000)
      expect(await outcome).toMatchObject({ retryable: true })
      expect(loaded.chat.$groupChats.get().Workshop.epoch).toBe(newEpoch)
      expect(loaded.chat.$groupChats.get().Workshop.log.map(entry => entry.text)).not.toContain('OLD_REPLY')
      expect(loaded.chat.$groupChats.get().Workshop.log.some(entry => entry.thread === thread)).toBe(true)
      expect(loaded.chat.$groupChats.get().Workshop.holds || {}).toEqual({})
      expect(loaded.gateway.rpcFor('session.interrupt')).toHaveLength(0)
      expect(turn).toHaveBeenCalledTimes(2)
    } finally {
      turn.mockRestore()
    }
  })

  it.each(['send', 'stop'] as const)(
    'does not acknowledge %s when its receipt write silently fails, then replays safely',
    async action => {
      const loaded = await load({ turn: () => '(pass)' })
      seed(loaded)
      loaded.chat.updateGroupChat(
        'Workshop',
        room => ({
          ...room,
          log: [{ at: 1, id: 'user:1', thread: 'thread:1', from: { kind: 'user', name: 'You' }, text: 'Review' }]
        }),
        { sync: false }
      )
      const shared = await import('./shared')
      const storage = scriptedStorage(loaded.gateway.storage)
      const workingSet = storage.storage.set

      storage.storage.set = (_key, value) => {
        const snapshot = value as Record<string, GroupChat>

        if (!Object.keys(snapshot.Workshop?.desktopCommandSettled || {}).length) {
          workingSet(_key, value)
        }
      }

      shared.setPluginCtx(storage)
      loaded.mailbox.queue({
        action,
        command_id: 'telegram:receipt',
        room_id: 'room-1',
        payload:
          action === 'send'
            ? { message: 'Review', recipients: members }
            : { target_message_id: 'user:1', target_thread_id: 'thread:1' }
      })

      const result = await settle(runCycle(loaded, 'desktop:one', [action]))
      expect(result).toEqual([expect.objectContaining({ success: false, retryable: true })])
      expect(loaded.mailbox.calls.some(call => call.method === 'groups.desktop.complete')).toBe(false)
      const submissions = loaded.gateway.rpcFor('prompt.submit').length
      const persisted = loaded.gateway.storage.get('group-chats') as Record<string, GroupChat>
      expect(persisted.Workshop.desktopCommandSettled?.['telegram:receipt']).toBeUndefined()

      storage.storage.set = workingSet
      loaded.mailbox.advance(46_000)
      loaded.mailbox.expire('telegram:receipt')
      expect(await settle(runCycle(loaded, 'desktop:one', [action]))).toEqual([
        expect.objectContaining({ success: true })
      ])
      expect(loaded.gateway.rpcFor('prompt.submit')).toHaveLength(submissions)
      expect(
        (loaded.gateway.storage.get('group-chats') as Record<string, GroupChat>).Workshop.desktopCommandSettled?.[
          'telegram:receipt'
        ]
      ).toMatchObject({ at: expect.any(Number), action })
    }
  )

  it('fails closed before dispatch if the command entry cannot be persisted', async () => {
    const loaded = await load()
    seed(loaded)
    const shared = await import('./shared')
    const storage = scriptedStorage(loaded.gateway.storage)
    const workingSet = storage.storage.set

    storage.storage.set = (key, value) => {
      if (!(value as Record<string, GroupChat>).Workshop?.log.some(entry => entry.external)) {
        workingSet(key, value)
      }
    }

    shared.setPluginCtx(storage)
    loaded.mailbox.queue({
      action: 'send',
      command_id: 'telegram:entry',
      room_id: 'room-1',
      payload: { message: 'Review', recipients: members }
    })
    expect(await settle(runCycle(loaded, 'desktop:one', ['send']))).toEqual([
      expect.objectContaining({ retryable: true })
    ])
    expect(loaded.gateway.rpcFor('prompt.submit')).toHaveLength(0)
    expect(loaded.mailbox.calls.some(call => call.method === 'groups.desktop.complete')).toBe(false)
    expect(loaded.chat.$groupChats.get().Workshop.desktopCommandSettled?.['telegram:entry']).toBeUndefined()
  })

  it('retries silent startup writes only on an activation event and preserves the token', async () => {
    const loaded = await load()
    seed(loaded)
    const shared = await import('./shared')
    const storage = scriptedStorage(loaded.gateway.storage)
    storage.storage.set = () => undefined
    loaded.gateway.storage.clear()
    shared.setPluginCtx(storage)
    const token = loaded.chat.$groupChats.get().Workshop.desktopAuthorityToken
    await expect(loaded.runtime.startDesktopRoomCommandRuntime()).rejects.toThrow('could not be saved')
    await vi.advanceTimersByTimeAsync(120_000)
    expect(loaded.mailbox.calls).toHaveLength(0)
    expect(vi.getTimerCount()).toBe(0)

    shared.setPluginCtx(scriptedStorage(loaded.gateway.storage))
    await loaded.runtime.startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(0)
    expect(loaded.mailbox.calls.some(call => call.method === 'groups.desktop.presence')).toBe(true)
    expect(loaded.chat.$groupChats.get().Workshop.desktopAuthorityToken).toBe(token)
  })

  it('stops advertising or claiming when persistence fails after startup, until an event retries', async () => {
    const loaded = await load()
    seed(loaded)
    await loaded.runtime.startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(0)
    const shared = await import('./shared')
    const storage = scriptedStorage(loaded.gateway.storage)
    storage.storage.set = () => undefined
    shared.setPluginCtx(storage)
    loaded.chat.updateGroupChat('Workshop', room => ({ ...room, desktopAuthorityConflict: true }), { sync: false })
    loaded.chat.updateGroupChat('Fresh', () => ({ log: [], watermarks: {}, roomId: 'fresh', members }), { sync: false })
    const before = loaded.mailbox.calls.length
    await vi.advanceTimersByTimeAsync(180_000)
    expect(loaded.mailbox.calls).toHaveLength(before)

    shared.setPluginCtx(scriptedStorage(loaded.gateway.storage))
    await loaded.runtime.startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(300)
    expect(loaded.mailbox.calls.length).toBeGreaterThan(before)
    const authorities = loaded.mailbox.calls.slice(before).flatMap(call => call.params.room_authorities || [])
    expect(authorities).not.toContainEqual(expect.objectContaining({ room_id: 'room-1' }))
  })

  it('keeps the same consumer on duplicate start while a command lease is active', async () => {
    const loaded = await load({ pollsBusy: 80, turn: () => 'late answer' })
    seed(loaded)
    loaded.mailbox.queue({
      action: 'send',
      command_id: 'telegram:active-lease',
      payload: { actor_display_name: 'David via Telegram', message: 'Long review', recipients: members },
      room_id: 'room-1'
    })
    await loaded.runtime.startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(500)
    const command = loaded.mailbox.commands.get('telegram:active-lease')
    expect(command?.state).toBe('claimed')
    const owner = command?.leaseOwner

    await loaded.runtime.startDesktopRoomCommandRuntime()
    await vi.advanceTimersByTimeAsync(15_000)
    expect(command?.leaseOwner).toBe(owner)

    const consumers = loaded.mailbox.calls
      .filter(call => call.method.startsWith('groups.desktop.'))
      .map(call => String(call.params.consumer_id))

    expect(new Set(consumers)).toEqual(new Set([owner]))
    loaded.runtime.stopDesktopRoomCommandRuntime()
    const stoppedAt = loaded.mailbox.calls.length
    await vi.advanceTimersByTimeAsync(120_000)
    expect(loaded.mailbox.calls.slice(stoppedAt).map(call => call.method)).toEqual([])
  })

  it('applies a Telegram-style queued send after reopen with actor provenance', async () => {
    const loaded = await load({ turn: () => '(pass)' })
    seed(loaded)
    loaded.mailbox.queue({
      action: 'send',
      command_id: 'telegram:783:1',
      payload: { actor_display_name: 'David via Telegram', message: 'Review the launch', recipients: members },
      room_id: 'room-1'
    })

    const outcomes = await settle(runCycle(loaded, 'desktop:one', ['send']))
    const room = loaded.chat.$groupChats.get().Workshop
    expect(outcomes).toEqual([{ commandId: 'telegram:783:1', connectionId: 'gateway-a', success: true }])
    expect(room.log.filter(entry => entry.id === 'telegram:783:1')).toEqual([
      expect.objectContaining({
        external: true,
        from: { kind: 'user', name: 'David via Telegram' },
        text: 'Review the launch'
      })
    ])
    expect(loaded.gateway.rpcFor('prompt.submit')).toHaveLength(1)
    expect(loaded.mailbox.commands.get('telegram:783:1')?.state).toBe('completed')
    const activity = await import('./group-activity')
    const queued = activity.$groupActivity.get().Workshop.events.find(event => event.kind === 'queued')
    expect(queued?.member).toBe('David via Telegram')
  })

  it('stops active mailbox work through the existing hold and interrupt path', async () => {
    const loaded = await load({ pollsBusy: 50, turn: () => 'late answer' })
    seed(loaded)
    loaded.mailbox.queue({
      action: 'send',
      command_id: 'telegram:send',
      payload: { actor_display_name: 'David via Telegram', message: 'Long review', recipients: members },
      room_id: 'room-1'
    })
    const send = runCycle(loaded, 'desktop:one', ['send'])
    await vi.advanceTimersByTimeAsync(500)
    loaded.mailbox.queue({
      action: 'stop',
      command_id: 'telegram:stop',
      payload: { target_command_id: 'telegram:send' },
      room_id: 'room-1'
    })
    const stopped = runCycle(loaded, 'desktop:one', ['stop'])

    await settle(Promise.all([send, stopped]))
    const room = loaded.chat.$groupChats.get().Workshop
    expect(room.running).toBe(false)
    expect(room.holds?.reviewer).toBeTruthy()
    expect(loaded.gateway.rpcFor('session.interrupt')).toHaveLength(1)
    expect(loaded.mailbox.commands.get('telegram:stop')?.state).toBe('completed')
  })

  it('replays after a lost completion without duplicating the user entry', async () => {
    const loaded = await load({ turn: () => '(pass)' })
    seed(loaded)
    loaded.mailbox.completeFailures = 2
    loaded.mailbox.queue({
      action: 'send',
      command_id: 'telegram:replay',
      payload: { actor_display_name: 'David via Telegram', message: 'Once only', recipients: members },
      room_id: 'room-1'
    })
    await settle(runCycle(loaded, 'desktop:one', ['send']))
    loaded.chat.$groupChats.set(structuredClone(loaded.chat.durableGroupChatRooms()))
    loaded.mailbox.expire('telegram:replay')
    loaded.mailbox.advance(46_000)
    loaded.mailbox.owners.set('room-1', { consumer: 'desktop:one', until: loaded.mailbox.clock - 1 })
    await settle(runCycle(loaded, 'desktop:two', ['send']))

    expect(loaded.chat.$groupChats.get().Workshop.log.filter(entry => entry.id === 'telegram:replay')).toHaveLength(1)
    expect(loaded.mailbox.commands.get('telegram:replay')?.state).toBe('completed')
  })

  it('recovers from a transient claim failure without a polling loop', async () => {
    const loaded = await load({ turn: () => '(pass)' })
    seed(loaded)
    loaded.mailbox.failClaims = 1
    loaded.mailbox.queue({
      action: 'send',
      command_id: 'telegram:retry',
      payload: { actor_display_name: 'David via Telegram', message: 'Try later', recipients: members },
      room_id: 'room-1'
    })
    expect(await runCycle(loaded, 'desktop:one', ['send'])).toEqual([])
    await settle(runCycle(loaded, 'desktop:one', ['send']))
    // One failed claim, one successful claim, then an empty queue check.
    expect(loaded.mailbox.calls.filter(call => call.method === 'groups.desktop.claim')).toHaveLength(3)
    expect(loaded.mailbox.commands.get('telegram:retry')?.state).toBe('completed')
  })

  it('fences two consumers and allows takeover only after expiry', async () => {
    const loaded = await load()
    seed(loaded)
    loaded.mailbox.queue({
      action: 'send',
      command_id: 'telegram:lease',
      payload: { actor_display_name: 'David via Telegram', message: 'Owned', recipients: members },
      room_id: 'room-1'
    })
    let release!: () => void

    const blocked = new Promise<void>(resolve => {
      release = resolve
    })

    const first = loaded.client.runDesktopRoomCommandCycle({
      consumerId: 'desktop:one',
      execute: async () => blocked,
      request: (_route, method, params) => loaded.mailbox.request(method, params),
      rooms: loaded.chat.$groupChats.get(),
      routes: [route]
    })

    await Promise.resolve()
    expect(await runCycle(loaded, 'desktop:two', ['send'])).toEqual([])
    loaded.mailbox.advance(46_000)
    loaded.mailbox.expire('telegram:lease')
    loaded.mailbox.owners.set('room-1', { consumer: 'desktop:one', until: loaded.mailbox.clock - 1 })
    release()
    await first
    await settle(runCycle(loaded, 'desktop:two', ['send']))
    expect(loaded.mailbox.commands.get('telegram:lease')?.state).toBe('completed')
  })

  it('follows a rename by room id but rejects same-name recreation and conflicts', async () => {
    const loaded = await load({ turn: () => '(pass)' })
    const room = seed(loaded)
    const descriptors = loaded.client.desktopRoomDescriptors(loaded.chat.$groupChats.get())
    loaded.chat.$groupChats.set({ Renamed: room })

    const renamed = loaded.runtime.executeDesktopRoomCommand(
      {
        action: 'send',
        command_id: 'telegram:rename',
        lease_token: 'lease:rename',
        payload: { actor_display_name: 'David via Telegram', message: 'After rename', recipients: members },
        room_id: 'room-1'
      },
      descriptors,
      { consumerId: 'desktop:one', request: async () => ({}), route, signal: null }
    )

    await settle(renamed)
    expect(loaded.chat.$groupChats.get().Renamed.log.some(entry => entry.id === 'telegram:rename')).toBe(true)

    loaded.chat.$groupChats.set({ Workshop: { ...room, log: [], roomId: 'room-2' } })
    await expect(
      loaded.runtime.executeDesktopRoomCommand(
        {
          action: 'send',
          command_id: 'telegram:stale',
          lease_token: 'lease:stale',
          payload: { actor_display_name: 'David via Telegram', message: 'Stale', recipients: members },
          room_id: 'room-1'
        },
        descriptors,
        { consumerId: 'desktop:one', request: async () => ({}), route, signal: null }
      )
    ).rejects.toThrow('no longer available')
    loaded.chat.$groupChats.set({
      First: room,
      Second: {
        ...room,
        desktopAuthorityHash: room.desktopAuthorityHash,
        desktopAuthorityToken: room.desktopAuthorityToken
      }
    })
    await expect(
      loaded.runtime.executeDesktopRoomCommand(
        {
          action: 'send',
          command_id: 'telegram:duplicate-room',
          lease_token: 'lease:duplicate',
          payload: { actor_display_name: 'David via Telegram', message: 'Ambiguous', recipients: members },
          room_id: 'room-1'
        },
        descriptors,
        { consumerId: 'desktop:one', request: async () => ({}), route, signal: null }
      )
    ).rejects.toThrow('no longer available')
    loaded.chat.$groupChats.set({
      Workshop: {
        ...room,
        desktopAuthorityHash: undefined,
        desktopAuthorityToken: undefined,
        desktopAuthorityConflict: true
      }
    })
    expect(loaded.client.desktopRoomDescriptors(loaded.chat.$groupChats.get())).toEqual([])
  })

  it('isolates malformed commands and drains six healthy rooms within one bounded claim', async () => {
    const loaded = await load({ turn: () => '(pass)' })
    const rooms: Record<string, GroupChat> = {}

    for (let index = 0; index < 6; index += 1) {
      const roomId = `room-${index}`
      loaded.chat.$groupChats.set(rooms)
      loaded.chat.updateGroupChat(`Room ${index}`, () => ({ log: [], members, roomId, sessions: {}, watermarks: {} }), {
        sync: false
      })
      Object.assign(rooms, loaded.chat.$groupChats.get())
      loaded.mailbox.tokens.set(roomId, String(rooms[`Room ${index}`].desktopAuthorityToken))
      loaded.mailbox.queue({
        action: 'send',
        command_id: `telegram:${index}`,
        payload: { actor_display_name: 'David via Telegram', message: `Message ${index}`, recipients: members },
        room_id: roomId
      })
    }

    loaded.mailbox.queue({
      action: 'send',
      command_id: 'telegram:bad',
      payload: { message: 42 },
      room_id: 'room-0'
    })
    await settle(runCycle(loaded, 'desktop:one', ['send']), 500)

    const claim = loaded.mailbox.calls.find(call => call.method === 'groups.desktop.claim')
    expect(claim?.params.room_authorities as unknown[]).toHaveLength(6)
    expect(Number(claim?.params.limit)).toBeLessThanOrEqual(8)
    expect([...loaded.mailbox.commands.values()].filter(command => command.state === 'completed')).toHaveLength(6)
    expect(loaded.mailbox.commands.get('telegram:bad')?.state).toBe('failed')
    expect(vi.getTimerCount()).toBeLessThanOrEqual(2)
  })

  it('review repro: lease loss fences a late member reply when interrupt cannot stop it', async () => {
    const loaded = await load({ pollsBusy: 8, turn: ({ n }) => (n === 1 ? 'LATE_LEASE_REPLY' : '(pass)') })
    const room = seed(loaded)
    const controller = new AbortController()

    const execution = loaded.runtime.executeDesktopRoomCommand(
      {
        action: 'send',
        command_id: 'telegram:lease-loss',
        lease_token: 'lease:one',
        payload: { actor_display_name: 'David via Telegram', message: 'Long work', recipients: members },
        room_id: 'room-1'
      },
      [
        {
          authorityHash: String(room.desktopAuthorityHash),
          authorityToken: String(room.desktopAuthorityToken),
          name: 'Workshop',
          roomId: 'room-1'
        }
      ],
      { consumerId: 'desktop:one', request: async () => ({}), route, signal: controller.signal }
    )

    const outcome = execution.then(
      () => null,
      error => error as Error & { retryable?: boolean }
    )

    await vi.advanceTimersByTimeAsync(250)
    controller.abort('lease-lost')
    await vi.advanceTimersByTimeAsync(5_000)
    expect(await outcome).toMatchObject({ retryable: true })

    for (let index = 0; index < 15; index += 1) {
      await vi.advanceTimersByTimeAsync(5_000)
    }

    expect(loaded.chat.$groupChats.get().Workshop.log.map(entry => entry.text)).not.toContain('LATE_LEASE_REPLY')
    expect(loaded.chat.$groupChats.get().Workshop.holds || {}).toEqual({})
    expect(loaded.chat.$groupChats.get().Workshop.stranded || {}).toEqual({})
    loaded.mailbox.queue({
      action: 'send',
      command_id: 'telegram:after-loss',
      room_id: 'room-1',
      payload: { message: 'New unrelated work', recipients: members }
    })
    expect(await settle(runCycle(loaded, 'desktop:one', ['send']), 240)).toEqual([
      expect.objectContaining({ success: true })
    ])
    expect(loaded.gateway.rpcFor('prompt.submit')).toHaveLength(2)
  })

  it('review repro: Stop fences a late member reply when interrupt cannot stop it', async () => {
    const loaded = await load({ pollsBusy: 8, turn: () => 'LATE_STOP_REPLY' })
    const room = seed(loaded)

    const descriptors = [
      {
        authorityHash: String(room.desktopAuthorityHash),
        authorityToken: String(room.desktopAuthorityToken),
        name: 'Workshop',
        roomId: 'room-1'
      }
    ]

    const send = loaded.runtime.executeDesktopRoomCommand(
      {
        action: 'send',
        command_id: 'telegram:send-stop',
        lease_token: 'lease:send',
        payload: { actor_display_name: 'David via Telegram', message: 'Long work', recipients: members },
        room_id: 'room-1'
      },
      descriptors,
      { consumerId: 'desktop:one', request: async () => ({}), route, signal: null }
    )

    await vi.advanceTimersByTimeAsync(250)
    await loaded.runtime.executeDesktopRoomCommand(
      {
        action: 'stop',
        command_id: 'telegram:stop-late',
        lease_token: 'lease:stop',
        payload: { target_command_id: 'telegram:send-stop' },
        room_id: 'room-1'
      },
      descriptors,
      { consumerId: 'desktop:one', request: async () => ({}), route, signal: null }
    )
    await vi.advanceTimersByTimeAsync(30_000)
    await send

    expect(loaded.chat.$groupChats.get().Workshop.log.map(entry => entry.text)).not.toContain('LATE_STOP_REPLY')
  })
})
