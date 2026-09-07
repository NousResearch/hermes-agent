import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { scriptedStorage } from './group-test-utils'
import type * as hostedClient from './hosted-room-client'
import { hostedRoomKey, hostedTranscript } from './hosted-room-protocol'
import type { HostedEvent, HostedRoomSummary } from './hosted-room-protocol'
import type { ProfileRoute } from './types'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

// All transport and persistence here are explicitly in-memory mocks. No live
// gateway, Electron, profile credentials, or agent generation is exercised.
const identity = { connectionId: 'mock-owner', authorityGatewayId: 'mock-authority', roomId: 'mock-room' }
const key = hostedRoomKey(identity)
const pendingKey = `hosted-input:${key}`
const directoryKey = `hosted-rooms:${identity.connectionId}`
const route: ProfileRoute = { connectionId: identity.connectionId, mode: 'remote', profile: 'default', targetProfile: 'default' }

const baseRoom: HostedRoomSummary = {
  room_id: identity.roomId, authority_gateway_id: identity.authorityGatewayId,
  authority_epoch: 1, name: 'Mock room', latest_seq: 0, members: []
}

const baseCapabilities = {
  protocol_version: 2, authority_gateway_id: identity.authorityGatewayId,
  methods: ['groups.list', 'groups.state', 'groups.send', 'groups.log', 'groups.stop'],
  features: ['room_identity', 'monotonic_log', 'idempotent_send', 'typed_events'],
  max_log_limit: 100, driver: true, persistent_process: true
}

function event(seq: number, patch: Partial<HostedEvent> = {}): HostedEvent {
  return {
    room_id: identity.roomId, seq, event_id: `mock-event-${seq}`, kind: 'room.created',
    actor: { kind: 'system', id: 'mock-system' }, payload: {}, created_at: 1, authority_epoch: 1, ...patch
  }
}

function page(events: HostedEvent[], cursor = events.at(-1)?.seq ?? 0, latest = cursor) {
  return { events, cursor, latest_seq: latest, has_more: cursor < latest, authority: { gateway_id: identity.authorityGatewayId, epoch: 1 } }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })

  return { promise, resolve, reject }
}

type Client = typeof hostedClient
let client: Client
let disk: Map<string, unknown>
let ctx: ReturnType<typeof scriptedStorage>
let rpc: ReturnType<typeof vi.fn<(route: ProfileRoute, method: string, params: Record<string, unknown>) => Promise<unknown>>>
let legacy: ReturnType<typeof vi.fn>
let log: HostedEvent[]
let room: HostedRoomSummary
let caps: Record<string, unknown>

function cache() { return client.$hostedRooms.get()[key] }

function seed() {
  client.$hostedRooms.set({ [key]: { identity, name: baseRoom.name, cursor: 0, events: [], loading: false, busy: false } })
}

async function respond(_route: ProfileRoute, method: string, params: Record<string, unknown>): Promise<unknown> {
  switch (method) {
    case 'groups.capabilities': return caps

    case 'groups.state': return { room: { ...room, latest_seq: log.at(-1)?.seq ?? room.latest_seq }, driver_status: { running: false } }

    case 'groups.list': return { rooms: [room], next_offset: null }
    case 'groups.log': {
      const since = Number(params.since_seq)
      const events = log.filter(row => row.seq > since).slice(0, Number(params.limit))

      return page(events, events.at(-1)?.seq ?? since, log.at(-1)?.seq ?? 0)
    }

    case 'groups.send': {
      // Mock idempotent server: client retry ID and canonical log ID differ.
      const eventId = `mock-canonical:${String(params.event_id)}`
      let committed = log.find(row => row.event_id === eventId)

      if (!committed) {
        committed = event(log.length + 1, {
          event_id: eventId, kind: 'message.user', actor: { kind: 'user', id: 'mock-human' },
          payload: params.payload as Record<string, unknown>, authority_epoch: null
        })
        log.push(committed)
      }

      return { accepted: true, client_event_id: params.event_id, event: committed, driver_started: true }
    }

    case 'groups.stop': return { cancelled: 0 }

    default: throw new Error(`Unexpected mock RPC: ${method}`)
  }
}

function sentCalls() { return rpc.mock.calls.filter(([, method]) => method === 'groups.send') }

beforeEach(async () => {
  vi.resetModules()

  for (const field of Object.keys(host)) {delete host[field]}
  disk = new Map()
  ctx = scriptedStorage(disk)
  room = structuredClone(baseRoom)
  caps = structuredClone(baseCapabilities)
  log = []
  rpc = vi.fn(respond)
  legacy = vi.fn(() => { throw new Error('Legacy fallback forbidden') })
  Object.assign(host, {
    profileRoutes: vi.fn(async () => [
      { connectionId: 'mock-other', profile: 'default' },
      { connectionId: identity.connectionId, profile: 'helper' }, route
    ]),
    requestProfile: rpc, request: legacy
  })
  client = await import('./hosted-room-client')
  const shared = await import('./shared')
  shared.setPluginCtx(ctx)
  seed()
})
afterEach(() => {
  expect(legacy).not.toHaveBeenCalled()
  expect(rpc.mock.calls.every(([, method]) => method.startsWith('groups.'))).toBe(true)
  vi.restoreAllMocks()
})

describe('attachment preflight before durable admission', () => {
  const file = { kind: 'file' as const, name: 'mock.txt', data: 'data:text/plain;base64,dGVzdA==' }

  it.each([
    ['too many files', Array.from({ length: 9 }, () => ({ ...file }))],
    ['missing bytes', [{ ...file, data: '' }]],
    ['invalid base64 padding', [{ ...file, data: 'data:text/plain;base64,QQ=Q' }]],
    ['invalid MIME type', [{ ...file, data: 'data:not a mime;base64,dGVzdA==' }]]
  ] as const)('rejects %s without permanently locking the room with an unsendable pending input', async (_name, files) => {
    expect(await client.sendHostedInput(key, '', null, [...files])).toBe(false)
    expect(disk.get(pendingKey)).toBeUndefined()
    expect(cache().pending).toBeUndefined()
    expect(rpc).not.toHaveBeenCalled()
    expect(cache().error).toBeTruthy()
    expect(await client.sendHostedInput(key, 'A corrected message')).toBe(true)
  })
})

describe('ownership and capabilities', () => {
  it('discovers and persists scoped native identities on the owning default route only', async () => {
    await client.discoverHostedRooms(identity.connectionId)
    expect(client.$hostedDirectories.get()[identity.connectionId]).toEqual({ keys: [key], loading: false })
    expect(disk.get(directoryKey)).toEqual([{ identity, name: room.name }])
    expect(rpc.mock.calls.every(([candidate]) => candidate === route)).toBe(true)
  })
  it('retains saved identities during an unsupported-backend outage', async () => {
    disk.set(directoryKey, [{ identity, name: room.name }])
    client.$hostedRooms.set({})
    rpc.mockRejectedValue(Object.assign(new Error('Method not found'), { code: -32601 }))
    await client.discoverHostedRooms(identity.connectionId)
    expect(client.$hostedDirectories.get()[identity.connectionId]).toMatchObject({ keys: [key], loading: false, error: expect.stringMatching(/does not support/) })
    expect(cache().identity).toEqual(identity)
    expect(rpc.mock.calls.map(([, method]) => method)).toEqual(['groups.capabilities'])
  })
  it.each(['refresh', 'send', 'stop'] as const)('fails closed on changed capability authority during %s', async action => {
    caps.authority_gateway_id = 'mock-impostor'

    if (action === 'refresh') {await client.refreshHostedRoom(key)}

    if (action === 'send') {expect(await client.sendHostedInput(key, 'Mock input')).toBe(false)}

    if (action === 'stop') {await client.stopHostedRoom(key)}
    expect(cache().error).toMatch(/different authority/)
    expect(rpc.mock.calls.map(([, method]) => method)).toEqual(['groups.capabilities'])
    expect(cache().identity).toEqual(identity)
  })
  it.each(['room_id', 'authority_gateway_id'] as const)('validates state %s before sending', async field => {
    room[field] = 'mock-impostor'
    expect(await client.sendHostedInput(key, 'Mock input')).toBe(false)
    expect(cache().error).toMatch(/authority/)
    expect(sentCalls()).toHaveLength(0)
    expect(disk.get(pendingKey)).toBeTruthy()
  })
  it.each([
    { protocol_version: 1 },
    { methods: ['groups.list', 'groups.state'] },
    { features: [] }
  ])('rejects unsupported protocol %j without any legacy fallback', async patch => {
    Object.assign(caps, patch)
    expect(await client.sendHostedInput(key, 'Mock input')).toBe(false)
    expect(cache().error).toMatch(/does not support/)
    expect(rpc.mock.calls.map(([, method]) => method)).toEqual(['groups.capabilities'])
  })
  it('keeps replay readable when worker is absent, but preserves input instead of sending', async () => {
    caps.driver = false
    await client.refreshHostedRoom(key)
    expect(cache().error).toBeUndefined()
    expect(cache().capabilities?.driver).toBe(false)
    expect(await client.sendHostedInput(key, 'Mock input')).toBe(false)
    expect(cache().error).toMatch(/worker is unavailable/)
    expect(sentCalls()).toHaveLength(0)
    expect(disk.get(pendingKey)).toMatchObject({ text: 'Mock input' })
  })
  it('will not borrow a different connection when the owner disappears', async () => {
    host.profileRoutes = vi.fn(async () => [{ connectionId: 'mock-other', profile: 'default' }])
    expect(await client.sendHostedInput(key, 'Mock input')).toBe(false)
    expect(rpc).not.toHaveBeenCalled()
    expect(cache().error).toMatch(/owning gateway/)
  })
  it('does not fall back when requestProfile is missing', async () => {
    delete host.requestProfile
    await client.refreshHostedRoom(key)
    expect(cache().error).toMatch(/cannot route/)
    expect(rpc).not.toHaveBeenCalled()
  })
})

describe('durable send and acknowledgements', () => {
  it('hands off the composer while admission is held, before delayed post-send refresh', async () => {
    const refreshing = deferred<void>()
    const release = deferred<void>()
    let negotiations = 0
    let draft = 'Mock input'
    rpc.mockImplementation(async (...args) => {
      if (args[1] === 'groups.capabilities' && ++negotiations === 2) {
        refreshing.resolve()
        await release.promise
      }

      return respond(...args)
    })

    const onQueued = vi.fn(() => {
      expect(cache().busy).toBe(true)
      expect(disk.get(pendingKey)).toMatchObject({ text: 'Mock input' })
      expect(rpc).not.toHaveBeenCalled()
      draft = ''
    })

    const sending = client.sendHostedInput(key, draft, null, [], onQueued)
    await refreshing.promise

    try {
      expect(onQueued).toHaveBeenCalledOnce()
      expect(draft).toBe('')
      expect(sentCalls()).toHaveLength(1)
    } finally {
      release.resolve()
      await sending
    }
  })
  it('does not retire the composer when durable storage fails', async () => {
    vi.spyOn(ctx.storage, 'set').mockRejectedValue(new Error('mock disk full'))
    const onQueued = vi.fn()
    expect(await client.sendHostedInput(key, 'Mock input', null, [], onQueued)).toBe(false)
    expect(onQueued).not.toHaveBeenCalled()
  })
  it('waits for the pending ID and payload to become durable before any RPC', async () => {
    const entered = deferred<void>()
    const durable = deferred<void>()
    const original = ctx.storage.set.bind(ctx.storage)
    vi.spyOn(ctx.storage, 'set').mockImplementation(async (storageKey, value) => {
      if (storageKey === pendingKey && value) { entered.resolve(); await durable.promise }

      return original(storageKey, value)
    })
    const sending = client.sendHostedInput(key, '  Mock input  ', 'mock-thread')
    await entered.promise
    expect(rpc).not.toHaveBeenCalled()
    durable.resolve()
    expect(await sending).toBe(true)
    expect(sentCalls()[0][2]).toMatchObject({ payload: { text: 'Mock input', thread_id: 'mock-thread' } })
    expect(cache().cursor).toBe(1)
    expect(disk.get(pendingKey)).toBeNull()
  })
  it('fails closed when the real plugin storage silently swallows a quota error', async () => {
    const { createPluginContext } = await import('../../contrib/plugin')

    const nativeContext = createPluginContext('hosted-room-quota-proof')

    ;(await import('./shared')).setPluginCtx(nativeContext)
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('Quota exceeded', 'QuotaExceededError')
    })
    expect(await client.sendHostedInput(key, 'Mock input', null, [
      { kind: 'file', name: 'mock.txt', data: 'data:text/plain;base64,dGVzdA==' }
    ])).toBe(false)
    expect(rpc).not.toHaveBeenCalled()
    expect(cache().pending).toBeUndefined()
    expect(cache().error).toMatch(/save|persist|storage/i)
  })
  it('does not send a payload when read-back differs from the saved input', async () => {
    vi.spyOn(ctx.storage, 'set').mockImplementation(async (storageKey, value) => {
      disk.set(storageKey, { ...value as Record<string, unknown>, text: 'Unexpected persisted text' })
    })
    expect(await client.sendHostedInput(key, 'Mock input')).toBe(false)
    expect(rpc).not.toHaveBeenCalled()
    expect(cache().error).toMatch(/save|persist|storage/i)
  })
  it('does not send when persistence fails', async () => {
    vi.spyOn(ctx.storage, 'set').mockRejectedValue(new Error('mock disk full'))
    expect(await client.sendHostedInput(key, 'Mock input')).toBe(false)
    expect(rpc).not.toHaveBeenCalled()
    expect(cache()).toMatchObject({ busy: false, error: 'mock disk full' })
  })
  it('retries uncertain acceptance with the same ID and yields only one committed transcript entry', async () => {
    let lost = false
    rpc.mockImplementation(async (...args) => {
      const result = await respond(...args)

      if (args[1] === 'groups.send' && !lost) { lost = true; throw new Error('mock connection lost after commit') }

      return result
    })
    expect(await client.sendHostedInput(key, 'Mock input', 'mock-thread')).toBe(false)
    const pending = structuredClone(disk.get(pendingKey))
    expect(pending).toMatchObject({ text: 'Mock input', threadId: 'mock-thread' })
    expect(cache().events).toEqual([])
    expect(await client.sendHostedInput(key)).toBe(true)
    expect(sentCalls()[1][2]).toEqual(sentCalls()[0][2])
    expect(hostedTranscript(cache()).log).toHaveLength(1)
    await client.refreshHostedRoom(key)
    expect(hostedTranscript(cache()).log).toHaveLength(1)
    expect(cache().pending).toBeUndefined()
    expect(disk.get(pendingKey)).toBeNull()
  })
  it('reloads a persisted unresolved input without auto-send, then explicitly retries its exact ID', async () => {
    const pending = { eventId: 'mock-persisted-id', text: 'Mock saved input', threadId: 'mock-saved-thread' }
    disk.set(pendingKey, pending)
    vi.resetModules()
    client = await import('./hosted-room-client')
    ;(await import('./shared')).setPluginCtx(ctx)
    seed()
    await client.refreshHostedRoom(key)
    expect(cache().pending).toEqual(pending)
    expect(sentCalls()).toHaveLength(0)
    expect(await client.sendHostedInput(key)).toBe(true)
    expect(sentCalls()[0][2]).toEqual({ room_id: identity.roomId, event_id: pending.eventId, payload: { text: pending.text, thread_id: pending.threadId } })
  })
  it('refuses a different payload while an input is unresolved', async () => {
    const pending = { eventId: 'mock-pending', text: 'Mock first', threadId: 'mock-thread' }
    disk.set(pendingKey, pending)
    expect(await client.sendHostedInput(key, 'Mock second')).toBe(false)
    expect(await client.sendHostedInput(key, 'Mock first', 'mock-other-thread')).toBe(false)
    expect(rpc).not.toHaveBeenCalled()
    expect(disk.get(pendingKey)).toEqual(pending)
  })
  it('serializes concurrent sends, refresh and stop behind the unresolved operation', async () => {
    const entered = deferred<void>()
    const release = deferred<unknown>()
    rpc.mockImplementation(async (...args) => {
      if (args[1] === 'groups.send') { entered.resolve();

 return release.promise }

      return respond(...args)
    })
    const first = client.sendHostedInput(key, 'Mock input')
    await entered.promise
    expect(await client.sendHostedInput(key, 'Mock input')).toBe(false)
    await client.refreshHostedRoom(key)
    await client.stopHostedRoom(key)
    expect(sentCalls()).toHaveLength(1)
    expect(rpc.mock.calls.some(([, method]) => method === 'groups.stop')).toBe(false)
    release.reject(new Error('mock send timeout'))
    expect(await first).toBe(false)
    expect(cache().busy).toBe(false)
  })
  it.each(['client id', 'payload', 'room', 'acceptance'] as const)('keeps the durable input after an invalid acknowledgement: %s', async invalid => {
    rpc.mockImplementation(async (...args) => {
      const result = await respond(...args)

      if (args[1] !== 'groups.send') {return result}
      const ack = structuredClone(result) as { client_event_id: string; accepted: boolean; event: HostedEvent }

      if (invalid === 'client id') {ack.client_event_id = 'wrong'}

      if (invalid === 'payload') {ack.event.payload.text = 'wrong'}

      if (invalid === 'room') {ack.event.room_id = 'wrong'}

      if (invalid === 'acceptance') {ack.accepted = false}

      return ack
    })
    expect(await client.sendHostedInput(key, 'Mock input')).toBe(false)
    expect(disk.get(pendingKey)).toBeTruthy()
    expect(cache().pending).toBeTruthy()
    expect(cache().cursor).toBe(0)
  })
  it('keeps pending when acknowledgement cannot be found in committed log', async () => {
    rpc.mockImplementation(async (...args) => {
      if (args[1] === 'groups.log' && args[2].limit === 1) {return page([])}

      return respond(...args)
    })
    expect(await client.sendHostedInput(key, 'Mock input')).toBe(false)
    expect(disk.get(pendingKey)).toBeTruthy()
    expect(cache().events).toEqual([])
  })
  it.each(['authority', 'payload', 'sequence', 'cursor'] as const)('does not discard pending on a same-ID but invalid committed proof: %s', async invalid => {
    rpc.mockImplementation(async (...args) => {
      const result = await respond(...args)

      if (args[1] !== 'groups.log' || args[2].limit !== 1) {return result}
      const proof = structuredClone(result) as ReturnType<typeof page>

      if (invalid === 'authority') {proof.authority.gateway_id = 'mock-impostor'}

      if (invalid === 'payload') {proof.events[0].payload.text = 'Mock different input'}

      if (invalid === 'sequence') {proof.events[0].seq = 9}

      if (invalid === 'cursor') {proof.cursor = 9}

      return proof
    })
    expect(await client.sendHostedInput(key, 'Mock input')).toBe(false)
    expect(disk.get(pendingKey)).toBeTruthy()
    expect(cache().pending).toBeTruthy()
  })
})

describe('atomic replay and async ordering', () => {
  it('replays multiple pages from the last validated cursor, including invisible control events', async () => {
    caps.max_log_limit = 1
    log = [event(1), event(2), event(3)]
    await client.refreshHostedRoom(key)
    expect(cache().cursor).toBe(3)
    expect(cache().events).toEqual(log)
    expect(rpc.mock.calls.filter(([, method]) => method === 'groups.log').map(([, , params]) => params.since_seq)).toEqual([0, 1, 2])
  })
  it('keeps the entire prior cache and cursor if a later page has a gap', async () => {
    log = [event(1)]
    await client.refreshHostedRoom(key)
    const before = cache()
    caps.max_log_limit = 1
    log = [event(1), event(2), event(4)]
    await client.refreshHostedRoom(key)
    expect(cache().cursor).toBe(before.cursor)
    expect(cache().events).toBe(before.events)
    expect(cache().room).toBe(before.room)
    expect(cache().error).toMatch(/gap/)
    expect(cache().loading).toBe(false)
  })
  it.each(['success', 'error'] as const)('ignores a stale competing refresh %s', async staleResult => {
    const entered = deferred<void>()
    const oldLog = deferred<unknown>()
    let held = false
    rpc.mockImplementation(async (...args) => {
      if (args[1] === 'groups.log' && !held) { held = true; entered.resolve();

 return oldLog.promise }

      return respond(...args)
    })
    const first = client.refreshHostedRoom(key)
    await entered.promise
    room.name = 'Mock newer name'
    log = [event(1)]
    await client.refreshHostedRoom(key)
    const newer = cache()

    if (staleResult === 'success') {oldLog.resolve(page([]))}
    else {oldLog.reject(new Error('mock old transport failure'))}

    await first
    expect(cache()).toBe(newer)
    expect(cache()).toMatchObject({ name: 'Mock newer name', cursor: 1, loading: false, error: undefined })
  })
  it('does not restore stale pending or replay after send supersedes a refresh', async () => {
    const entered = deferred<void>()
    const oldState = deferred<unknown>()
    let held = false
    rpc.mockImplementation(async (...args) => {
      if (args[1] === 'groups.state' && !held) { held = true; entered.resolve();

 return oldState.promise }

      return respond(...args)
    })
    const refreshing = client.refreshHostedRoom(key)
    await entered.promise
    expect(await client.sendHostedInput(key, 'Mock input')).toBe(true)
    const latest = cache()
    oldState.resolve({ room: baseRoom })
    await refreshing
    expect(cache()).toBe(latest)
    expect(cache().cursor).toBe(1)
    expect(cache().pending).toBeUndefined()
  })
})
