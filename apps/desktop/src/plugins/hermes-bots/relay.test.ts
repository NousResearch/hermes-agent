/**
 * The cross-connection bot relay — the Desktop-as-router loops that let a bot
 * on one gateway reach a bot on another (Aug 2026 ruling: connections ARE the
 * peer set).
 *
 * Three incidents are pinned here:
 *
 *  - #93091 — the drain was poll-only, so a DM waited out the interval before
 *    it moved (#92760 "slow replies"). The gateway now broadcasts
 *    `bot_relay.outbox.pending`; a burst of signals collapses to ONE drain,
 *    a push landing mid-drain re-schedules instead of being swallowed, and
 *    the interval survives as a BACKSTOP only. Item 2 of the same issue: a
 *    transient `profiles.list` blip must not be pushed as "that machine is
 *    empty" — the gateway reads absence from a fresh roster as offline.
 *  - #93594 — each registered connection's pooled socket is pinned open while
 *    the relay runs, so the drain reuses one persistent WebSocket instead of
 *    dialing and tearing one down per tick.
 *  - the waiter contract — a missing target connection must still post a reply,
 *    or the sending agent hangs until its own timeout.
 *
 * Everything is driven through the two real lifecycle doors, `startBotRelay` /
 * `stopBotRelay`; only the SDK `host` and the attention hooks are mocked.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createHash } from 'node:crypto'

import type { ProfileRoute } from './types'

const { clearBotAttentionMock, hostMock, noteBotAttentionMock, UnboundedCache } = vi.hoisted(() => ({
  clearBotAttentionMock: vi.fn(),
  hostMock: {
    agents: vi.fn(),
    connections: vi.fn(),
    onEvent: vi.fn(),
    profileRoutes: vi.fn(),
    requestProfile: vi.fn(),
    retainProfile: vi.fn(),
    retainProfileSocket: vi.fn(),
    warmAgent: vi.fn()
  } as Record<string, unknown>,
  noteBotAttentionMock: vi.fn(),
  // Stand-in for the SDK's LruCache. Its ceiling has its own unit test and no
  // fixture here approaches it, so the double just drops the bound.
  UnboundedCache: class extends Map {
    constructor(_max: number) {
      super()
    }
  }
}))

vi.mock('@hermes/plugin-sdk', () => ({ host: hostMock, LruCache: UnboundedCache }))

vi.mock('./data', () => ({
  botHandle: (name: string) => (name === 'default' ? 'hermes' : name),
  clearBotAttention: clearBotAttentionMock,
  noteBotAttention: noteBotAttentionMock
}))

const RELAY_PUSH_DEBOUNCE_MS = 250
const RELAY_DRAIN_INTERVAL_MS = 30_000
const RELAY_ROUTE_RECONNECT_GRACE_MS = 30_000

const route = (id: string): ProfileRoute => ({
  connectionId: id,
  mode: 'remote',
  profile: 'default',
  targetProfile: 'default'
})

/** Every RPC through one table, recording what each connection was asked. */
interface RelayCall {
  connectionId: string
  method: string
  params: Record<string, unknown>
}

function respondWith(handler: (call: RelayCall) => unknown) {
  const calls: RelayCall[] = []

  ;(hostMock.requestProfile as ReturnType<typeof vi.fn>).mockImplementation(
    async (target: ProfileRoute, method: string, params: Record<string, unknown>) => {
      const call = { connectionId: target.connectionId, method, params: structuredClone(params ?? {}) }

      calls.push(call)

      return handler(call)
    }
  )

  return calls
}

/** Pins handed out by the mocked retention door, in grant order. */
interface Pin {
  released: boolean
  route: ProfileRoute
}

function trackRetention() {
  const pins: Pin[] = []

  ;(hostMock.retainProfileSocket as ReturnType<typeof vi.fn>).mockImplementation((pinned: ProfileRoute) => {
    const pin: Pin = { released: false, route: pinned }

    pins.push(pin)

    return () => {
      pin.released = true
    }
  })

  return pins
}

async function loadRelay() {
  vi.resetModules()

  return import('./relay')
}

/** Fire the gateway's pending-envelope broadcast and let the debounced drain
 *  run to completion. */
async function pushAndSettle(times = 1) {
  const listener = (hostMock.onEvent as ReturnType<typeof vi.fn>).mock.calls.at(-1)?.[1] as () => void

  for (let i = 0; i < times; i += 1) {
    listener()
  }

  await vi.advanceTimersByTimeAsync(RELAY_PUSH_DEBOUNCE_MS + 10)
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.clearAllMocks()
  vi.stubGlobal('crypto', {
    subtle: {
      digest: async (_algorithm: string, input: BufferSource) => {
        const bytes = input instanceof ArrayBuffer
          ? new Uint8Array(input)
          : new Uint8Array(input.buffer, input.byteOffset, input.byteLength)
        const digest = createHash('sha256').update(bytes).digest()
        return digest.buffer.slice(digest.byteOffset, digest.byteOffset + digest.byteLength)
      }
    }
  } as unknown as Crypto)
  hostMock.onEvent = vi.fn(() => vi.fn())
  hostMock.agents = vi.fn(async () => ({ agents: [], sources: [] }))
  hostMock.connections = vi.fn(async () => [])
  hostMock.profileRoutes = vi.fn(async () => [route('a'), route('b')])
  hostMock.requestProfile = vi.fn(async () => ({}))
  hostMock.retainProfile = vi.fn(async () => vi.fn())
  hostMock.retainProfileSocket = vi.fn(() => vi.fn())
  hostMock.warmAgent = vi.fn()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('push-notified drain (#93091)', () => {
  it('warms every registered gateway after a cold renderer launch', async () => {
    hostMock.connections = vi.fn(async () => [{ id: 'a' }, { id: 'b' }, { id: 'c' }])
    hostMock.profileRoutes = vi.fn(async () => [route('a'), route('b'), route('c')])
    respondWith(() => ({ envelopes: [] }))

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(1000)

    expect(hostMock.warmAgent).toHaveBeenCalledTimes(3)
    expect(hostMock.warmAgent).toHaveBeenCalledWith('a', 'default')
    expect(hostMock.warmAgent).toHaveBeenCalledWith('b', 'default')
    expect(hostMock.warmAgent).toHaveBeenCalledWith('c', 'default')

    stopBotRelay()
  })

  it('collapses a burst of pending signals into ONE drain', async () => {
    const calls = respondWith(() => ({ envelopes: [] }))
    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(0)
    calls.length = 0

    await pushAndSettle(6)

    // One drain visits each connection's outbox exactly once.
    expect(calls.filter(call => call.method === 'bot_relay.outbox.drain')).toHaveLength(2)

    stopBotRelay()
  })

  it('drains again for a signal that lands after the window closed', async () => {
    const calls = respondWith(() => ({ envelopes: [] }))
    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(0)
    calls.length = 0

    await pushAndSettle()
    await pushAndSettle(2)

    expect(calls.filter(call => call.method === 'bot_relay.outbox.drain')).toHaveLength(4)

    stopBotRelay()
  })

  it('keeps the interval poll as a BACKSTOP at the slow cadence', async () => {
    // The poll was 4s back when it WAS the delivery path — which (before
    // route retention) meant a fresh WebSocket dial + teardown per connection
    // every 4s. Push carries envelope latency now; the poll only covers older
    // backends and events that never reach the tap.
    const calls = respondWith(() => ({ envelopes: [] }))
    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(0)
    calls.length = 0

    await vi.advanceTimersByTimeAsync(RELAY_DRAIN_INTERVAL_MS - 1000)

    expect(calls.filter(call => call.method === 'bot_relay.outbox.drain')).toHaveLength(0)

    await vi.advanceTimersByTimeAsync(2000)

    expect(calls.filter(call => call.method === 'bot_relay.outbox.drain')).toHaveLength(2)

    stopBotRelay()
  })

  it('re-schedules a push that raced an in-flight drain instead of dropping it', async () => {
    // The gateway signature is monotone — one event per new envelope, never
    // re-broadcast — so a push swallowed by the busy guard would strand its
    // envelope until the poll.
    let release!: () => void

    const firstDrain = new Promise<void>(resolve => {
      release = resolve
    })

    let drains = 0

    const calls = respondWith(async call => {
      if (call.method === 'bot_relay.outbox.drain') {
        drains += 1

        if (drains === 1) {
          await firstDrain
        }
      }

      return { envelopes: [] }
    })

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(0)
    calls.length = 0

    await pushAndSettle()
    const midFlight = drains

    // A second push while the first drain is still awaiting its RPC.
    await pushAndSettle()
    expect(drains).toBe(midFlight)

    release()
    await vi.advanceTimersByTimeAsync(RELAY_PUSH_DEBOUNCE_MS + 10)

    expect(drains).toBeGreaterThan(midFlight)

    stopBotRelay()
  })

  it('never schedules a drain once the relay is stopped', async () => {
    const calls = respondWith(() => ({ envelopes: [] }))
    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(0)
    stopBotRelay()
    calls.length = 0

    await pushAndSettle(3)
    await vi.advanceTimersByTimeAsync(RELAY_DRAIN_INTERVAL_MS * 2)

    expect(calls).toHaveLength(0)
  })

  it('feature-detects the event door and disposes its subscription', async () => {
    const unsubscribe = vi.fn()

    hostMock.onEvent = vi.fn(() => unsubscribe)

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    expect(hostMock.onEvent).toHaveBeenCalledWith('bot_relay.outbox.pending', expect.any(Function))

    stopBotRelay()
    expect(unsubscribe).toHaveBeenCalledTimes(1)

    // An older shell has no event door at all; the relay still starts.
    hostMock.onEvent = undefined
    const legacy = await loadRelay()

    expect(() => legacy.startBotRelay()).not.toThrow()
    legacy.stopBotRelay()
  })
})

describe('relay-route socket retention (#93594)', () => {
  it('pins each connection ONCE across many drain ticks', async () => {
    const pins = trackRetention()

    respondWith(() => ({ envelopes: [] }))

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()

    for (let tick = 0; tick < 4; tick += 1) {
      await pushAndSettle()
    }

    expect(pins.map(pin => pin.route.connectionId)).toEqual(['a', 'b'])
    expect(pins.every(pin => !pin.released)).toBe(true)

    stopBotRelay()
  })

  it('releases exactly the pin of a connection that left the registry', async () => {
    const pins = trackRetention()

    respondWith(() => ({ envelopes: [] }))

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await pushAndSettle()

    hostMock.profileRoutes = vi.fn(async () => [route('a'), route('c')])
    await pushAndSettle()

    expect(pins.filter(pin => pin.released).map(pin => pin.route.connectionId)).toEqual(['b'])
    expect(pins.map(pin => pin.route.connectionId)).toEqual(['a', 'b', 'c'])

    stopBotRelay()
  })

  it('drops every pin on stop, and stays idempotent', async () => {
    const pins = trackRetention()

    respondWith(() => ({ envelopes: [] }))

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await pushAndSettle()
    stopBotRelay()

    expect(pins.every(pin => pin.released)).toBe(true)

    // A second stop releases nothing new — the pins are already gone.
    const releasedCount = pins.filter(pin => pin.released).length

    stopBotRelay()
    expect(pins.filter(pin => pin.released)).toHaveLength(releasedCount)
  })

  it('unpins everything when the peer set drops below two connections', async () => {
    // Retention follows the relay-ELIGIBLE set: with nothing to relay,
    // nothing stays pinned.
    const pins = trackRetention()

    respondWith(() => ({ envelopes: [] }))

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await pushAndSettle()
    expect(pins).toHaveLength(2)

    hostMock.profileRoutes = vi.fn(async () => [route('a')])
    await pushAndSettle()

    expect(pins.every(pin => pin.released)).toBe(true)

    stopBotRelay()
  })

  it('is feature-detected: an older shell without the door is a no-op', async () => {
    hostMock.retainProfileSocket = undefined

    respondWith(() => ({ envelopes: [] }))

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await expect(pushAndSettle()).resolves.toBeUndefined()

    stopBotRelay()
  })
})

describe('the roster loop pushes the OTHER connections’ agents', () => {
  it('includes a registered gateway whose renderer route has not materialized yet', async () => {
    hostMock.connections = vi.fn(async () => [
      { id: 'a', kind: 'local' },
      { id: 'b', kind: 'ssh' },
      { id: 'm5', kind: 'ssh' }
    ])
    hostMock.profileRoutes = vi.fn(async () => [route('a'), route('b')])
    hostMock.agents = vi.fn(async () => ({ agents: [], sources: [] }))

    const calls = respondWith(call => {
      if (call.method === 'profiles.list' && call.connectionId === 'm5') {
        throw new Error('lazy profile route not materialized')
      }

      return call.method === 'profiles.list' ? { profiles: [{ name: 'default' }] } : {}
    })

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(0)

    expect(hostMock.retainProfile).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'm5', profile: 'default' })
    )
    expect(hostMock.warmAgent).toHaveBeenCalledWith('m5', 'default')
    expect(calls).not.toContainEqual(
      expect.objectContaining({ connectionId: 'm5', method: 'profiles.list' })
    )
    expect(calls).not.toContainEqual(
      expect.objectContaining({ connectionId: 'm5', method: 'bot_relay.roster.sync' })
    )
    const pushedToA = calls.find(call => call.method === 'bot_relay.roster.sync' && call.connectionId === 'a')

    expect(pushedToA?.params.agents).toEqual(
      expect.arrayContaining([expect.objectContaining({ connection_id: 'm5', profile: 'default' })])
    )

    stopBotRelay()
  })

  it('gives each gateway a union roster that excludes its own agents', async () => {
    const calls = respondWith(call => {
      if (call.method === 'profiles.list') {
        return { profiles: [{ name: call.connectionId === 'a' ? 'default' : 'ops' }] }
      }

      return {}
    })

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(0)

    const syncs = calls.filter(call => call.method === 'bot_relay.roster.sync')

    expect(syncs.map(call => call.connectionId)).toEqual(['a', 'b'])
    expect(syncs[0].params.agents).toEqual([
      expect.objectContaining({ connection_id: 'b', handle: 'ops', profile: 'ops' })
    ])
    // The primary profile is published by its callable alias, never "default".
    expect(syncs[1].params.agents).toEqual([
      expect.objectContaining({ connection_id: 'a', handle: 'hermes', profile: 'default' })
    ])

    stopBotRelay()
  })

  it('never conflates a transient fetch failure with an empty connection', async () => {
    // A live machine whose profiles.list blips must not be pushed as absent:
    // the gateway-side liveness check reads "absent from a fresh roster" as
    // definitively offline and refuses enqueues with a false runtime_offline
    // (#93091 item 2).
    let failB = false

    const calls = respondWith(call => {
      if (call.method === 'profiles.list') {
        if (call.connectionId === 'b' && failB) {
          throw new Error('socket blip')
        }

        return { profiles: [{ name: call.connectionId === 'a' ? 'default' : 'ops' }] }
      }

      return {}
    })

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(0)
    calls.length = 0

    failB = true
    await vi.advanceTimersByTimeAsync(60_000)

    const pushedToA = calls.find(call => call.method === 'bot_relay.roster.sync' && call.connectionId === 'a')

    expect(pushedToA?.params.agents).toEqual([expect.objectContaining({ profile: 'ops' })])

    stopBotRelay()
  })

  it('drops the cached rows of a connection that genuinely disconnected', async () => {
    const calls = respondWith(call =>
      call.method === 'profiles.list' ? { profiles: [{ name: call.connectionId }] } : {}
    )

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(0)

    // 'b' leaves the registry and 'c' arrives: b's agents must not linger.
    hostMock.profileRoutes = vi.fn(async () => [route('a'), route('c')])
    calls.length = 0
    await vi.advanceTimersByTimeAsync(60_000)

    const pushedToA = calls.find(call => call.method === 'bot_relay.roster.sync' && call.connectionId === 'a')

    expect(pushedToA?.params.agents).toEqual([expect.objectContaining({ profile: 'c' })])

    stopBotRelay()
  })

  it('stays quiet with a single connection — there is no peer to relay to', async () => {
    hostMock.profileRoutes = vi.fn(async () => [route('a')])

    const calls = respondWith(() => ({}))
    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(0)

    expect(calls).toHaveLength(0)

    stopBotRelay()
  })
})

describe('the drain loop wires drain → deliver → reply', () => {
  const envelope = {
    id: 'env-1',
    message: 'status?',
    target_connection: 'b',
    target_profile: 'ops'
  }

  const typedEnvelope = (id: string, targetConnection: string, targetProfile: string, targetHandle: string) => ({
    schema: 'asm-hermes-a2a-envelope/v2',
    id,
    message_id: id,
    idempotency_key: `mission:relay:${id}`,
    type: 'REQUEST',
    from_agent: 'hermes',
    to_agent: targetHandle,
    target_connection: targetConnection,
    target_profile: targetProfile,
    target_handle: targetHandle,
    message: `status for ${targetHandle}`,
    scope: { mutation: 'none', production: 'none' },
    expires_at: Date.now() / 1000 + 60,
    authority_effect: 'none'
  })

  const targetReceipt = (
    messageId: string,
    targetConnection: string,
    targetProfile: string,
    targetHandle: string,
    reply = `reply from ${targetHandle}`
  ) => ({
    schema: 'asm-hermes-a2a-target-receipt/v1',
    status: 'completed',
    idempotency_sha256: '1'.repeat(64),
    message_id: messageId,
    delivery_sha256: '2'.repeat(64),
    target_sha256: '3'.repeat(64),
    target_connection: targetConnection,
    target_profile: targetProfile,
    target_handle: targetHandle,
    started_at: '2026-09-04T20:00:00+00:00',
    completed_at: '2026-09-04T20:00:01+00:00',
    reply_sha256: createHash('sha256').update(reply, 'utf8').digest('hex')
  })

  it.each([
    ['a', 'b', 'ops', 'ops', 'a-to-b'],
    ['b', 'a', 'default', 'hermes', 'b-to-a']
  ])('requires a readback receipt in the %s direction', async (senderId, targetId, targetProfile, targetHandle, id) => {
    const currentEnvelope = typedEnvelope(`${'a'.repeat(31)}${id.slice(-1)}`, targetId, targetProfile, targetHandle)
    const receipt = targetReceipt(currentEnvelope.message_id, targetId, targetProfile, targetHandle)
    const calls = respondWith(call => {
      if (call.method === 'bot_relay.outbox.drain') {
        return { envelopes: call.connectionId === senderId ? [currentEnvelope] : [] }
      }

      if (call.method === 'bot_relay.deliver') {
        return { reply: `reply from ${targetHandle}`, target_receipt: receipt }
      }

      if (call.method === 'bot_relay.receipt.read') {
        return { receipt }
      }

      return {}
    })

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await pushAndSettle()

    expect(calls.find(call => call.method === 'bot_relay.deliver')).toMatchObject({
      connectionId: targetId,
      params: {
        message: currentEnvelope.message,
        profile: targetProfile,
        envelope: currentEnvelope
      }
    })
    expect(calls.find(call => call.method === 'bot_relay.receipt.read')).toMatchObject({
      connectionId: targetId,
      params: {
        message_id: currentEnvelope.message_id,
        idempotency_key: currentEnvelope.idempotency_key,
        envelope: currentEnvelope
      }
    })
    expect(calls.find(call => call.method === 'bot_relay.reply')).toMatchObject({
      connectionId: senderId,
      params: {
        id: currentEnvelope.id,
        reply: `reply from ${targetHandle}`,
        target_receipt: receipt
      }
    })

    stopBotRelay()
  })

  it('does not post structured success when target readback is missing or changed', async () => {
    const currentEnvelope = typedEnvelope(`${'b'.repeat(32)}`, 'b', 'ops', 'ops')
    const receipt = targetReceipt(currentEnvelope.message_id, 'b', 'ops', 'ops', 'unverified reply')
    let readback: unknown = {}
    const calls = respondWith(call => {
      if (call.method === 'bot_relay.outbox.drain') {
        return { envelopes: call.connectionId === 'a' ? [currentEnvelope] : [] }
      }

      if (call.method === 'bot_relay.deliver') {
        return { reply: 'unverified reply', target_receipt: receipt }
      }

      if (call.method === 'bot_relay.receipt.read') {
        return { receipt: readback }
      }

      return {}
    })

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await pushAndSettle()

    expect(calls.find(call => call.method === 'bot_relay.reply')?.params).toMatchObject({
      id: currentEnvelope.id,
      reason: 'target_receipt_unverified'
    })
    expect(clearBotAttentionMock).not.toHaveBeenCalledWith('b::ops')

    calls.length = 0
    readback = { ...receipt, target_connection: 'other' }
    await pushAndSettle()

    expect(calls.find(call => call.method === 'bot_relay.reply')?.params).toMatchObject({
      id: currentEnvelope.id,
      reason: 'target_receipt_mismatch'
    })

    stopBotRelay()
  })

  it('rejects a structured receipt whose reply digest does not match the returned reply', async () => {
    const currentEnvelope = typedEnvelope(`${'c'.repeat(32)}`, 'b', 'ops', 'ops')
    const receipt = targetReceipt(currentEnvelope.message_id, 'b', 'ops', 'ops')
    const calls = respondWith(call => {
      if (call.method === 'bot_relay.outbox.drain') {
        return { envelopes: call.connectionId === 'a' ? [currentEnvelope] : [] }
      }

      if (call.method === 'bot_relay.deliver') {
        return {
          reply: 'tampered reply',
          target_receipt: { ...receipt, reply_sha256: '0'.repeat(64) }
        }
      }

      return {}
    })

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await pushAndSettle()

    expect(calls.some(call => call.method === 'bot_relay.receipt.read')).toBe(false)
    expect(calls.find(call => call.method === 'bot_relay.reply')?.params).toMatchObject({
      id: currentEnvelope.id,
      reason: 'target_receipt_mismatch'
    })

    stopBotRelay()
  })

  it('delivers on the target’s own socket and posts the reply to the sender', async () => {
    const calls = respondWith(call => {
      if (call.method === 'bot_relay.outbox.drain') {
        return { envelopes: call.connectionId === 'a' ? [envelope] : [] }
      }

      if (call.method === 'bot_relay.deliver') {
        return { reply: 'all green' }
      }

      return {}
    })

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await pushAndSettle()

    expect(calls.find(call => call.method === 'bot_relay.deliver')).toMatchObject({
      connectionId: 'b',
      params: { message: 'status?', profile: 'ops' }
    })
    expect(calls.find(call => call.method === 'bot_relay.reply')).toMatchObject({
      connectionId: 'a',
      params: { id: 'env-1', reply: 'all green' }
    })
    // A delivered background DM is this bot's "good turn".
    expect(clearBotAttentionMock).toHaveBeenCalledWith('b::ops')

    stopBotRelay()
  })

  it('still posts a reply when the target connection is gone — the waiter must never dangle', async () => {
    const calls = respondWith(call =>
      call.method === 'bot_relay.outbox.drain'
        ? { envelopes: call.connectionId === 'a' ? [{ ...envelope, target_connection: 'ghost' }] : [] }
        : {}
    )

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await pushAndSettle()
    await vi.advanceTimersByTimeAsync(RELAY_ROUTE_RECONNECT_GRACE_MS + 1000)

    expect(calls.some(call => call.method === 'bot_relay.deliver')).toBe(false)
    expect(calls.find(call => call.method === 'bot_relay.reply')?.params.error).toMatch(
      /'ghost' is not connected to this Desktop right now/
    )

    stopBotRelay()
  })

  it('re-acquires a target route that reconnects after the envelope was claimed', async () => {
    const calls = respondWith(call => {
      if (call.method === 'bot_relay.outbox.drain') {
        return { envelopes: call.connectionId === 'a' ? [envelope] : [] }
      }

      if (call.method === 'bot_relay.deliver') {
        return { reply: 'reconnected' }
      }

      return {}
    })

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(0)
    calls.length = 0

    // The drain sees only the sender. The first bounded re-read sees the
    // target return and must deliver the already-claimed envelope exactly once.
    hostMock.profileRoutes = vi
      .fn()
      .mockResolvedValueOnce([route('a'), route('c')])
      .mockResolvedValue([route('a'), route('b'), route('c')])

    await pushAndSettle()
    await vi.advanceTimersByTimeAsync(1000)

    expect(hostMock.warmAgent).toHaveBeenCalledWith('b', 'ops')
    expect(calls.filter(call => call.method === 'bot_relay.deliver')).toEqual([
      expect.objectContaining({
        connectionId: 'b',
        params: expect.objectContaining({ message: 'status?', profile: 'ops' })
      })
    ])
    expect(calls.find(call => call.method === 'bot_relay.reply')?.params).toMatchObject({
      id: 'env-1',
      reply: 'reconnected'
    })

    stopBotRelay()
  })

  it('synthesizes a credential-free route for a registered SSH target whose seed is absent', async () => {
    hostMock.connections = vi.fn(async () => [{ id: 'a', kind: 'local' }, { id: 'b', kind: 'ssh' }])
    hostMock.profileRoutes = vi.fn(async () => [route('a'), route('c')])

    const calls = respondWith(call => {
      if (call.method === 'bot_relay.outbox.drain') {
        return { envelopes: call.connectionId === 'a' ? [envelope] : [] }
      }

      if (call.method === 'bot_relay.deliver') {
        return { reply: 'lazy dial complete' }
      }

      return {}
    })

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(0)
    calls.length = 0
    await pushAndSettle()

    expect(hostMock.warmAgent).toHaveBeenCalledWith('b', 'default')
    expect(calls.find(call => call.method === 'bot_relay.deliver')).toMatchObject({
      connectionId: 'b',
      params: { message: 'status?', profile: 'ops' }
    })
    expect(calls.find(call => call.method === 'bot_relay.reply')?.params).toMatchObject({
      id: 'env-1',
      reply: 'lazy dial complete'
    })

    stopBotRelay()
  })

  it('attempts the registered route after bounded retention recovery fails', async () => {
    hostMock.connections = vi.fn(async () => [{ id: 'a', kind: 'local' }, { id: 'b', kind: 'ssh' }])
    hostMock.profileRoutes = vi.fn(async () => [route('a'), route('c')])
    hostMock.retainProfile = vi.fn(async () => {
      throw new Error('renderer route is still reconciling')
    })

    const calls = respondWith(call => {
      if (call.method === 'bot_relay.outbox.drain') {
        return { envelopes: call.connectionId === 'a' ? [envelope] : [] }
      }

      if (call.method === 'bot_relay.deliver') {
        return { reply: 'direct lazy dial complete' }
      }

      return {}
    })

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(0)
    calls.length = 0
    const drain = pushAndSettle()
    await vi.advanceTimersByTimeAsync(RELAY_ROUTE_RECONNECT_GRACE_MS + 1000)
    await drain

    expect(calls.find(call => call.method === 'bot_relay.deliver')).toMatchObject({
      connectionId: 'b',
      params: { message: 'status?', profile: 'ops' }
    })
    expect(calls.find(call => call.method === 'bot_relay.reply')?.params).toMatchObject({
      id: 'env-1',
      reply: 'direct lazy dial complete'
    })

    stopBotRelay()
  })

  it('recovers a target from union source identity while registry reconciliation lags', async () => {
    hostMock.connections = vi.fn(async () => [{ id: 'a', kind: 'local' }])
    hostMock.agents = vi.fn(async () => ({
      agents: [],
      sources: [{ connectionId: 'b', kind: 'ssh', label: 'M5', reachable: true }]
    }))
    hostMock.profileRoutes = vi.fn(async () => [route('a'), route('c')])

    const calls = respondWith(call => {
      if (call.method === 'bot_relay.outbox.drain') {
        return { envelopes: call.connectionId === 'a' ? [envelope] : [] }
      }

      if (call.method === 'bot_relay.deliver') {
        return { reply: 'union identity dial complete' }
      }

      return {}
    })

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(0)
    calls.length = 0
    await pushAndSettle()

    expect(calls.find(call => call.method === 'bot_relay.deliver')).toMatchObject({
      connectionId: 'b',
      params: { message: 'status?', profile: 'ops' }
    })

    stopBotRelay()
  })

  it('waits for the registered target warm dial before using its synthesized route', async () => {
    let finishWarm!: () => void

    const warmPending = new Promise<void>(resolve => {
      finishWarm = resolve
    })

    hostMock.connections = vi.fn(async () => [{ id: 'a', kind: 'local' }, { id: 'b', kind: 'ssh' }])
    hostMock.profileRoutes = vi.fn(async () => [route('a'), route('c')])
    hostMock.warmAgent = vi.fn(() => warmPending)

    const calls = respondWith(call => {
      if (call.method === 'bot_relay.outbox.drain') {
        return { envelopes: call.connectionId === 'a' ? [envelope] : [] }
      }

      if (call.method === 'bot_relay.deliver') {
        return { reply: 'dial was ready' }
      }

      return {}
    })

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(0)
    calls.length = 0
    const drain = pushAndSettle()
    await vi.advanceTimersByTimeAsync(100)

    expect(calls.some(call => call.method === 'bot_relay.deliver')).toBe(false)

    finishWarm()
    await drain

    expect(calls.find(call => call.method === 'bot_relay.deliver')).toMatchObject({
      connectionId: 'b',
      params: { message: 'status?', profile: 'ops' }
    })

    stopBotRelay()
  })

  it('forwards the gateway’s typed failure reason to both the reply and the badge', async () => {
    // #93091: bot_relay.deliver classifies the failed turn and ships the code
    // in `data.reason`; a classified code beats re-parsing free text.
    const failure = Object.assign(new Error('401 invalid x-api-key'), {
      data: { reason: 'provider_auth_or_access' }
    })

    const calls = respondWith(call => {
      if (call.method === 'bot_relay.outbox.drain') {
        return { envelopes: call.connectionId === 'a' ? [envelope] : [] }
      }

      if (call.method === 'bot_relay.deliver') {
        throw failure
      }

      return {}
    })

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await pushAndSettle()

    expect(calls.find(call => call.method === 'bot_relay.reply')?.params).toMatchObject({
      error: '401 invalid x-api-key',
      id: 'env-1',
      reason: 'provider_auth_or_access'
    })
    expect(noteBotAttentionMock).toHaveBeenCalledWith('b::ops', 'provider_auth_or_access')

    stopBotRelay()
  })

  it('skips an older backend that rejects the drain RPC, and keeps going', async () => {
    const calls = respondWith(call => {
      if (call.method === 'bot_relay.outbox.drain') {
        if (call.connectionId === 'a') {
          throw new Error('unknown method')
        }

        return { envelopes: [] }
      }

      return {}
    })

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await pushAndSettle()

    expect(calls.filter(call => call.method === 'bot_relay.outbox.drain').map(call => call.connectionId)).toEqual([
      'a',
      'b'
    ])

    stopBotRelay()
  })
})

describe('stop halts both loops', () => {
  it('leaves no timer able to reach the gateway after teardown', async () => {
    const calls = respondWith(() => ({ envelopes: [] }))
    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await vi.advanceTimersByTimeAsync(0)
    stopBotRelay()
    calls.length = 0

    await vi.advanceTimersByTimeAsync(5 * 60_000)

    expect(calls).toHaveLength(0)
  })

  it('does not leak a mid-drain rerun into the next start', async () => {
    let release!: () => void

    const held = new Promise<void>(resolve => {
      release = resolve
    })

    let drains = 0

    respondWith(async call => {
      if (call.method === 'bot_relay.outbox.drain') {
        drains += 1

        if (drains === 1) {
          await held
        }
      }

      return { envelopes: [] }
    })

    const { startBotRelay, stopBotRelay } = await loadRelay()

    startBotRelay()
    await pushAndSettle()
    await pushAndSettle()
    stopBotRelay()
    release()
    await vi.advanceTimersByTimeAsync(RELAY_PUSH_DEBOUNCE_MS + 10)

    const afterStop = drains

    startBotRelay()
    await vi.advanceTimersByTimeAsync(RELAY_PUSH_DEBOUNCE_MS + 10)

    expect(drains).toBe(afterStop)

    stopBotRelay()
  })
})
