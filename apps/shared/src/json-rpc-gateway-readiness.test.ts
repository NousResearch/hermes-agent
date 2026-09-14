import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { GatewayEvent } from './gateway-events'
import { JsonRpcGatewayClient } from './json-rpc-gateway'

class ReplaySocket extends EventTarget {
  static OPEN = 1
  readyState = 0
  requests: Array<{ id: string; method: string; params?: Record<string, unknown> }> = []

  send(data: string) {
    this.requests.push(JSON.parse(data))
  }

  open() {
    this.readyState = 1
    this.dispatchEvent(new Event('open'))
  }

  close() {
    this.readyState = 3
    this.dispatchEvent(new CloseEvent('close'))
  }

  frame(frame: unknown) {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(frame) }))
  }

  event(seq: number, type = 'message.delta') {
    this.frame({ method: 'event', params: { type, session_id: 'running', seq } })
  }

  reply(result: unknown) {
    this.frame({ id: this.requests[0].id, result })
  }
}

let client: JsonRpcGatewayClient
let sockets: ReplaySocket[]

async function connect() {
  const pending = client.connect('wss://replay.invalid/api/ws')
  const socket = sockets[sockets.length - 1]
  socket.open()
  await pending

  return socket
}

beforeEach(() => {
  vi.useFakeTimers()
  sockets = []
  client = new JsonRpcGatewayClient({
    heartbeatIntervalMs: 0,
    socketFactory: () => {
      const socket = new ReplaySocket()
      sockets.push(socket)

      return socket as unknown as WebSocket
    }
  })
})

afterEach(() => {
  client.close()
  vi.clearAllTimers()
  vi.useRealTimers()
})

describe('replay readiness belongs to the socket generation', () => {
  it.each(['reply-only', 'ready-before-reply'] as const)('rejects a changed watermark epoch: %s', async order => {
    const first = await connect()
    first.frame({ method: 'event', params: { type: 'gateway.ready', payload: { replay_epoch: 'epoch-A' } } })
    first.event(97, 'message.start')
    client.invalidate()
    const second = await connect()
    expect(second.requests[0].params).toEqual({ session_id: 'running', last_seen: 97, replay_epoch: 'epoch-A' })

    const outcome = client.waitForReplay().then(
      () => 'ready',
      () => 'not-ready'
    )

    if (order === 'ready-before-reply') {
      second.frame({ method: 'event', params: { type: 'gateway.ready', payload: { replay_epoch: 'epoch-B' } } })
    }

    second.reply({ events: [], latest_seq: 0, truncated: false, count: 0, epoch: 'epoch-B' })
    expect(await outcome).toBe('not-ready')
    expect(client.getSeqWatermarks()).toEqual({})
  })

  it('does not roll a new announced epoch back to an older in-flight reply', async () => {
    const first = await connect()
    first.frame({ method: 'event', params: { type: 'gateway.ready', payload: { replay_epoch: 'epoch-A' } } })
    first.event(97, 'message.start')
    client.invalidate()
    const second = await connect()

    const outcome = client.waitForReplay().then(
      () => 'ready',
      () => 'not-ready'
    )

    second.frame({ method: 'event', params: { type: 'gateway.ready', payload: { replay_epoch: 'epoch-B' } } })
    second.event(1, 'message.start')
    const staleIdle = vi.fn()
    client.onEvent(event => {
      if (event.type === 'session.info') {
        staleIdle(event)
      }
    })
    second.reply({
      events: [{ type: 'session.info', session_id: 'running', seq: 98, payload: { running: false } }],
      epoch: 'epoch-A'
    })
    expect(await outcome).toBe('not-ready')
    expect(staleIdle).not.toHaveBeenCalled()
    expect(client.getSeqWatermarks()).toEqual({ running: 1 })
    client.invalidate()
    const third = await connect()
    expect(third.requests[0].params).toEqual({ session_id: 'running', last_seen: 1, replay_epoch: 'epoch-B' })
    third.reply({ events: [], epoch: 'epoch-B' })
    await expect(client.waitForReplay()).resolves.toBeUndefined()
  })

  it('accepts same-epoch catch-up when ready arrives before the reply', async () => {
    const first = await connect()
    first.frame({ method: 'event', params: { type: 'gateway.ready', payload: { replay_epoch: 'epoch-A' } } })
    first.event(97, 'message.start')
    client.invalidate()
    const second = await connect()
    second.frame({ method: 'event', params: { type: 'gateway.ready', payload: { replay_epoch: 'epoch-A' } } })
    second.reply({ events: [], epoch: 'epoch-A' })
    await expect(client.waitForReplay()).resolves.toBeUndefined()
    expect(client.getSeqWatermarks()).toEqual({ running: 97 })
  })

  it('publishes the barrier before open listeners and drains held live frames before releasing it', async () => {
    const first = await connect()
    await expect(client.waitForReplay()).resolves.toBeUndefined()
    first.event(1, 'message.start')
    client.invalidate()
    const events: number[] = []
    client.onEvent(event => events.push((event as GatewayEvent & { seq: number }).seq))
    let ready = false

    const off = client.onState(state => {
      if (state === 'open') {
        void client.waitForReplay().then(() => {
          ready = true
        })
      }
    })

    const second = await connect()
    second.event(3)
    await vi.advanceTimersByTimeAsync(1000)
    expect(ready).toBe(false)
    expect(events).toEqual([])
    second.reply({ events: [{ type: 'message.delta', session_id: 'running', seq: 2 }] })
    await client.waitForReplay()
    expect(ready).toBe(true)
    expect(events).toEqual([2, 3])
    off()
  })

  it('does not let an older replay dispatch or clear the new generation hold', async () => {
    const first = await connect()
    first.event(1, 'message.start')
    client.invalidate()
    const old = await connect()

    const oldOutcome = client.waitForReplay().then(
      () => 'ready',
      () => 'superseded'
    )

    old.event(2)
    client.invalidate()
    const replacement = await connect()
    const seen: number[] = []
    client.onEvent(event => seen.push((event as GatewayEvent & { seq: number }).seq))
    replacement.event(4)
    old.reply({ events: [{ type: 'session.info', session_id: 'running', seq: 3, payload: { running: false } }] })
    await Promise.resolve()
    expect(seen).toEqual([])
    expect(replacement.requests).toHaveLength(1)
    replacement.reply({ events: [{ type: 'message.delta', session_id: 'running', seq: 3 }] })
    await client.waitForReplay()
    expect(await oldOutcome).toBe('superseded')
    expect(seen).toEqual([3, 4])
    expect(client.getSeqWatermarks()).toEqual({ running: 4 })
  })

  it.each(['timeout', 'error', 'truncated'] as const)(
    'does not call %s replay readiness or synthesize settlement',
    async failure => {
      const first = await connect()
      first.event(1, 'message.start')
      client.invalidate()
      const second = await connect()
      const onEvent = vi.fn()
      client.onEvent(onEvent)

      const outcome = client.waitForReplay().then(
        () => 'ready',
        () => 'not-ready'
      )

      if (failure === 'timeout') {
        await vi.advanceTimersByTimeAsync(10001)
      } else if (failure === 'error') {
        second.frame({ id: second.requests[0].id, error: { code: -32601, message: 'unavailable' } })
      } else {
        second.reply({ events: [], truncated: true })
      }

      expect(await outcome).toBe('not-ready')

      if (failure === 'truncated') {
        expect(onEvent.mock.calls.map(([event]) => event.type)).toEqual(['session.replay_gap'])
        expect(client.getSeqWatermarks()).toEqual({})
      } else {
        expect(onEvent).not.toHaveBeenCalled()
      }

      expect(client.connectionState).toBe('open')
      // A real event remains authoritative even when lossless catch-up failed.
      onEvent.mockClear()
      second.event(2, 'session.info')
      expect(onEvent).toHaveBeenCalledOnce()
    }
  )

  it.each(['truncated', 'snapshot_required', 'epoch-change'] as const)(
    'hands a canonical %s gap to snapshot consumers without resetting another session',
    async reason => {
      const first = await connect()

      for (const [session_id, replay_epoch, seq] of [
        ['a', 'a-old', 41],
        ['b', 'b-stable', 7]
      ] as const) {
        first.frame({ method: 'event', params: { type: 'message.delta', session_id, replay_epoch, seq } })
      }

      client.invalidate()
      const second = await connect()
      const onEvent = vi.fn()
      client.onEvent(onEvent)
      expect(second.requests.map(request => request.params)).toEqual([
        { session_id: 'a', last_seen: 41, replay_epoch: 'a-old' },
        { session_id: 'b', last_seen: 7, replay_epoch: 'b-stable' }
      ])
      second.frame({
        id: second.requests[0].id,
        result: {
          events: [{ type: 'message.delta', session_id: 'a', seq: 42 }],
          replay_epoch: 'a-new',
          latest_seq: 1,
          ...(reason === 'epoch-change' ? {} : { [reason]: true })
        }
      })
      second.frame({
        id: second.requests[1].id,
        result: {
          events: [{ type: 'message.delta', session_id: 'b', replay_epoch: 'b-stable', seq: 8 }],
          replay_epoch: 'b-stable'
        }
      })
      // This only proves dispatch. No snapshot consumer completed a resume,
      // and no turn admission/settlement is asserted by this barrier.
      await expect(client.waitForReplay()).resolves.toBeUndefined()
      expect(onEvent.mock.calls.map(([event]) => [event.type, event.session_id])).toEqual([
        ['session.replay_gap', 'a'],
        ['message.delta', 'b']
      ])
      expect(client.getSeqWatermarks()).toEqual({ b: 8 })
    }
  )

  it('keeps canonical per-session replay epochs across a legacy process epoch change', async () => {
    const first = await connect()
    first.frame({ method: 'event', params: { type: 'gateway.ready', payload: { replay_epoch: 'process-old' } } })
    first.frame({
      method: 'event',
      params: { type: 'message.delta', session_id: 'a', replay_epoch: 'session-stable', seq: 41 }
    })
    client.invalidate()
    const second = await connect()
    const onEvent = vi.fn()
    client.onEvent(onEvent)
    second.frame({ method: 'event', params: { type: 'gateway.ready', payload: { replay_epoch: 'process-new' } } })
    expect(second.requests[0].params).toEqual({ session_id: 'a', last_seen: 41, replay_epoch: 'session-stable' })
    expect(client.getSeqWatermarks()).toEqual({ a: 41 })
    second.reply({ events: [], epoch: 'session-stable', replay_epoch: 'session-stable' })
    await expect(client.waitForReplay()).resolves.toBeUndefined()
    expect(client.getSeqWatermarks()).toEqual({ a: 41 })
    expect(onEvent.mock.calls.map(([event]) => event.type)).toEqual(['gateway.ready'])
  })
})
