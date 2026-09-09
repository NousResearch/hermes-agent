import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type GatewayEvent, JsonRpcGatewayClient } from './json-rpc-gateway'

class ReplaySocket extends EventTarget {
  static OPEN = 1
  readyState = 0
  requests: Array<{ id: string; method: string }> = []

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
      expect(onEvent).not.toHaveBeenCalled()
      expect(client.connectionState).toBe('open')
      // A real event remains authoritative even when lossless catch-up failed.
      second.event(2, 'session.info')
      expect(onEvent).toHaveBeenCalledOnce()
    }
  )
})
