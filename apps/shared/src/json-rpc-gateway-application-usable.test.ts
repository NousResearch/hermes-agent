/**
 * #C3 (t_406dbcb3): whenApplicationUsable — the wait-for-usable seam that
 * post-connect deferred work rides (the deferred list refreshes gate on it).
 *
 * The contract mirrors C1's phase-emission condition: first server RESPONSE
 * frame + every replay barrier settled. These tests pin the seam itself; the
 * hook-level wiring is proven in apps/desktop's use-gateway-boot-deferred-refresh.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { JsonRpcGatewayClient } from './json-rpc-gateway'

class FakeWebSocket extends EventTarget {
  static instances: FakeWebSocket[] = []

  readyState = 0
  sent: string[] = []

  constructor(_url: string) {
    super()
    FakeWebSocket.instances.push(this)
  }

  send(data: string): void {
    this.sent.push(data)
  }

  close(): void {
    this.readyState = 3
    this.dispatchEvent(new CloseEvent('close'))
  }

  open(): void {
    this.readyState = 1
    this.dispatchEvent(new Event('open'))
  }

  serverFrame(obj: unknown): void {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(obj) }))
  }

  /** Respond to the last sent frame carrying `method`; fails the test if absent. */
  answer(method: string, result: unknown): void {
    const raw = [...this.sent].reverse().find(entry => {
      try {
        return (JSON.parse(entry) as { method?: string }).method === method
      } catch {
        return false
      }
    })

    if (!raw) {
      throw new Error(`no frame with method ${method} was sent`)
    }

    const { id } = JSON.parse(raw) as { id: number | string }
    this.serverFrame({ jsonrpc: '2.0', id, result })
  }
}

const makeClient = () =>
  new JsonRpcGatewayClient({
    socketFactory: url => new FakeWebSocket(url) as unknown as WebSocket,
    heartbeatIntervalMs: 0,
    heartbeatDeadlineMs: 0,
    connectTimeoutMs: 1000
  })

const settle = () => new Promise(resolve => setTimeout(resolve, 0))

describe('JsonRpcGatewayClient.whenApplicationUsable (#C3)', () => {
  beforeEach(() => {
    FakeWebSocket.instances = []
    vi.useFakeTimers({ toFake: ['Date'] })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('waits for the first response frame, not just the open event', async () => {
    const client = makeClient()
    const connected = client.connect('ws://x')

    sockets()[0].open()
    await connected

    // gateway.ready is an EVENT frame: the composer gate is the first RESPONSE.
    sockets()[0].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'gateway.ready', payload: {} } })

    let resolved = false

    void client.whenApplicationUsable().then(() => {
      resolved = true
    })

    await settle()
    expect(resolved).toBe(false)

    // The first RESPONSE frame (a matched id) is the usable stamp — an event
    // frame is not one. (Desktop's boot rides `client.capabilities`; any RPC
    // answers the same contract, so the test uses its own.)
    void client.request('test.echo', {})
    sockets()[0].answer('test.echo', { pong: true })
    await settle()
    expect(resolved).toBe(true)
  })

  it('waits behind an outstanding replay barrier even after usable', async () => {
    const client = makeClient()
    const first = client.connect('ws://x')

    sockets()[0].open()
    await first
    sockets()[0].serverFrame({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'message.delta', session_id: 's1', seq: 1, payload: {} }
    })

    // The socket dies with a watermark held; the reconnect opens a replay barrier.
    sockets()[0].close()
    const second = client.connect('ws://x')

    sockets()[1].open()
    await second

    let resolved = false

    void client.whenApplicationUsable().then(() => {
      resolved = true
    })

    void client.request('test.echo', {})
    sockets()[1].answer('test.echo', { pong: true })
    await settle()
    // Usable arrived, but session.events.since is still outstanding: the
    // transcript's parked reads have not flushed — do not call it usable.
    expect(resolved).toBe(false)

    sockets()[1].answer('session.events.since', { events: [], latest_seq: 1 })
    await settle()
    expect(resolved).toBe(true)
  })

  it('resolves immediately when the generation already reached usable', async () => {
    const client = makeClient()
    const connected = client.connect('ws://x')

    sockets()[0].open()
    await connected
    void client.request('test.echo', {})
    sockets()[0].answer('test.echo', { pong: true })
    await settle()

    await expect(client.whenApplicationUsable()).resolves.toBeUndefined()
  })
})

const sockets = (): FakeWebSocket[] => FakeWebSocket.instances
