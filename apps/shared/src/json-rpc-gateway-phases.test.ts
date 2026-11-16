import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { JsonRpcGatewayClient, type GatewaySocketPhases } from './json-rpc-gateway'

/** Openable/closable WebSocket stand-in (same shape as json-rpc-gateway-replay.test.ts). */
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

  lastRequest(): { id: string; method: string } {
    return JSON.parse(this.sent[this.sent.length - 1] ?? '{}')
  }
}

const sockets = (): FakeWebSocket[] => FakeWebSocket.instances as unknown as FakeWebSocket[]

const makeClient = (onSocketPhases?: (phases: GatewaySocketPhases) => void) =>
  new JsonRpcGatewayClient({
    socketFactory: url => new FakeWebSocket(url) as unknown as WebSocket,
    onSocketPhases,
    heartbeatIntervalMs: 0,
    heartbeatDeadlineMs: 0,
    connectTimeoutMs: 1000
  })

let phases: GatewaySocketPhases[]

beforeEach(() => {
  FakeWebSocket.instances = []
  phases = []
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(1_000)
})

afterEach(() => {
  vi.useRealTimers()
})

const settle = () => new Promise(resolve => setTimeout(resolve, 0))

describe('JsonRpcGatewayClient socket-phase observation (C1, t_c1f387b2)', () => {
  it('emits one complete record per healthy generation: dial, ready, usable', async () => {
    const client = makeClient(p => phases.push(p))
    const connected = client.connect('ws://x')

    vi.setSystemTime(1_040)
    sockets()[0].open()
    await connected

    vi.setSystemTime(1_055)
    sockets()[0].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'gateway.ready', payload: {} } })

    vi.setSystemTime(1_060)
    const pending = client.request('session.list', {})
    const req = sockets()[0].lastRequest()

    sockets()[0].serverFrame({ jsonrpc: '2.0', result: { sessions: [] }, id: req.id })
    await pending
    await settle()

    expect(phases).toHaveLength(1)
    expect(phases[0]).toEqual({ dial_ms: 40, ready_ms: 15, replay_barrier_ms: null, usable_ms: 20 })

    client.close()
  })

  it('waits behind the replay barrier: the single record carries usable AND replay_barrier', async () => {
    const client = makeClient(p => phases.push(p))
    let connected = client.connect('ws://x')

    sockets()[0].open()
    await connected
    sockets()[0].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'gateway.ready', payload: {} } })
    // A watermarked session: the NEXT generation must wait behind its replay.
    sockets()[0].serverFrame({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'message.delta', session_id: 's1', seq: 4 }
    })

    const pending = client.request('session.list', {})
    const req = sockets()[0].lastRequest()

    sockets()[0].serverFrame({ jsonrpc: '2.0', result: { sessions: [] }, id: req.id })
    await pending
    await settle()
    expect(phases).toHaveLength(1)
    expect(phases[0].usable_ms).not.toBeNull()
    expect(phases[0].replay_barrier_ms).toBeNull()

    // Drop: generation 2 opens and fetches the replay. Its response is BOTH the first
    // response (usable) and the barrier release; one emitted record carries both.
    sockets()[0].close()
    phases = []
    connected = client.connect('ws://x')
    sockets()[1].open()
    await connected

    const replayReq = sockets()[1].sent
      .map(line => JSON.parse(line))
      .find(frame => frame.method === 'session.events.since')

    expect(replayReq).toBeDefined()
    expect(phases).toHaveLength(0) // barrier outstanding — nothing emitted before its response

    sockets()[1].serverFrame({
      jsonrpc: '2.0',
      result: { events: [], latest_seq: 4, truncated: false, count: 0, epoch: 'e1', open_requests: [] },
      id: replayReq.id
    })
    await settle()
    await settle()

    expect(phases).toHaveLength(1)
    expect(phases[0].usable_ms).not.toBeNull()
    expect(phases[0].replay_barrier_ms).not.toBeNull()

    client.close()
  })

  it('a generation that dies mid-handshake reports what it reached, once', () => {
    const client = makeClient(p => phases.push(p))
    const connected = client.connect('ws://x')

    connected.catch(() => undefined)
    sockets()[0].open()
    vi.setSystemTime(1_010)
    sockets()[0].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'gateway.ready', payload: {} } })

    sockets()[0].close()
    sockets()[0].close() // a second close must not double-report

    expect(phases).toHaveLength(1)
    expect(phases[0]).toEqual({ dial_ms: 0, ready_ms: 10, replay_barrier_ms: null, usable_ms: null })
  })
})
