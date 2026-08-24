import { beforeEach, describe, expect, it, vi } from 'vitest'

import { JsonRpcGatewayClient } from './json-rpc-gateway'

/**
 * Minimal EventTarget-based WebSocket stand-in so the seq-tracking and
 * replay-resume logic can be driven with real dispatch semantics.
 */
class FakeWebSocket extends EventTarget {
  static OPEN = 1
  static instances: FakeWebSocket[] = []

  readyState = 0
  sent: string[] = []
  url: string

  constructor(url: string) {
    super()
    this.url = url
    FakeWebSocket.instances.push(this)
  }

  send(data: string): void {
    this.sent.push(data)
  }

  close(): void {
    this.readyState = 3
    this.dispatchEvent(new CloseEvent('close'))
  }

  // Test drivers
  open(): void {
    this.readyState = 1
    this.dispatchEvent(new Event('open'))
  }

  serverFrame(obj: unknown): void {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(obj) }))
  }

  lastRequest(): { id: string; method: string; params: Record<string, unknown> } {
    const last = this.sent[this.sent.length - 1]

    return JSON.parse(last ?? '{}')
  }
}

let sockets: FakeWebSocket[]

const makeClient = () => {
  const client = new JsonRpcGatewayClient({
    socketFactory: url => new FakeWebSocket(url) as unknown as WebSocket,
    heartbeatIntervalMs: 0,
    heartbeatDeadlineMs: 0,
    connectTimeoutMs: 1000
  })

  return client
}

describe('JsonRpcGatewayClient event-seq tracking + replay resume', () => {
  beforeEach(() => {
    FakeWebSocket.instances = []
    sockets = FakeWebSocket.instances as unknown as FakeWebSocket[]
  })

  it('records per-session seq watermarks from live events', async () => {
    const client = makeClient()
    const p = client.connect('ws://x')
    sockets[0].open()
    await p

    sockets[0].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 4 } })
    sockets[0].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 2 } }) // out of order / late
    sockets[0].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'tool.start', session_id: 's2', seq: 9 } })
    sockets[0].serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'skin.changed' } }) // no sid/seq

    expect(client.getSeqWatermarks()).toEqual({ s1: 4, s2: 9 })
    client.close()
  })

  it('fetches replay on reconnect for sessions it has watermarks for', async () => {
    const client = makeClient()

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first

    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.start', session_id: 's1', seq: 1 } })
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 5 } })

    // Drop and reconnect.
    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second

    // The reconnect triggered a replay fetch — flush microtasks.
    await vi.waitFor(() => {
      const req = sock.lastRequest()
      expect(req.method).toBe('session.events.since')
      expect(req.params).toMatchObject({ session_id: 's1', last_seen: 5 })
    })

    client.close()
  })

  it('dispatches replayed events through the normal handler path', async () => {
    const client = makeClient()
    const seen: string[] = []
    client.on('tool.complete', e => seen.push(`live:${String((e.payload as { n?: number }).n)}`))

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 3 } })

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second

    await vi.waitFor(async () => {
      const req = sock.lastRequest()
      expect(req.method).toBe('session.events.since')
      // Answer the replay request with two missed events.
      sock.serverFrame({
        jsonrpc: '2.0',
        id: req.id,
        result: {
          events: [
            { type: 'tool.complete', session_id: 's1', seq: 4, payload: { n: 1 } },
            { type: 'tool.complete', session_id: 's1', seq: 5, payload: { n: 2 } }
          ],
          latest_seq: 5,
          truncated: false,
          count: 2
        }
      })
      await Promise.resolve()
      expect(seen).toEqual(['live:1', 'live:2'])
    })

    expect(client.getSeqWatermarks().s1).toBe(5)
    client.close()
  })

  it('does not attempt replay when nothing was ever observed', async () => {
    const client = makeClient()
    const p = client.connect('ws://x')
    sockets[0].open()
    await p
    // No events ever seen → close+reconnect must NOT fire a replay RPC.
    client.invalidate('drop')
    const p2 = client.connect('ws://x')
    sockets[sockets.length - 1].open()
    await p2
    await new Promise(r => setTimeout(r, 20))

    expect(sockets[sockets.length - 1].sent).toHaveLength(0)
    client.close()
  })

  it('replayed seqs advance watermarks but never regress them', async () => {
    const client = makeClient()
    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'status.update', session_id: 's1', seq: 10 } })

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second

    await vi.waitFor(() => {
      expect(sock.lastRequest().method).toBe('session.events.since')
    })
    // Replay returns a STALE frame (seq 2 < watermark 10): watermark must hold.
    const req = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: req.id,
      result: { events: [{ type: 'status.update', session_id: 's1', seq: 2 }], latest_seq: 10, truncated: false, count: 1 }
    })
    await Promise.resolve()
    expect(client.getSeqWatermarks().s1).toBe(10)
    client.close()
  })
})
