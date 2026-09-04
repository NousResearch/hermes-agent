// @vitest-environment jsdom

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

  it('rejects envelope-shaped replay elements (the #94219 server-shape bug)', async () => {
    const client = makeClient()
    const seen: string[] = []
    client.on('message.delta', () => seen.push('delta'))

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 1 } })
    expect(seen).toEqual(['delta']) // the pre-drop live frame

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second

    await vi.waitFor(() => {
      expect(sock.lastRequest().method).toBe('session.events.since')
    })
    const req = sock.lastRequest()
    // Pre-fix servers returned FULL JSON-RPC envelopes. The client must not
    // dispatch those blindly — and this documents why the server now sends
    // bare event objects.
    sock.serverFrame({
      jsonrpc: '2.0',
      id: req.id,
      result: {
        events: [{ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 2 } }],
        latest_seq: 2,
        truncated: false,
        count: 1
      }
    })
    await Promise.resolve()
    // Envelope-shaped replay elements must add nothing beyond the live frame.
    expect(seen).toEqual(['delta'])
    client.close()
  })

  it('holds live frames racing the replay fetch — no double dispatch, no skipped gap', async () => {
    const client = makeClient()
    const seen: number[] = []
    client.on('message.delta', e => seen.push((e as unknown as { seq: number }).seq))

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 2 } })
    expect(seen).toEqual([2]) // pre-drop live frame dispatches normally

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second

    await vi.waitFor(() => {
      expect(sock.lastRequest().method).toBe('session.events.since')
    })

    // LIVE frames 5 and 6 arrive while the replay (which carries 3,4,5) is
    // still in flight. They must be parked, not dispatched ahead of the gap.
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 5 } })
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 6 } })
    expect(seen).toEqual([2]) // still only the pre-drop frame — 5/6 parked

    const req = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: req.id,
      result: {
        events: [
          { type: 'message.delta', session_id: 's1', seq: 3 },
          { type: 'message.delta', session_id: 's1', seq: 4 },
          { type: 'message.delta', session_id: 's1', seq: 5 }
        ],
        latest_seq: 5,
        truncated: false,
        count: 3
      }
    })

    // In-order, exactly once: replayed 3,4,5 then the parked live 6 —
    // the parked duplicate of 5 is seq-gated out.
    await vi.waitFor(() => {
      expect(seen).toEqual([2, 3, 4, 5, 6])
    })
    expect(client.getSeqWatermarks().s1).toBe(6)
    client.close()
  })

  it('resyncs authoritative history instead of dispatching a truncated tail', async () => {
    const client = makeClient()
    const seen: number[] = []
    let finishResync!: () => void

    const resync = new Promise<void>(resolve => {
      finishResync = resolve
    })

    const gaps: string[] = []

    client.on('message.delta', e => seen.push((e as unknown as { seq: number }).seq))
    client.onReplayGap(async sid => {
      gaps.push(sid)
      await resync

      return true
    })

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'message.delta', session_id: 's1', seq: 2, replay_generation: 'gen-A' }
    })

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second
    await vi.waitFor(() => expect(sock.lastRequest().method).toBe('session.events.since'))

    sock.serverFrame({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'message.delta', session_id: 's1', seq: 6, replay_generation: 'gen-A' }
    })
    const req = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: req.id,
      result: {
        events: [
          { type: 'message.delta', session_id: 's1', seq: 4, replay_generation: 'gen-A' },
          { type: 'message.delta', session_id: 's1', seq: 5, replay_generation: 'gen-A' }
        ],
        latest_seq: 5,
        truncated: true,
        generation: 'gen-A'
      }
    })

    await vi.waitFor(() => expect(gaps).toEqual(['s1']))
    expect(seen).toEqual([2])
    finishResync()
    await vi.waitFor(() => expect(seen).toEqual([2, 6]))
    expect(client.getSeqWatermarks().s1).toBe(6)
    client.close()
  })

  it('accepts reset seqs after a session replay generation changes', async () => {
    const client = makeClient()
    const seen: number[] = []
    client.on('message.delta', e => seen.push((e as unknown as { seq: number }).seq))

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'message.delta', session_id: 's1', seq: 10, replay_generation: 'gen-A' }
    })

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second
    await vi.waitFor(() => expect(sock.lastRequest().method).toBe('session.events.since'))

    sock.serverFrame({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'message.delta', session_id: 's1', seq: 1, replay_generation: 'gen-B' }
    })
    const req = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: req.id,
      result: {
        events: [{ type: 'message.delta', session_id: 's1', seq: 1, replay_generation: 'gen-B' }],
        latest_seq: 1,
        truncated: false,
        generation: 'gen-B'
      }
    })

    await vi.waitFor(() => expect(sock.lastRequest().params.last_seen).toBe(0))
    const freshRequest = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: freshRequest.id,
      result: {
        events: [{ type: 'message.delta', session_id: 's1', seq: 1, replay_generation: 'gen-B' }],
        latest_seq: 1,
        truncated: false,
        generation: 'gen-B'
      }
    })

    await vi.waitFor(() => expect(seen).toEqual([10, 1]))
    expect(client.getSeqWatermarks().s1).toBe(1)
    client.close()
  })

  it('refetches a changed generation from zero before releasing later live frames', async () => {
    const client = makeClient()
    const seen: number[] = []
    client.on('message.delta', event => seen.push((event as unknown as { seq: number }).seq))

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'message.delta', session_id: 's1', seq: 10, replay_generation: 'gen-A' }
    })

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second
    await vi.waitFor(() => expect(sock.lastRequest().params).toMatchObject({ session_id: 's1', last_seen: 10 }))

    const staleRequest = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: staleRequest.id,
      result: { events: [], latest_seq: 2, truncated: false, generation: 'gen-B' }
    })

    await vi.waitFor(() => expect(sock.lastRequest().params).toMatchObject({ session_id: 's1', last_seen: 0 }))
    sock.serverFrame({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'message.delta', session_id: 's1', seq: 3, replay_generation: 'gen-B' }
    })

    const freshRequest = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: freshRequest.id,
      result: {
        events: [
          { type: 'message.delta', session_id: 's1', seq: 1, replay_generation: 'gen-B' },
          { type: 'message.delta', session_id: 's1', seq: 2, replay_generation: 'gen-B' }
        ],
        latest_seq: 2,
        truncated: false,
        generation: 'gen-B'
      }
    })

    await vi.waitFor(() => expect(seen).toEqual([10, 1, 2, 3]))
    expect(client.getSeqWatermarks().s1).toBe(3)
    client.close()
  })

  it('preserves the old generation checkpoint when zero-based recovery fails', async () => {
    const client = makeClient()
    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'message.delta', session_id: 's1', seq: 10, replay_generation: 'gen-A' }
    })

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second
    await vi.waitFor(() => expect(sock.lastRequest().params.last_seen).toBe(10))

    const staleRequest = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: staleRequest.id,
      result: { events: [], latest_seq: 2, truncated: true, generation: 'gen-B' }
    })
    await vi.waitFor(() => expect(sock.lastRequest().params.last_seen).toBe(0))

    const freshRequest = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: freshRequest.id,
      error: { code: -32000, message: 'replay unavailable' }
    })

    await vi.waitFor(() => expect(sock.readyState).toBe(3))
    expect(client.getSeqWatermarks().s1).toBe(10)

    const third = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await third
    await vi.waitFor(() => expect(sock.lastRequest().params).toMatchObject({ session_id: 's1', last_seen: 10 }))
    client.close()
  })

  it('retries without releasing parked frames when truncated history recovery fails', async () => {
    const client = makeClient()
    const seen: number[] = []
    client.on('message.delta', e => seen.push((e as unknown as { seq: number }).seq))
    client.onReplayGap(async () => false)

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 2 } })

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second
    await vi.waitFor(() => expect(sock.lastRequest().method).toBe('session.events.since'))
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 6 } })
    const req = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: req.id,
      result: { events: [], latest_seq: 5, truncated: true }
    })

    await vi.waitFor(() => expect(sock.readyState).toBe(3))
    expect(seen).toEqual([2])
    expect(client.getSeqWatermarks().s1).toBe(2)
    client.close()
  })

  it('keeps the old watermark when truncated replay has no history handler', async () => {
    const client = makeClient()
    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 2 } })

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second
    await vi.waitFor(() => expect(sock.lastRequest().method).toBe('session.events.since'))
    const req = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: req.id,
      result: { events: [], latest_seq: 5, truncated: true }
    })

    await vi.waitFor(() => expect(sock.readyState).toBe(3))
    expect(client.getSeqWatermarks().s1).toBe(2)
    client.close()
  })

  it('keeps the old watermark when truncated history recovery rejects', async () => {
    const client = makeClient()
    client.onReplayGap(async () => {
      throw new Error('history unavailable')
    })

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 2 } })

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second
    await vi.waitFor(() => expect(sock.lastRequest().method).toBe('session.events.since'))
    const req = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: req.id,
      result: { events: [], latest_seq: 5, truncated: true }
    })

    await vi.waitFor(() => expect(sock.readyState).toBe(3))
    expect(client.getSeqWatermarks().s1).toBe(2)
    client.close()
  })

  it('retries without releasing parked frames when the replay request rejects', async () => {
    const client = makeClient()
    const seen: number[] = []
    client.on('message.delta', e => seen.push((e as unknown as { seq: number }).seq))

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 2 } })

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second
    await vi.waitFor(() => expect(sock.lastRequest().method).toBe('session.events.since'))
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 6 } })
    const req = sock.lastRequest()
    sock.serverFrame({ jsonrpc: '2.0', id: req.id, error: { code: -32000, message: 'replay unavailable' } })

    await vi.waitFor(() => expect(sock.readyState).toBe(3))
    expect(seen).toEqual([2])
    expect(client.getSeqWatermarks().s1).toBe(2)
    client.close()
  })

  it('retries without releasing parked frames when a history handler throws synchronously', async () => {
    const client = makeClient()
    const seen: number[] = []
    client.on('message.delta', e => seen.push((e as unknown as { seq: number }).seq))
    client.onReplayGap(() => {
      throw new Error('history unavailable')
    })

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 2 } })

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second
    await vi.waitFor(() => expect(sock.lastRequest().method).toBe('session.events.since'))
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 6 } })
    const req = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: req.id,
      result: { events: [], latest_seq: 5, truncated: true }
    })

    await vi.waitFor(() => expect(sock.readyState).toBe(3))
    expect(seen).toEqual([2])
    expect(client.getSeqWatermarks().s1).toBe(2)
    client.close()
  })

  it('accepts reset seqs from a generation-less older backend', async () => {
    const client = makeClient()
    const seen: number[] = []
    client.on('message.delta', e => seen.push((e as unknown as { seq: number }).seq))

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 10 } })

    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second
    await vi.waitFor(() => expect(sock.lastRequest().method).toBe('session.events.since'))
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 1 } })
    const req = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: req.id,
      result: { events: [], latest_seq: 1, truncated: false }
    })

    await vi.waitFor(() => expect(seen).toEqual([10, 1]))
    expect(client.getSeqWatermarks().s1).toBe(1)
    client.close()
  })

  it('clears stale watermarks when the backend epoch changes (restart poisoning)', async () => {
    const client = makeClient()

    const first = client.connect('ws://x')
    let sock = sockets[sockets.length - 1]
    sock.open()
    await first
    // Learn epoch A and a high watermark.
    sock.serverFrame({
      jsonrpc: '2.0',
      method: 'event',
      params: { type: 'gateway.ready', payload: { replay_epoch: 'epoch-A' } }
    })
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 97 } })
    expect(client.getSeqWatermarks()).toEqual({ s1: 97 })

    // Backend restarts: reconnect, replay under a NEW epoch returns nothing
    // (fresh process, empty ring) — pre-fix the client kept watermark 97 and
    // silently believed it missed nothing, forever.
    client.invalidate('drop')
    const second = client.connect('ws://x')
    sock = sockets[sockets.length - 1]
    sock.open()
    await second

    await vi.waitFor(() => {
      expect(sock.lastRequest().method).toBe('session.events.since')
    })
    const req = sock.lastRequest()
    sock.serverFrame({
      jsonrpc: '2.0',
      id: req.id,
      result: { events: [], latest_seq: 0, truncated: false, count: 0, epoch: 'epoch-B' }
    })

    await vi.waitFor(() => {
      expect(client.getSeqWatermarks()).toEqual({})
    })

    // New-epoch events build fresh watermarks from scratch.
    sock.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 3 } })
    expect(client.getSeqWatermarks()).toEqual({ s1: 3 })
    client.close()
  })
})
