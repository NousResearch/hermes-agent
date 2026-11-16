import { beforeEach, describe, expect, it, vi } from 'vitest'

import { JsonRpcGatewayClient } from './json-rpc-gateway'

/**
 * Minimal EventTarget-based WebSocket stand-in, matching the replay suite's: the folded-handshake
 * tests drive real dispatch semantics and count the frames the client actually sends.
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

  open(): void {
    this.readyState = 1
    this.dispatchEvent(new Event('open'))
  }

  serverFrame(obj: unknown): void {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(obj) }))
  }

  requests(): Array<{ id: string; method: string; params: Record<string, unknown> }> {
    return this.sent.map(line => JSON.parse(line) as { id: string; method: string; params: Record<string, unknown> })
  }

  requestFor(method: string) {
    return this.requests().find(request => request.method === method)
  }

  answer(request: { id: string }, result: unknown): void {
    this.serverFrame({ jsonrpc: '2.0', id: request.id, result })
  }

  fail(request: { id: string }, code: number, message: string): void {
    this.serverFrame({ jsonrpc: '2.0', id: request.id, error: { code, message } })
  }
}

let sockets: FakeWebSocket[]

const makeClient = () =>
  new JsonRpcGatewayClient({
    socketFactory: url => new FakeWebSocket(url) as unknown as WebSocket,
    heartbeatIntervalMs: 0,
    heartbeatDeadlineMs: 0,
    connectTimeoutMs: 1000
  })

/** A `gateway.ready` frame; `resumed` advertises the folded handshake (a session-authority socket). */
const readyFrame = (resumed: boolean, replayEpoch = 'proc-1') => ({
  jsonrpc: '2.0',
  method: 'event',
  params: {
    type: 'gateway.ready',
    payload: {
      skin: {},
      skin_etag: 'etag-1',
      change_events: true,
      heartbeat: false,
      replay_epoch: replayEpoch,
      resumed_handshake: resumed
    }
  }
})

/** Connect, learn the backend's handshake from its ready frame, watermark two sessions, reconnect. */
const reconnectWithWatermarks = async (client: JsonRpcGatewayClient) => {
  const first = client.connect('ws://x')
  const firstSocket = sockets[sockets.length - 1]
  firstSocket.open()
  await first
  firstSocket.serverFrame(readyFrame(true))
  firstSocket.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 4 } })
  firstSocket.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's2', seq: 7 } })
  client.invalidate('drop')

  const second = client.connect('ws://x')
  const socket = sockets[sockets.length - 1]
  socket.open()
  await second

  return socket
}

describe('JsonRpcGatewayClient folded post-upgrade handshake', () => {
  beforeEach(() => {
    FakeWebSocket.instances = []
    sockets = FakeWebSocket.instances as unknown as FakeWebSocket[]
  })

  it('folds capabilities + every session replay into ONE client.hello round trip', async () => {
    const client = makeClient()
    const socket = await reconnectWithWatermarks(client)

    // Nothing goes out before the ready frame (the fold's advertisement decides the first frame).
    expect(socket.requests()).toHaveLength(0)
    socket.serverFrame(readyFrame(true))

    await vi.waitFor(() => expect(socket.requestFor('client.hello')).toBeTruthy())

    const requests = socket.requests()
    expect(requests).toHaveLength(1)
    expect(requests[0].method).toBe('client.hello')
    expect(requests[0].params).toMatchObject({
      capabilities: { server_requests: true },
      replay_epoch: 'proc-1',
      watermarks: { s1: 4, s2: 7 }
    })
    // The legacy sequence's two other round trips are gone.
    expect(socket.requestFor('client.capabilities')).toBeUndefined()
    expect(socket.requestFor('session.events.since')).toBeUndefined()

    const barrier = client.sessionReplayBarrier('s1')
    const hello = socket.requestFor('client.hello')!
    socket.answer(hello, {
      capabilities_result: { server_requests: ['clarify'], declines_not_shown: true, canonical_clarify_batch: true },
      skin_etag: 'etag-1',
      open_requests: [],
      per_session_replay: [
        { session_id: 's1', events: [{ type: 'message.complete', session_id: 's1', seq: 6, payload: {} }], latest_seq: 6, last_sequence: 6, epoch: 'proc-1', replay_epoch: 'proc-1', truncated: false, snapshot_required: false, count: 1 },
        { session_id: 's2', events: [], latest_seq: 7, last_sequence: 7, epoch: 'proc-1', replay_epoch: 'proc-1', truncated: false, snapshot_required: false, count: 0 }
      ]
    })

    await expect(barrier).resolves.toBe(true)
    expect(client.getSeqWatermarks()).toEqual({ s1: 6, s2: 7 })
    // The one answer is the first response: the socket is application-usable on it.
    await expect(client.whenApplicationUsable()).resolves.toBeUndefined()
    client.close()
  })

  it('falls back to the legacy sequence on the SAME socket when the backend answers -32601', async () => {
    const client = makeClient()
    const socket = await reconnectWithWatermarks(client)
    const before = sockets.length

    socket.serverFrame(readyFrame(true))
    await vi.waitFor(() => expect(socket.requestFor('client.hello')).toBeTruthy())

    // A backend that advertised the fold but does not serve it.
    socket.fail(socket.requestFor('client.hello')!, -32601, 'unknown method: client.hello')

    await vi.waitFor(() => expect(socket.requests().filter(r => r.method === 'session.events.since')).toHaveLength(2))
    expect(sockets.length).toBe(before) // never a redial
    expect(socket.requestFor('client.capabilities')).toBeTruthy()

    const since = socket.requests().filter(r => r.method === 'session.events.since')
    expect(since.map(r => r.params)).toEqual([
      { session_id: 's1', last_seen: 4 },
      { session_id: 's2', last_seen: 7 }
    ])

    for (const request of since) {
      socket.answer(request, { events: [], latest_seq: 0, truncated: false, count: 0, epoch: 'proc-1' })
    }

    await expect(client.sessionReplayBarrier('s1')).resolves.toBe(true)
    client.close()
  })

  it('keeps the legacy open-time fan-out for a backend that does not advertise the fold', async () => {
    const client = makeClient()
    const first = client.connect('ws://x')
    let socket = sockets[sockets.length - 1]
    socket.open()
    await first
    socket.serverFrame(readyFrame(false))
    socket.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 3 } })
    client.invalidate('drop')

    const second = client.connect('ws://x')
    socket = sockets[sockets.length - 1]
    socket.open()
    await second

    // Requested at open, without waiting for a ready frame — unchanged from before the fold.
    await vi.waitFor(() => expect(socket.requestFor('session.events.since')).toBeTruthy())
    socket.serverFrame(readyFrame(false))
    expect(socket.requestFor('client.hello')).toBeUndefined()
    expect(socket.requestFor('client.capabilities')).toBeTruthy()
    client.close()
  })

  it('holds live frames racing the fold and dispatches them after the replayed window', async () => {
    const client = makeClient()
    const seen: number[] = []
    client.on('message.delta', event => seen.push(Number(event.seq)))

    const socket = await reconnectWithWatermarks(client)
    // The first generation's own live frames (s1:4, s2:7) dispatched as they arrived.
    expect(seen).toEqual([4, 7])
    socket.serverFrame(readyFrame(true))
    await vi.waitFor(() => expect(socket.requestFor('client.hello')).toBeTruthy())

    // Races the answer: parked behind the replay barrier, never dispatched ahead of the gap.
    socket.serverFrame({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', session_id: 's1', seq: 9 } })
    expect(seen).toEqual([4, 7])

    const hello = socket.requestFor('client.hello')!
    socket.answer(hello, {
      capabilities_result: { server_requests: [] },
      skin_etag: '',
      open_requests: [],
      per_session_replay: [
        { session_id: 's1', events: [{ type: 'message.delta', session_id: 's1', seq: 5, payload: {} }], latest_seq: 5, last_sequence: 5, epoch: 'proc-1', replay_epoch: 'proc-1', truncated: false, snapshot_required: false, count: 1 },
        { session_id: 's2', events: [], latest_seq: 7, last_sequence: 7, epoch: 'proc-1', replay_epoch: 'proc-1', truncated: false, snapshot_required: false, count: 0 }
      ]
    })

    await vi.waitFor(() => expect(seen).toEqual([4, 7, 5, 9]))
    client.close()
  })

  it('skips a truncated window batched by the fold and advances past its head', async () => {
    const client = makeClient()
    const seen: number[] = []
    client.on('message.delta', event => seen.push(Number(event.seq)))

    const socket = await reconnectWithWatermarks(client)
    expect(seen).toEqual([4, 7])
    socket.serverFrame(readyFrame(true))
    await vi.waitFor(() => expect(socket.requestFor('client.hello')).toBeTruthy())

    const hello = socket.requestFor('client.hello')!
    socket.answer(hello, {
      capabilities_result: { server_requests: [] },
      skin_etag: '',
      open_requests: [],
      per_session_replay: [
        { session_id: 's1', events: [{ type: 'message.delta', session_id: 's1', seq: 40, payload: {} }], latest_seq: 40, last_sequence: 40, epoch: 'proc-1', replay_epoch: 'proc-1', truncated: true, snapshot_required: false, count: 1 },
        { session_id: 's2', events: [], latest_seq: 7, last_sequence: 7, epoch: 'proc-1', replay_epoch: 'proc-1', truncated: false, snapshot_required: false, count: 0 }
      ]
    })

    await expect(client.sessionReplayBarrier('s1')).resolves.toBe(true)
    await vi.waitFor(() => expect(client.getSeqWatermarks()).toEqual({ s1: 40, s2: 7 }))
    // The truncated window is skipped whole — the hole is never painted — but the cursor moved past it.
    expect(seen).toEqual([4, 7])
    client.close()
  })

  it('drops the cursor for a session the owner refused, and keeps it for one it did not', async () => {
    const client = makeClient()
    const socket = await reconnectWithWatermarks(client)
    socket.serverFrame(readyFrame(true))
    await vi.waitFor(() => expect(socket.requestFor('client.hello')).toBeTruthy())

    const hello = socket.requestFor('client.hello')!
    socket.answer(hello, {
      capabilities_result: { server_requests: [] },
      skin_etag: '',
      open_requests: [],
      per_session_replay: [
        { session_id: 's1', events: [], latest_seq: 0, last_sequence: 0, epoch: '', replay_epoch: '', truncated: false, snapshot_required: false, count: 0, error: 'not_found' },
        { session_id: 's2', events: [], latest_seq: 0, last_sequence: 0, epoch: '', replay_epoch: '', truncated: false, snapshot_required: false, count: 0, error: 'permission_denied' }
      ]
    })

    await expect(client.sessionReplayBarrier('s1')).resolves.toBe(true)
    expect(client.getSeqWatermarks()).toEqual({ s2: 7 })
    client.close()
  })

  it('re-delivers the folded open_requests to the request handlers', async () => {
    const client = makeClient()
    const seen: string[] = []
    client.onRequest(request => {
      seen.push(request.method)

      return true
    })

    const socket = await reconnectWithWatermarks(client)
    socket.serverFrame(readyFrame(true))
    await vi.waitFor(() => expect(socket.requestFor('client.hello')).toBeTruthy())

    const hello = socket.requestFor('client.hello')!
    socket.answer(hello, {
      capabilities_result: { server_requests: ['approval'] },
      skin_etag: '',
      open_requests: [{ id: 'srq-1', method: 'approval', params: { session_id: 's1' } }],
      per_session_replay: []
    })

    await vi.waitFor(() => expect(seen).toEqual(['approval']))
    client.close()
  })
})
