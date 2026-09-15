import { describe, expect, it, vi } from 'vitest'

import { JsonRpcGatewayError, JsonRpcRequestChannel, type JsonRpcTransport, type ServerRequest } from './json-rpc-channel.js'

const spyTransport = () => {
  const sent: string[] = []
  const transport: JsonRpcTransport = { send: text => void sent.push(text) }

  return { sent, transport, last: () => JSON.parse(sent.at(-1) ?? '{}') as { id: string; method: string } }
}

describe('JsonRpcRequestChannel', () => {
  it('routes responses to the pending call and keeps JSON-RPC code/data on errors', async () => {
    const events: string[] = []
    const channel = new JsonRpcRequestChannel({ onEvent: ev => void events.push(ev.type), requestIdPrefix: 'x' })
    const { transport, last } = spyTransport()

    channel.attach(transport)

    const ok = channel.request<{ ok: boolean }>('session.create', { cols: 80 })
    const okId = last().id
    const failing = channel.request('projects.create')
    const failId = last().id

    expect(okId).toBe('x1')
    channel.handleFrame(JSON.stringify({ id: okId, jsonrpc: '2.0', result: { ok: true } }))
    channel.handleFrame(
      JSON.stringify({
        error: { code: -32601, data: { method: 'projects.create' }, message: 'unknown method: projects.create' },
        id: failId,
        jsonrpc: '2.0'
      })
    )
    channel.handleFrame(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'session.info', payload: {} } }))
    // Non-JSON and unknown ids are ignored, never thrown.
    expect(channel.handleFrame('not json')).toBeNull()
    channel.handleFrame(JSON.stringify({ id: 'never-sent', jsonrpc: '2.0', result: 1 }))

    await expect(ok).resolves.toEqual({ ok: true })
    const error = (await failing.catch((e: unknown) => e)) as JsonRpcGatewayError

    expect(error).toBeInstanceOf(JsonRpcGatewayError)
    expect(error.code).toBe(-32601)
    expect(error.data).toEqual({ method: 'projects.create' })
    expect(events).toEqual(['session.info'])
  })

  // A stdout line that parses to a non-object used to reach `frame.id` and
  // throw out of the readline handler (uncaughtException in the Ink process).
  it.each(['null', '42', '"str"', 'true'])('ignores the non-object JSON line %s without throwing', text => {
    const events: string[] = []
    const channel = new JsonRpcRequestChannel({ onEvent: ev => void events.push(ev.type) })

    channel.attach(spyTransport().transport)

    expect(() => channel.handleFrame(text)).not.toThrow()
    expect(channel.handleFrame(text)).toBeNull()
    expect(events).toEqual([])
  })

  it('detach fails every in-flight call and a per-call timeout names the method', async () => {
    vi.useFakeTimers()

    try {
      const channel = new JsonRpcRequestChannel({ requestTimeoutMs: 60_000 })
      const { transport } = spyTransport()

      channel.attach(transport)

      const slow = expect(channel.request('a.slow', {}, 1_000)).rejects.toThrow('request timed out after 1s: a.slow')
      const untilDetach = channel.request('b.wait')

      await vi.advanceTimersByTimeAsync(1_000)
      await slow

      channel.detach(new Error('gateway exited (1)'))
      await expect(untilDetach).rejects.toThrow('gateway exited (1)')
      await expect(channel.request('c.after')).rejects.toThrow('gateway not connected')
    } finally {
      vi.useRealTimers()
    }
  })

  it("'any-inbound' liveness (desktop/web): pings while frames keep arriving and reports a silent transport", async () => {
    vi.useFakeTimers()

    try {
      const failures: string[] = []

      const channel = new JsonRpcRequestChannel({
        heartbeatDeadlineMs: 300,
        heartbeatIntervalMs: 100,
        heartbeatLiveness: 'any-inbound',
        onHeartbeatFailure: e => void failures.push(e.message)
      })

      const { sent, transport } = spyTransport()

      channel.attach(transport)
      channel.startHeartbeat()

      for (let i = 0; i < 6; i++) {
        await vi.advanceTimersByTimeAsync(100)
        channel.handleFrame(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'status.update' } }))
      }

      expect(sent.filter(f => f.includes('gateway.ping')).length).toBeGreaterThanOrEqual(5)
      expect(failures).toEqual([])

      await vi.advanceTimersByTimeAsync(400)
      expect(failures).toEqual(['WebSocket heartbeat acknowledgement timed out'])
      // Failure stops the timer: no further pings after the report.
      const pings = sent.length
      await vi.advanceTimersByTimeAsync(500)
      expect(sent.length).toBe(pings)
    } finally {
      vi.useRealTimers()
    }
  })

  // TUI contract: a backend whose request loop is wedged may still stream
  // deltas; only a pong (or a response to our own request) proves it can
  // answer, so notifications alone must NOT keep the transport alive.
  it("'response' liveness (default, TUI): unanswered pings fail the heartbeat even while deltas stream", async () => {
    vi.useFakeTimers()

    try {
      const failures: string[] = []

      const channel = new JsonRpcRequestChannel({
        heartbeatDeadlineMs: 300,
        heartbeatIntervalMs: 100,
        onHeartbeatFailure: e => void failures.push(e.message)
      })

      const { sent, transport } = spyTransport()

      channel.attach(transport)
      channel.startHeartbeat()

      const pingIds = () =>
        sent.map(f => JSON.parse(f) as { id: string; method: string }).filter(f => f.method === 'gateway.ping').map(f => f.id)

      // Pongs arrive: alive well past the deadline.
      for (let i = 0; i < 6; i++) {
        await vi.advanceTimersByTimeAsync(100)
        channel.handleFrame(JSON.stringify({ id: pingIds().at(-1), jsonrpc: '2.0', result: { ok: true } }))
      }

      expect(failures).toEqual([])

      // A response to one of our own requests also counts.
      const call = channel.request('session.list')
      const callId = (JSON.parse(sent.at(-1)!) as { id: string }).id

      await vi.advanceTimersByTimeAsync(250)
      channel.handleFrame(JSON.stringify({ id: callId, jsonrpc: '2.0', result: [] }))
      await call
      await vi.advanceTimersByTimeAsync(100)
      expect(failures).toEqual([])

      // Deltas keep streaming but no ping is answered → dead.
      for (let i = 0; i < 4; i++) {
        await vi.advanceTimersByTimeAsync(100)
        channel.handleFrame(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', payload: {} } }))
      }

      expect(failures).toEqual(['WebSocket heartbeat acknowledgement timed out'])
    } finally {
      vi.useRealTimers()
    }
  })

  // Server→client requests (tui_gateway/server_requests.py): the backend asks,
  // the client answers with a RESPONSE frame carrying the same id.
  it('routes a server request to the first accepting handler and answers -32601 when nobody accepts', () => {
    const unhandled: string[] = []
    const channel = new JsonRpcRequestChannel({ onUnhandledRequest: req => void unhandled.push(req.method) })
    const { sent, transport } = spyTransport()

    channel.attach(transport)
    channel.onRequest(req => (req.method === 'clarify' ? void req.respond({ answer: 'yes' }) : false))

    channel.handleFrame(JSON.stringify({ id: 'srq-1', jsonrpc: '2.0', method: 'clarify', params: { session_id: 's1' } }))
    channel.handleFrame(JSON.stringify({ id: 'srq-2', jsonrpc: '2.0', method: 'tour', params: { session_id: 's1' } }))

    const frames = sent.map(f => JSON.parse(f) as { id: string; result?: unknown; error?: { code: number } })

    expect(frames[0]).toEqual({ id: 'srq-1', jsonrpc: '2.0', result: { answer: 'yes' } })
    expect(frames[1].id).toBe('srq-2')
    expect(frames[1].error?.code).toBe(-32601)
    expect(unhandled).toEqual(['tour'])
  })

  it.each(['respond', 'fail'] as const)('ignores a stale %s after reconnect while the replay can still answer', async reply => {
    const delivered: ServerRequest[] = []
    const channel = new JsonRpcRequestChannel()
    const oldConnection = spyTransport()
    const newConnection = spyTransport()
    const request = { id: 'srq-1', method: 'clarify', params: { session_id: 's1' } }

    channel.attach(oldConnection.transport)
    channel.onRequest(req => void delivered.push(req))
    channel.handleFrame(JSON.stringify({ jsonrpc: '2.0', ...request }))
    channel.detach(new Error('connection lost'))
    channel.attach(newConnection.transport)

    const resume = channel.request('session.resume', { session_id: 's1' })

    channel.handleFrame(
      JSON.stringify({
        id: newConnection.last().id,
        jsonrpc: '2.0',
        result: { open_requests: [request], session_id: 's1' }
      })
    )
    await resume
    expect(delivered).toHaveLength(2)
    expect(delivered[1].replayed).toBe(true)

    if (reply === 'respond') {
      delivered[0].respond({ answer: 'stale' })
    } else {
      delivered[0].fail(-32603, 'stale failure')
    }

    expect(oldConnection.sent).toEqual([])
    expect(newConnection.sent).toHaveLength(1) // Only session.resume, no stale reply.

    delivered[1].respond({ answer: 'current' })
    delivered[1].respond({ answer: 'duplicate' })
    delivered[1].fail(-32603, 'duplicate failure')
    expect(newConnection.sent.slice(1).map(text => JSON.parse(text))).toEqual([
      { id: request.id, jsonrpc: '2.0', result: { answer: 'current' } }
    ])
  })

  it('re-delivers open_requests from a response before the caller sees the result, tagged replayed', async () => {
    const delivered: Array<{ id: string; replayed?: boolean }> = []
    const channel = new JsonRpcRequestChannel()
    const { sent, transport } = spyTransport()

    channel.attach(transport)
    channel.onRequest(req => void delivered.push({ id: req.id, replayed: req.replayed }))

    const resume = channel.request<{ session_id: string }>('session.resume', { session_id: 's1' })
    const rid = (JSON.parse(sent.at(-1)!) as { id: string }).id

    channel.handleFrame(
      JSON.stringify({
        id: rid,
        jsonrpc: '2.0',
        result: {
          open_requests: [{ id: 'srq-9', method: 'sudo', params: { session_id: 's1' } }],
          session_id: 's1'
        }
      })
    )
    await expect(resume).resolves.toMatchObject({ session_id: 's1' })
    expect(delivered).toEqual([{ id: 'srq-9', replayed: true }])
  })
})
