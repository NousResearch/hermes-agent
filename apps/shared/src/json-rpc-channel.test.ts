import { describe, expect, it, vi } from 'vitest'

import { JsonRpcGatewayError, JsonRpcRequestChannel, type JsonRpcTransport } from './json-rpc-channel.js'

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

    const ok = channel.request('session.create', { cols: 80 })
    const okId = last().id
    const failing = channel.request('projects.create', { name: 'example' })
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
    channel.handleFrame(
      JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'session.info', payload: {} } })
    )
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

      const slow = expect(channel.requestUntyped('a.slow', {}, 1_000)).rejects.toThrow(
        'request timed out after 1s: a.slow'
      )

      const untilDetach = channel.requestUntyped('b.wait', {})

      await vi.advanceTimersByTimeAsync(1_000)
      await slow

      channel.detach(new Error('gateway exited (1)'))
      await expect(untilDetach).rejects.toThrow('gateway exited (1)')
      await expect(channel.requestUntyped('c.after', {})).rejects.toThrow('gateway not connected')
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
        sent
          .map(f => JSON.parse(f) as { id: string; method: string })
          .filter(f => f.method === 'gateway.ping')
          .map(f => f.id)

      // Pongs arrive: alive well past the deadline.
      for (let i = 0; i < 6; i++) {
        await vi.advanceTimersByTimeAsync(100)
        channel.handleFrame(JSON.stringify({ id: pingIds().at(-1), jsonrpc: '2.0', result: { ok: true } }))
      }

      expect(failures).toEqual([])

      // A response to one of our own requests also counts.
      const call = channel.request('session.list', {})
      const callId = (JSON.parse(sent.at(-1)!) as { id: string }).id

      await vi.advanceTimersByTimeAsync(250)
      channel.handleFrame(JSON.stringify({ id: callId, jsonrpc: '2.0', result: [] }))
      await call
      await vi.advanceTimersByTimeAsync(100)
      expect(failures).toEqual([])

      // Deltas keep streaming but no ping is answered → dead.
      for (let i = 0; i < 4; i++) {
        await vi.advanceTimersByTimeAsync(100)
        channel.handleFrame(
          JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'message.delta', payload: {} } })
        )
      }

      expect(failures).toEqual(['WebSocket heartbeat acknowledgement timed out'])
    } finally {
      vi.useRealTimers()
    }
  })

  // Server→client requests (tui_gateway/server_requests.py): the backend asks,
  // the client answers with a RESPONSE frame carrying the same id.
  it('advertises server-request support once per gateway.ready and shrugs off an older backend', () => {
    // A backend that never hears client.capabilities treats a WebSocket client as a build older than
    // server→client requests and fails every clarify/approval for it at once (tui_gateway/server_requests.py).
    const channel = new JsonRpcRequestChannel({ requestIdPrefix: 'c' })
    const { sent, transport, last } = spyTransport()

    channel.attach(transport)
    channel.handleFrame(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'session.info', session_id: 's' } }))
    expect(sent).toHaveLength(0)

    channel.handleFrame(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'gateway.ready', payload: {} } }))
    expect(sent).toHaveLength(1)
    expect(JSON.parse(sent[0])).toMatchObject({ method: 'client.capabilities', params: { server_requests: true } })

    // An older backend answers -32601: nothing rejects out of the channel.
    channel.handleFrame(JSON.stringify({ error: { code: -32601, message: 'unknown method' }, id: last().id, jsonrpc: '2.0' }))
    expect(sent).toHaveLength(1)
  })

  it('routes a server request by generated method and answers -32601 when the method is unknown', () => {
    const unhandled: string[] = []
    const channel = new JsonRpcRequestChannel({ onUnhandledRequest: req => void unhandled.push(req.method) })
    const { sent, transport } = spyTransport()

    channel.attach(transport)
    channel.onServerRequest('clarify', req => req.respond({ answer: 'yes' }))

    channel.handleFrame(
      JSON.stringify({
        id: 'srq-1',
        jsonrpc: '2.0',
        method: 'clarify',
        params: { choices: [], kind: 'single', multi_select: false, question: 'Continue?', session_id: 's1' }
      })
    )
    channel.handleFrame(
      JSON.stringify({ id: 'srq-2', jsonrpc: '2.0', method: 'not.declared', params: { session_id: 's1' } })
    )

    const frames = sent.map(f => JSON.parse(f) as { id: string; result?: unknown; error?: { code: number } })

    expect(frames[0]).toEqual({ id: 'srq-1', jsonrpc: '2.0', result: { answer: 'yes' } })
    expect(frames[1].id).toBe('srq-2')
    expect(frames[1].error?.code).toBe(-32601)
    expect(unhandled).toEqual(['not.declared'])
  })

  it('stamps the kind on a contract-7 clarify (no discriminator) so a current-main backend is still answered', () => {
    const channel = new JsonRpcRequestChannel()
    const { sent, transport } = spyTransport()
    const seen: unknown[] = []

    channel.attach(transport)
    channel.onServerRequest('clarify', req => {
      seen.push(req.params)
      req.respond(req.params.kind === 'batch' ? { answers: { q1: 'a' }, timed_out: false } : { answer: 'yes' })
    })

    channel.handleFrame(
      JSON.stringify({
        id: 'srq-old-batch',
        jsonrpc: '2.0',
        method: 'clarify',
        params: { questions: [{ qid: 'q1', question: 'Which?', choices: [], multi_select: false }], session_id: 's1' }
      })
    )
    channel.handleFrame(
      JSON.stringify({ id: 'srq-old-single', jsonrpc: '2.0', method: 'clarify', params: { question: 'Go?', session_id: 's1' } })
    )
    channel.handleFrame(JSON.stringify({ id: 'srq-bad', jsonrpc: '2.0', method: 'clarify', params: 'nope' }))

    const frames = sent.map(f => JSON.parse(f) as { id: string; result?: unknown; error?: { code: number } })

    expect(seen).toEqual([
      { answers: null, kind: 'batch', questions: [{ choices: [], multi_select: false, qid: 'q1', question: 'Which?' }], session_id: 's1' },
      { choices: null, kind: 'single', multi_select: false, question: 'Go?', session_id: 's1' }
    ])
    expect(frames[0]).toEqual({ id: 'srq-old-batch', jsonrpc: '2.0', result: { answers: { q1: 'a' }, timed_out: false } })
    expect(frames[1]).toEqual({ id: 'srq-old-single', jsonrpc: '2.0', result: { answer: 'yes' } })
    expect(frames[2].error?.code).toBe(-32602)
  })

  it('re-delivers open_requests through their generated method, tagged replayed', async () => {
    const delivered: Array<{ id: string; replayed?: boolean; sessionId: string | null }> = []
    const channel = new JsonRpcRequestChannel()
    const { sent, transport } = spyTransport()

    channel.attach(transport)
    channel.onServerRequest(
      'sudo',
      req => void delivered.push({ id: req.id, replayed: req.replayed, sessionId: req.sessionId })
    )

    const resume = channel.request('session.resume', { session_id: 's1' })
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
    expect(delivered).toEqual([{ id: 'srq-9', replayed: true, sessionId: 's1' }])
  })

  // Regression (2026-09-15 clarify spinner): a throwing handler used to escape
  // deliverRequest inside the socket listener — no response frame at all, so
  // the backend (clarify_tool: 3600s deadline) waited out the whole block.
  it('answers -32603 when a handler throws, keeps later frames working, and still answers -32601 otherwise', () => {
    const crashed: Array<{ id: string; method: string; message: string }> = []
    const unhandled: string[] = []

    const channel = new JsonRpcRequestChannel({
      onRequestHandlerError: (error, req) => void crashed.push({ id: req.id, method: req.method, message: error.message }),
      onUnhandledRequest: req => void unhandled.push(req.method)
    })

    const { sent, transport } = spyTransport()

    channel.attach(transport)
    // Declared methods only: an undeclared name is answered -32601 before any handler runs.
    channel.onServerRequest('sudo', () => {
      throw new Error('handler exploded')
    })
    channel.onServerRequest('clarify', req => req.respond({ answer: 'yes' }))

    channel.handleFrame(JSON.stringify({ id: 'srq-1', jsonrpc: '2.0', method: 'sudo', params: { session_id: 's1' } }))
    channel.handleFrame(JSON.stringify({ id: 'srq-2', jsonrpc: '2.0', method: 'nobody', params: { session_id: 's1' } }))

    const frames = sent.map(f => JSON.parse(f) as { id: string; error?: { code: number; message?: string } })

    expect(frames[0].id).toBe('srq-1')
    expect(frames[0].error?.code).toBe(-32603)
    expect(frames[0].error?.message).toContain('sudo')
    expect(frames[1].id).toBe('srq-2')
    expect(frames[1].error?.code).toBe(-32601)
    // The crash reports through its own hook; it is not an "unhandled" request.
    expect(crashed).toEqual([{ id: 'srq-1', method: 'sudo', message: 'handler exploded' }])
    expect(unhandled).toEqual(['nobody'])

    // The channel survives: a normal request after the crash still routes.
    channel.handleFrame(JSON.stringify({ id: 'srq-3', jsonrpc: '2.0', method: 'clarify', params: { session_id: 's1' } }))
    const third = sent.at(-1)!
    expect((JSON.parse(third) as { id: string }).id).toBe('srq-3')
    expect((JSON.parse(third) as { result?: { answer?: string } }).result?.answer).toBe('yes')
  })

  // Desktop (store/gateway.ts) and the TUI (gatewayClient.ts) dispatch through onAnyServerRequest,
  // so a renderer exception there is the production crash path, not the per-method one.
  it('answers -32603 when a catch-all handler throws and keeps later requests working', () => {
    const crashed: string[] = []
    const channel = new JsonRpcRequestChannel({ onRequestHandlerError: (_error, req) => void crashed.push(req.id) })
    const { sent, transport } = spyTransport()

    channel.attach(transport)

    let calls = 0

    channel.onAnyServerRequest(req => {
      calls += 1

      if (calls === 1) {
        throw new Error('dispatcher exploded')
      }

      if (req.method === 'clarify') {
        req.respond({ answer: 'yes' })
      }
    })

    expect(() =>
      channel.handleFrame(JSON.stringify({ id: 'srq-1', jsonrpc: '2.0', method: 'clarify', params: { session_id: 's1' } }))
    ).not.toThrow()
    channel.handleFrame(JSON.stringify({ id: 'srq-2', jsonrpc: '2.0', method: 'clarify', params: { session_id: 's1' } }))

    const frames = sent.map(f => JSON.parse(f) as { id: string; error?: { code: number }; result?: { answer?: string } })

    expect(frames.map(f => f.id)).toEqual(['srq-1', 'srq-2'])
    expect(frames[0].error?.code).toBe(-32603)
    expect(frames[1].result?.answer).toBe('yes')
    expect(crashed).toEqual(['srq-1'])
  })
})
