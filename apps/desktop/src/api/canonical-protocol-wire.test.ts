// @vitest-environment node
import { expect, test, vi } from 'vitest'

import { HermesGateway } from './client'

test('named canonical prompt listeners receive replay-safe IDs before answering on the wire', async () => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const sent: any[] = []
  server.on('connection', (socket: any) => {
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())
      sent.push(frame)
      socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result: { status: 'resolved' } }))
    })
    socket.send(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'approval.request', session_id: 's', payload: { prompt_id: 'p', execution_generation: 4, choices: ['once', 'deny'] } } }))
  })
  const client = new HermesGateway()
  let projected: any
  let received!: () => void
  const ready = new Promise<void>(resolve => { received = resolve })
  client.on('approval.request', event => { projected = { ...(event.payload as object) }; received() })

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?native_dial=fixture&ticket=one-use`)
    await ready
    expect(projected.request_id).toBe('p')
    await client.request('approval.respond', { session_id: 's', request_id: projected.request_id, choice: 'once' })
    expect(sent.at(-1).params).toEqual({ session_id: 's', prompt_id: 'p', execution_generation: 4, choice: 'once' })
  } finally {
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test('a legacy (non-canonical) dial strips canonical-only identity keys the serve contract refuses', async () => {
  // `hermes serve` answers 4000 "out of sync" for an unknown key, which the remote-topology E2E
  // surfaced as a permanent "Session unavailable" on a URL+token connection.
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const sent: any[] = []
  server.on('connection', (socket: any) => {
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())
      sent.push(frame)
      socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result: { session_id: 'abc' } }))
    })
  })
  const client = new HermesGateway()

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?ticket=legacy`)
    await client.request('session.create', { cols: 96, source: 'desktop', cwd: '/x', fast: false, request_id: 'r1' })
    expect(sent.at(-1).params).toEqual({ cols: 96, source: 'desktop', cwd: '/x', fast: false })
    await client.request('session.branch_stored', { cols: 96, source: 'desktop', parent_session_id: 'p', request_id: 'r2' })
    expect(sent.at(-1).method).toBe('session.branch_stored')
    expect(sent.at(-1).params).not.toHaveProperty('request_id')
  } finally {
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test('a legacy dial branches from a message with the serve contract\'s count, never the canonical row boundary', async () => {
  // N20: the legacy `session.branch` contract is closed (extra="forbid"); `through_message_id`
  // answered 4000 "out of sync", which is not a missing method, so "Branch from here" failed.
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const allowed = new Set(['session_id', 'name', 'count', 'idempotency_key'])
  server.on('connection', (socket: any) => {
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())
      const unknown = Object.keys(frame.params).filter(key => !allowed.has(key))

      socket.send(JSON.stringify(unknown.length
        ? { jsonrpc: '2.0', id: frame.id, error: { code: 4000, message: `invalid params: ${unknown}` } }
        : { jsonrpc: '2.0', id: frame.id, result: { session_id: 'child', kept: frame.params.count } }))
    })
  })
  const client = new HermesGateway()

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?ticket=legacy`)
    await expect(client.request('session.branch', { session_id: 'parent', idempotency_key: 'k', count: 3, through_message_id: 41 }))
      .resolves.toMatchObject({ session_id: 'child', kept: 3 })
  } finally {
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test('a compress settle resumes under the caller\'s original timeout and abort signal', async () => {
  // R4: the follow-up `session.resume` must not fall back to the default 120 s deadline
  // and ignore the caller's AbortSignal (a cancelled compress would otherwise hang).
  const { JsonRpcGatewayClient } = await import('@hermes/shared')
  const wire: Array<[string, unknown, AbortSignal | undefined]> = []

  const spy = vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async function (method: string, _params?: unknown, timeoutMs?: number, signal?: AbortSignal) {
    wire.push([method, timeoutMs, signal])

    return (method === 'session.mutate'
      ? { session_id: 's', revision: 5, operation: 'compress', target_session_id: 's', message_count: 2 }
      : { session_id: 's', revision: 4, execution_generation: 1, messages: [] }) as never
  })

  const client = new HermesGateway()
  Object.assign(client, { canonical: true })

  try {
    await client.request('session.resume', { session_id: 's' })
    const controller = new AbortController()
    await client.request('session.compress', { session_id: 's' }, 7_000, controller.signal)
    // Both legs spend one shared caller deadline (canonical-deadline.test pins the exact
    // remaining budget), so each gets a positive share of the 7 s, never the 120 s default.
    expect(wire.slice(1).map(([method]) => method)).toEqual(['session.mutate', 'session.resume'])
    const [mutateMs, resumeMs] = wire.slice(1).map(([, timeoutMs]) => timeoutMs as number)
    expect(mutateMs).toBeGreaterThan(0)
    expect(mutateMs).toBeLessThanOrEqual(7_000)
    expect(resumeMs).toBeGreaterThan(0)
    expect(resumeMs).toBeLessThanOrEqual(mutateMs)
    expect(wire[2][2]).toBe(controller.signal)
  } finally {
    spy.mockRestore()
  }
})

test('shared-socket equal-ID streams use replay owners and unknown epochs reconcile without changing controls', async () => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const calls: any[] = []
  let socket: any

  const snapshots: Record<string, any> = {
    alpha: { session_id: 'same', revision: 4, execution_generation: 2, replay_epoch: 'epoch-alpha', prompts: [] },
    beta: { session_id: 'same', revision: 20, execution_generation: 9, replay_epoch: 'epoch-beta', prompts: [] }
  }

  server.on('connection', (connected: any) => {
    socket = connected
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())
      calls.push(frame)
      const snapshot = snapshots[frame.params.profile]

      const result = frame.method === 'session.resume' ? snapshot : frame.method === 'prompt.submit'
        ? { admission_id: 'accepted-beta', ref: { profile_id: 'beta', session_id: 'same' }, status: 'queued' }
        : { ...snapshot, operation: frame.params.operation }

      socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }))
    })
  })
  const client = new HermesGateway()
  const named: any[] = []
  const wildcard: any[] = []
  client.on('session.info', event => named.push(event))
  client.onAny(event => { if (event.type === 'session.info') {wildcard.push(event)} })

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?native_dial=fixture&ticket=one-use`)
    await client.request('session.resume', { session_id: 'same', profile: 'alpha' })
    await client.request('prompt.submit', { session_id: 'same', profile: 'beta', submission_id: 'beta-input', text: 'beta' })
    expect(calls.filter(frame => frame.method === 'session.resume').map(frame => frame.params.profile)).toEqual(['alpha', 'beta'])

    const send = (epoch: string, generation: number, revision: number) => socket.send(JSON.stringify({
      jsonrpc: '2.0', method: 'event', params: { type: 'session.info', session_id: 'same',
        replay_epoch: epoch, execution_generation: generation, payload: { revision, execution_generation: generation } }
    }))

    snapshots.alpha = { ...snapshots.alpha, revision: 5, execution_generation: 3 }
    snapshots.beta = { ...snapshots.beta, revision: 21, execution_generation: 10 }
    send('epoch-beta', 10, 21)
    send('epoch-alpha', 3, 5)
    await vi.waitFor(() => expect(named).toHaveLength(2))
    expect(named.map(event => event.profile)).toEqual(['beta', 'alpha'])
    expect(wildcard.map(event => event.profile)).toEqual(['beta', 'alpha'])
    await client.request('session.title', { session_id: 'same', profile: 'alpha', title: 'alpha title' })
    expect(calls.find(frame => frame.method === 'session.mutate').params.expected_revision).toBe(5)
    send('unrecognized-epoch', 999, 999)
    await vi.waitFor(() => expect(calls.filter(frame => frame.method === 'session.resume')).toHaveLength(4))
    expect(named).toHaveLength(2)
    await client.request('session.interrupt', { session_id: 'same', profile: 'alpha' })
    expect(calls.find(frame => frame.method === 'session.interrupt').params.execution_generation).toBe(3)
    await expect(client.request('session.interrupt', { session_id: 'same' })).rejects.toThrow('several profiles')
  } finally {
    client.close()

    for (const peer of server.clients) {peer.terminate()}
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
