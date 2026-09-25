// @vitest-environment node
import { expect, test } from 'vitest'

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
