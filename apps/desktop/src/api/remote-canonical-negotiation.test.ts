// @vitest-environment node
import type { IncomingMessage } from 'node:http'

import { expect, test, vi } from 'vitest'

import { HermesGateway } from './client'

interface FixtureSocket {
  on(event: 'message', handler: (bytes: Buffer) => void): void
  send(data: string): void
  terminate(): void
}

const snapshot = {
  session_id: 'chat', stored_session_id: 'chat', messages: [], prompts: [], pending: [],
  revision: 1, execution_generation: 1, authority_epoch: 1, replay_epoch: 'fixture',
  last_sequence: 0, running: false, info: { profile_name: 'engineer' }
}

test('a public authenticated canonical gateway retains the attached profile and prepares submissions', async () => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const frames: Array<{ method: string; params: Record<string, unknown> }> = []
  let protocol: unknown
  server.on('connection', (socket: FixtureSocket, request: IncomingMessage) => {
    protocol = request.headers['sec-websocket-protocol']
    socket.on('message', bytes => {
      const frame = JSON.parse(bytes.toString())
      frames.push(frame)
      let result: unknown = {}

      if (frame.method === 'session.resume') {result = snapshot}

      if (frame.method === 'prompt.submit') {
        if (frame.params.profile !== 'engineer' || typeof frame.params.submission_id !== 'string') {
          socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, error: { code: 4001, message: 'permission_denied' } }))

          return
        }

        result = { admission_id: 'admission', ref: { session_id: 'chat' }, sequence: 1,
          status: 'queued', outcome: null, authority_epoch: 1, execution_generation: 1 }
      }

      socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }))
    })
    socket.send(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: {
      type: 'gateway.ready', payload: { session_authority: true }
    } }))
  })
  const client = new HermesGateway()
  const ready = new Promise<void>(resolve => client.on('gateway.ready', () => resolve()))

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?ticket=public-fixture`)
    await ready
    await client.request('session.resume', { session_id: 'chat', profile: 'engineer' })
    await expect(client.request('prompt.submit', { session_id: 'chat', text: 'fixture input' })).resolves.toMatchObject({ status: 'queued' })
    expect(frames.find(frame => frame.method === 'prompt.submit')?.params).toMatchObject({
      session_id: 'chat', profile: 'engineer', submission_id: expect.any(String)
    })
    expect(protocol).toBeUndefined()
  } finally {
    client.close()

    for (const socket of server.clients) {socket.terminate()}
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test.each([{}, { session_authority: false }, { session_authority: 'true' }])('a reconnect resets protocol selection for a non-canonical advertisement %j', async advertisement => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const creates: Record<string, unknown>[] = []
  let connection = 0
  server.on('connection', (socket: FixtureSocket) => {
    const canonical = ++connection === 1
    socket.on('message', bytes => {
      const frame = JSON.parse(bytes.toString())

      if (frame.method === 'session.create') {creates.push(frame.params)}
      socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result: {} }))
    })
    socket.send(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: {
      type: 'gateway.ready', payload: canonical ? { session_authority: true } : advertisement
    } }))
  })
  const client = new HermesGateway()
  const address = server.address() as { port: number }
  const url = `ws://127.0.0.1:${address.port}/api/ws?ticket=public-fixture`
  let ready = new Promise<void>(resolve => client.on('gateway.ready', () => resolve()))

  try {
    await client.connect(url)
    await ready
    await client.request('session.create', { source: 'desktop', cols: 80, request_id: 'first' })
    expect(creates[0]).toHaveProperty('request_id', 'first')
    client.close()
    ready = new Promise<void>(resolve => client.on('gateway.ready', () => resolve()))
    await client.connect(url)
    await ready
    await client.request('session.create', { source: 'desktop', cols: 80, request_id: 'second' })
    expect(creates[1]).not.toHaveProperty('request_id')
  } finally {
    client.close()

    for (const socket of server.clients) {socket.terminate()}
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test.each(['viewer', 'peer'] as const)('%s disconnect rejects an unfinished public protocol handshake', async closer => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  let peer: FixtureSocket | undefined
  server.on('connection', (socket: FixtureSocket) => { peer = socket })
  const client = new HermesGateway()
  const opened = new Promise<void>(resolve => client.onState(state => { if (state === 'open') { resolve() } }))
  const address = server.address() as { port: number }
  const connected = client.connect(`ws://127.0.0.1:${address.port}/api/ws?ticket=public-fixture`)
  const rejected = expect(connected).rejects.toThrow('before protocol readiness')

  try {
    await opened

    if (closer === 'viewer') { client.close() } else { peer?.terminate() }
    await rejected
  } finally {
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test('a public readiness timeout closes the failed transport and permits redial', async () => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  let connections = 0
  server.on('connection', (socket: FixtureSocket) => {
    if (++connections > 1) {
      socket.send(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: {
        type: 'gateway.ready', payload: { session_authority: true }
      } }))
    }

    socket.on('message', bytes => {
      const frame = JSON.parse(bytes.toString())
      socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result: {} }))
    })
  })
  const client = new HermesGateway()
  const address = server.address() as { port: number }
  const url = `ws://127.0.0.1:${address.port}/api/ws?ticket=public-fixture`
  const opened = new Promise<void>(resolve => client.onState(state => { if (state === 'open') { resolve() } }))
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })

  try {
    const rejected = expect(client.connect(url)).rejects.toThrow('protocol readiness timed out')
    await opened
    await vi.advanceTimersByTimeAsync(30_001)
    await rejected
    expect(client.connectionState).toBe('closed')
    vi.useRealTimers()
    await client.connect(url)
    await expect(client.request('ping')).resolves.toEqual({})
    expect(connections).toBe(2)
  } finally {
    client.close()
    vi.useRealTimers()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test('requests made at transport open wait for public protocol readiness', async () => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const frames: Array<{ method: string; params: Record<string, unknown> }> = []

  let announceReady = () => {}
  server.on('connection', (socket: FixtureSocket) => {
    announceReady = () => socket.send(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: {
      type: 'gateway.ready', payload: { session_authority: true }
    } }))
    socket.on('message', bytes => {
      const frame = JSON.parse(bytes.toString())
      frames.push(frame)
      const result = frame.method === 'session.resume' ? snapshot : {}
      socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }))
    })
  })
  const client = new HermesGateway()
  const opened = new Promise<void>(resolve => client.onState(state => { if (state === 'open') { resolve() } }))
  const address = server.address() as { port: number }
  const connected = client.connect(`ws://127.0.0.1:${address.port}/api/ws?ticket=public-fixture`)
  void connected.catch(() => {})

  try {
    await opened
    const resumed = client.request('session.resume', { session_id: 'chat', profile: 'engineer' })
    void resumed.catch(() => {})
    await new Promise(resolve => setTimeout(resolve, 25))
    expect(frames.filter(frame => frame.method === 'session.resume')).toHaveLength(0)
    announceReady()
    await connected
    await resumed
    expect(frames.find(frame => frame.method === 'session.resume')?.params).toMatchObject({
      session_id: 'chat', profile: 'engineer'
    })
  } finally {
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
