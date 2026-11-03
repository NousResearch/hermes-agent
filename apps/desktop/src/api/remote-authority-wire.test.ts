// @vitest-environment node
import { expect, test, vi } from 'vitest'

import { HermesGateway } from './client'

interface FixtureSocket {
  on(event: 'message', handler: (bytes: Buffer) => void): void
  on(event: 'close', handler: () => void): void
  send(data: string): void
  terminate(): void
}

const readyFrame = JSON.stringify({ jsonrpc: '2.0', method: 'event', params: {
  type: 'gateway.ready', payload: { session_authority: true }
} })

async function listen() {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))

  return server
}

async function shut(client: HermesGateway, server: Awaited<ReturnType<typeof listen>>) {
  client.close()

  for (const socket of server.clients) { socket.terminate() }
  await new Promise<void>(resolve => server.close(() => resolve()))
}

test('a public authority dial negotiates before create, submit and model mutation', async () => {
  const server = await listen()
  const sent: Array<{ method: string; params: Record<string, any> }> = []
  server.on('connection', (socket: FixtureSocket) => {
    const ready = setTimeout(() => socket.send(readyFrame), 30)
    socket.on('close', () => clearTimeout(ready))
    socket.on('message', bytes => {
      const frame = JSON.parse(bytes.toString())
      sent.push(frame)
      const params = frame.params ?? {}
      let result: Record<string, unknown> = {}
      let error: { code: number; message: string } | undefined

      if (frame.method === 'session.create') {
        if (params.source !== 'gui' || 'cols' in params || !params.request_id) {
          error = { code: 4001, message: 'invalid_params' }
        } else {
          result = { session_id: 's', stored_session_id: 's', revision: 1, execution_generation: 2,
            info: { desktop_protocol: 'hermes-gateway-v1', profile_name: 'worker' } }
        }
      } else if (frame.method === 'prompt.submit') {
        result = { admission_id: 'admitted', status: 'queued', ref: { session_id: 's', profile_id: '/profiles/worker' } }
      } else if (frame.method === 'config.set') {
        error = { code: 4001, message: 'use_session_mutation_model' }
      } else if (frame.method === 'session.mutate') {
        result = { session_id: 's', revision: 2, operation: 'model', model: 'picked' }
      }

      socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, ...(error ? { error } : { result }) }))

      if (frame.method === 'prompt.submit') {
        socket.send(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: {
          type: 'message.start', session_id: 's', payload: {}
        } }))
      }
    })
  })
  const client = new HermesGateway()
  let incomingProfile: string | undefined
  client.on('message.start', event => { incomingProfile = event.profile })

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?ticket=remote-session-ticket`)
    await expect(client.request('session.create', { source: 'desktop', profile: 'worker', cols: 96, fast: false }))
      .resolves.toMatchObject({ session_id: 's' })
    await expect(client.request('prompt.submit', { session_id: 's', profile: 'worker', text: 'hello', submission_id: 'one' }))
      .resolves.toMatchObject({ session_id: 's', submission_id: 'one', admission_id: 'admitted' })
    await vi.waitFor(() => expect(incomingProfile).toBe('worker'))
    await expect(client.request('config.set', { session_id: 's', profile: 'worker', key: 'model', value: 'picked --provider custom' }))
      .resolves.toMatchObject({ ok: true })
    expect(sent.find(frame => frame.method === 'session.mutate')?.params).toMatchObject({ operation: 'model', expected_revision: 1 })
    expect(sent.some(frame => frame.method === 'config.set')).toBe(false)
    // Remote auth stays ticket based: negotiating the wire grants no private native-dial credential.
    expect(sent.some(frame => frame.method === 'client.capabilities' && frame.params.desktop_local)).toBe(false)
  } finally {
    await shut(client, server)
  }
})

test.each(['abort', 'deadline'] as const)('a request waiting for public readiness honours its %s and sends nothing', async exit => {
  const server = await listen()
  const sent: string[] = []

  let announceReady = () => {}
  server.on('connection', (socket: FixtureSocket) => {
    announceReady = () => socket.send(readyFrame)
    socket.on('message', bytes => {
      const frame = JSON.parse(bytes.toString())
      sent.push(frame.method)
      socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result: {} }))
    })
  })
  const client = new HermesGateway()
  const opened = new Promise<void>(resolve => client.onState(state => { if (state === 'open') { resolve() } }))
  const address = server.address() as { port: number }
  const connected = client.connect(`ws://127.0.0.1:${address.port}/api/ws?ticket=public-fixture`)
  void connected.catch(() => {})

  try {
    await opened
    const controller = new AbortController()

    const waiting = exit === 'abort'
      ? client.request('session.create', { source: 'desktop' }, undefined, controller.signal)
      : client.request('session.create', { source: 'desktop' }, 40)

    const settled = expect(waiting).rejects.toThrow(exit === 'abort' ? /abort/i : 'timed out')

    if (exit === 'abort') { setTimeout(() => controller.abort(), 20) }
    await settled
    announceReady()
    await connected
    await expect(client.request('ping')).resolves.toEqual({})
    expect(sent).toContain('ping')
    expect(sent).not.toContain('session.create')
  } finally {
    await shut(client, server)
  }
})
