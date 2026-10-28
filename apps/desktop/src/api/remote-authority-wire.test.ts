// @vitest-environment node
import { expect, test, vi } from 'vitest'

import { HermesGateway } from './client'

test('URL remote authority negotiates before create, submit and model mutation', async () => {
  const packageName = 'ws'
  const { WebSocketServer } = await import(packageName)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const sent: Array<{ method: string; params: Record<string, any> }> = []
  server.on('connection', (socket: any) => {
    const ready = setTimeout(() => socket.send(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: {
      type: 'gateway.ready', payload: { session_authority: true }
    } })), 30)

    socket.on('close', () => clearTimeout(ready))
    socket.on('message', (bytes: Buffer) => {
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
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test('a remote close before protocol readiness rejects the dial instead of retaining a stale gate', async () => {
  const packageName = 'ws'
  const { WebSocketServer } = await import(packageName)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  server.on('connection', (socket: any) => socket.close())
  const client = new HermesGateway()

  try {
    const address = server.address() as { port: number }
    await expect(client.connect(`ws://127.0.0.1:${address.port}/api/ws?ticket=remote`)).rejects.toThrow('not connected')
  } finally {
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test('a failed remote dial cannot cancel a replacement started by its close handler', async () => {
  const packageName = 'ws'
  const { WebSocketServer } = await import(packageName)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  let connections = 0
  server.on('connection', (socket: any) => {
    if (++connections === 1) { socket.close();

 return }

    socket.send(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: {
      type: 'gateway.ready', payload: { session_authority: true }
    } }))
  })
  const client = new HermesGateway()
  const address = server.address() as { port: number }
  const url = `ws://127.0.0.1:${address.port}/api/ws?ticket=remote`
  let redial: Promise<void> | undefined

  const off = client.onState(state => {
    if (state === 'closed' && !redial) { redial = client.connect(url) }
  })

  try {
    await expect(client.connect(url)).rejects.toThrow('not connected')
    await redial
    expect(client.connectionState).toBe('open')
  } finally {
    off()
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
