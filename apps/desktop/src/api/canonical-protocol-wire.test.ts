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
