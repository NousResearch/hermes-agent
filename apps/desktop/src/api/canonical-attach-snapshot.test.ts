// @vitest-environment node
import { expect, test } from 'vitest'

import { HermesGateway } from './client'

// R6: a clarification raised before this viewer attached arrives only in the resume snapshot.
// The attach must deliver it to the question card and mark the session attached, so the
// answer goes straight out without another resume.
test('a cold attach delivers the snapshot prompt and marks the session attached', async () => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const sent: any[] = []
  server.on('connection', (socket: any) => {
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())
      sent.push(frame)

      const result = frame.method === 'session.resume'
        ? { session_id: 's', stored_session_id: 's', revision: 2, execution_generation: 5, running: true, messages: [],
            prompts: [{ kind: 'clarify', prompt_id: 'c1', execution_generation: 5, question: 'Which target?', choices: [] }] }
        : { status: 'resolved' }

      socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }))
    })
  })
  const client = new HermesGateway()
  const delivered: any[] = []
  client.onRequest(request => { delivered.push(request) })

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?native_dial=fixture&ticket=one-use`)
    await client.request('session.resume', { session_id: 's' })
    expect(delivered.map(request => [request.method, request.id, request.replayed])).toEqual([['clarify', 'c1', true]])

    sent.length = 0
    await client.request('clarify.lock', { request_id: 'c1', question_id: 'c1', answer: 'prod' })
    expect(sent.map(frame => frame.method)).toEqual(['clarify.respond'])
  } finally {
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
