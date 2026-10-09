// @vitest-environment node
import { expect, test } from 'vitest'

import { HermesGateway } from './client'

// A canonical shared clarify prompt reaches the question card in its one `questions[]` shape,
// and the card's `clarify.lock` lands as the generation-fenced `clarify.respond`.
test('a canonical clarify prompt drives the question card and its lock answers on the fenced wire', async () => {
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
    socket.send(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'clarify.request', session_id: 's', payload: {
      kind: 'clarify', prompt_id: 'c1', execution_generation: 7, question: 'Which target?', choices: ['staging', 'prod'], multi_select: false
    } } }))
  })
  const client = new HermesGateway()
  let delivered: any
  let received!: () => void
  const ready = new Promise<void>(resolve => { received = resolve })
  client.onRequest(request => { delivered = request; received() })

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?native_dial=fixture&ticket=one-use`)
    await ready
    expect(delivered.method).toBe('clarify')
    expect(delivered.params.questions).toEqual([{ qid: 'c1', question: 'Which target?', choices: ['staging', 'prod'], multi_select: false }])

    // Skips arrive as a null lock: the fenced respond records them as the empty answer.
    sent.length = 0
    await client.request('clarify.lock', { request_id: 'c1', question_id: 'c1', answer: null })
    expect(sent.map(frame => [frame.method, frame.params])).toEqual([
      ['session.resume', { session_id: 's', defer_history: true, omit_messages: true }],
      ['clarify.respond', { session_id: 's', execution_generation: 7, prompt_id: 'c1', answer: '' }]
    ])
  } finally {
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
