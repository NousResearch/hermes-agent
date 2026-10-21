// @vitest-environment node
import { expect, test } from 'vitest'

import { HermesGateway } from './client'

// R5: the primary socket serves every multiplexed profile and the authority selects one PER
// REQUEST from `profile`. A sibling session's rebuilt requests (CAS mutations, prompt replies,
// the compress follow-up resume) must keep the route, or they reach the launch profile.
test('a sibling profile route survives canonical request translation', async () => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const sent: any[] = []
  server.on('connection', (socket: any) => {
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())
      sent.push(frame)

      const snapshot = { session_id: 's', stored_session_id: 's', revision: 2, execution_generation: 5, messages: [],
        prompts: [{ kind: 'approval', prompt_id: 'p1', execution_generation: 5, command: 'rm -rf build', choices: ['once', 'deny'] }] }

      const result = frame.method === 'session.resume' ? snapshot
        : frame.method === 'session.mutate' ? { session_id: 's', revision: 3, operation: frame.params.operation, title: 'Renamed', message_count: 1 }
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
    await client.request('session.resume', { session_id: 's', profile: 'sibling' })
    await client.request('session.title', { session_id: 's', title: 'Renamed', profile: 'sibling' })
    await client.request('session.compress', { session_id: 's', profile: 'sibling' })
    delivered[0]?.respond({ choice: 'once' })
    await new Promise(resolve => setTimeout(resolve, 50))

    expect(sent.map(frame => [frame.method, frame.params.profile])).toEqual([
      ['session.resume', 'sibling'],
      ['session.mutate', 'sibling'],
      ['session.mutate', 'sibling'],
      ['session.resume', 'sibling'],
      ['approval.respond', 'sibling']
    ])
  } finally {
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
