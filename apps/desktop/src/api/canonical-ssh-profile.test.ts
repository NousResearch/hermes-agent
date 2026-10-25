// @vitest-environment node
import { expect, test } from 'vitest'

import { canonicalOwnerProfile } from './canonical-protocol'
import { HermesGateway } from './client'

test('an SSH display alias routes its first create and subsequent mutation to the backend profile', async () => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const sent: Array<{ method: string; params: Record<string, unknown> }> = []
  server.on('connection', (socket: any) => socket.on('message', (bytes: Buffer) => {
    const frame = JSON.parse(bytes.toString())

    if (frame.method === 'client.capabilities') {
      socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result: { server_requests: [] } }))

      return
    }

    sent.push(frame)

    const snapshot = {
      session_id: 's', stored_session_id: 's', revision: 2, execution_generation: 5, messages: [], prompts: [],
      info: { profile_name: 'writer', profile_id: '/remote/profiles/writer' }
    }

    const result = frame.method === 'session.create' ? snapshot
      : { session_id: 's', revision: 3, operation: 'title', title: 'Renamed' }

    socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }))

    if (frame.method === 'session.mutate') {
      socket.send(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: {
        type: 'session.info', session_id: 's', profile: 'writer', payload: snapshot.info
      } }))
    }
  }))

  const client = new HermesGateway()
  const events: any[] = []
  client.on('session.info', event => { events.push(event) })

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?native_dial=fixture&ticket=fixture`, {
      profile: 'default', remoteProfile: 'writer'
    })
    const created = await client.request<{ info: { profile_name: string } }>('session.create', { profile: 'default', source: 'desktop' })
    expect(created.info.profile_name).toBe('default')
    await client.request('session.title', { session_id: 's', profile: 'default', title: 'Renamed' })
    await new Promise(resolve => setTimeout(resolve, 30))
    expect(sent.map(frame => [frame.method, frame.params.profile])).toEqual([
      ['session.create', 'writer'], ['session.mutate', 'writer']
    ])
    expect(events[0].profile).toBe('default')
    expect(canonicalOwnerProfile(events[0])).toBe('default')
  } finally {
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
