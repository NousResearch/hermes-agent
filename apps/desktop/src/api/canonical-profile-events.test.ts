// @vitest-environment node
import { expect, test, vi } from 'vitest'

import { HermesGateway } from './client'

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
