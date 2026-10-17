// @vitest-environment node
import { expect, test } from 'vitest'

import { CanonicalDesktopProtocol } from './canonical-protocol'
import { HermesGateway } from './client'

test('model options use the acknowledged generation before mutation events arrive', async () => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))

  const owners: Record<string, { revision: number; execution_generation: number }> = {
    alpha: { revision: 4, execution_generation: 2 },
    beta: { revision: 20, execution_generation: 9 }
  }

  const accepted: string[] = []

  server.on('connection', (socket: any) => {
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())
      const p = frame.params
      const owner = owners[p.profile]

      if (frame.method === 'session.mutate') {
        const reason = p.expected_generation !== owner.execution_generation ? 'stale_generation'
          : p.expected_revision !== owner.revision ? 'revision_conflict' : null

        if (reason) {
          socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id,
            error: { code: 4001, message: reason, data: { reason } } }))

          return
        }

        owner.revision++
        owner.execution_generation++
        accepted.push(`${p.profile}:${p.operation}`)
      }

      // Deliberately send no event: the response must be enough for the next edit.
      socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result: {
        session_id: 'same', ...owner, messages: [], prompts: [], running: false,
        ...(p.operation === 'reasoning' ? { effort: p.payload.effort } : {})
      } }))
    })
  })
  const client = new HermesGateway()

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?native_dial=fixture&ticket=one-use`)

    for (const profile of ['alpha', 'beta']) {
      await client.request('session.resume', { session_id: 'same', profile })
    }

    await client.request('config.set', { session_id: 'same', profile: 'alpha', key: 'model', value: 'new-model --provider test' })
    await expect(client.request('config.set', { session_id: 'same', profile: 'alpha', key: 'reasoning', value: 'high' }))
      .resolves.toMatchObject({ value: 'high' })
    await expect(client.request('config.set', { session_id: 'same', profile: 'alpha', key: 'reasoning', value: 'low' }))
      .resolves.toMatchObject({ value: 'low' })
    await expect(client.request('config.set', { session_id: 'same', profile: 'beta', key: 'reasoning', value: 'medium' }))
      .resolves.toMatchObject({ value: 'medium' })
    expect(accepted).toEqual(['alpha:model', 'alpha:reasoning', 'alpha:reasoning', 'beta:reasoning'])
  } finally {
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test('replayed mutation receipts cannot rewind or cross a session generation', () => {
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's', profile: 'alpha' }, { session_id: 's', revision: 4, execution_generation: 2 })
  const params = protocol.prepare('config.set', { session_id: 's', profile: 'alpha', key: 'model', value: 'new-model --provider test' })
  const receipt = { session_id: 's', revision: 5, execution_generation: 3 }
  protocol.result('config.set', params, receipt)
  protocol.event({ type: 'session.info', session_id: 's', profile: 'alpha', payload: { revision: 7, execution_generation: 5 } })
  protocol.result('config.set', params, receipt)
  expect(protocol.prepare('session.interrupt', { session_id: 's', profile: 'alpha' }).execution_generation).toBe(5)
  expect(() => protocol.result('config.set', params, { ...receipt, session_id: 'other', execution_generation: 99 }))
    .toThrow('Metadata receipt destination mismatch')
  expect(protocol.prepare('session.interrupt', { session_id: 's', profile: 'alpha' }).execution_generation).toBe(5)
})
