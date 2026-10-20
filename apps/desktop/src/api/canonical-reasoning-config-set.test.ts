// @vitest-environment node
import { expect, test } from 'vitest'

import { HermesGateway } from './client'

// B14: main's Desktop /reasoning <level> changed the session's reasoning through legacy config.set.
// On the shared owner the session scope is the revision-fenced `session.mutate operation=reasoning`;
// `--global` is the owner's `/reasoning` handler behind `config.set reasoning scope=global`.
test('typed /reasoning sets the session level through the owner mutation; --global reaches the owner', async () => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const sent: any[] = []
  server.on('connection', (socket: any) => {
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())
      sent.push(frame)
      const p = frame.params

      const result =
        frame.method === 'session.resume'
          ? { session_id: 's', revision: 4, execution_generation: 2, running: false, messages: [], prompts: [] }
          : frame.method === 'session.mutate'
            ? {
                session_id: p.session_id,
                operation: p.operation,
                revision: 5,
                execution_generation: 3,
                effort: p.payload.effort
              }
            : frame.method === 'config.set'
              ? { key: 'reasoning', value: p.value, scope: p.scope }
              : { value: 'medium' }

      socket.send(
        JSON.stringify(
          result
            ? { jsonrpc: '2.0', id: frame.id, result }
            : {
                jsonrpc: '2.0',
                id: frame.id,
                error: { code: 4001, message: 'invalid_params', data: { reason: 'invalid_params' } }
              }
        )
      )
    })
  })
  const client = new HermesGateway()

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?native_dial=fixture&ticket=one-use`)
    await client.request('session.resume', { session_id: 's' })

    // reasoningSlashParams('high', 's') and ('high --session', 's'): the result's `value` is what
    // applyReasoningSlashResult / writeSessionReasoningEffort read.
    await expect(
      client.request('config.set', { key: 'reasoning', session_id: 's', value: 'high' })
    ).resolves.toMatchObject({ value: 'high' })
    await expect(
      client.request('config.set', { key: 'reasoning', session_id: 's', value: 'low', scope: 'session' })
    ).resolves.toMatchObject({ value: 'low' })
    await expect(
      client.request('config.set', { key: 'reasoning', session_id: 's', value: 'high', scope: 'global' })
    ).resolves.toMatchObject({ value: 'high', scope: 'global' })

    const mutations = sent.filter(frame => frame.method === 'session.mutate').map(frame => frame.params)
    expect(mutations).toMatchObject([
      {
        session_id: 's',
        operation: 'reasoning',
        payload: { effort: 'high' },
        expected_revision: 4,
        expected_generation: 2
      },
      { session_id: 's', operation: 'reasoning', payload: { effort: 'low' }, expected_revision: 5 }
    ])
    expect(sent.filter(frame => frame.method === 'config.set').map(frame => frame.params)).toEqual([
      { key: 'reasoning', session_id: 's', value: 'high', scope: 'global' }
    ])
    // The bare form's read still reaches the owner.
    await expect(client.request('config.get', { key: 'reasoning', session_id: 's' })).resolves.toEqual({
      value: 'medium'
    })
  } finally {
    client.close()

    for (const socket of server.clients) {
      socket.terminate()
    }

    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
