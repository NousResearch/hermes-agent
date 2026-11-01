// @vitest-environment node
import { expect, test } from 'vitest'

import { CanonicalDesktopProtocol } from './canonical-protocol'
import { HermesGateway } from './client'

// The desktop's model-option pill (Effort / Fast) writes `config.set key=reasoning`, which the
// canonical protocol sends as `session.mutate operation=reasoning` fenced by the session's
// execution generation. The owner raises that generation on every accepted policy edit, and the
// desktop only learned the new value from the `session.updated` broadcast, so a lost broadcast
// left the next edit sending the pre-mutation generation: the owner refused it `stale_generation`
// and the composer reported "Model option update failed".
test('the client adopts the generation its own mutation receipt reports', () => {
  const protocol = new CanonicalDesktopProtocol()

  protocol.result('session.resume', { session_id: 's' }, {
    session_id: 's',
    revision: 0,
    execution_generation: 0,
    running: false,
    messages: [],
    prompts: []
  })

  const first = protocol.prepare('config.set', { key: 'reasoning', session_id: 's', value: 'high' })

  expect(first).toMatchObject({ operation: 'reasoning', expected_revision: 0, expected_generation: 0 })

  // The owner committed it: the receipt names the generation it reached. No `session.updated`
  // broadcast is delivered here, which is the loss this test exists to survive.
  protocol.result('config.set', first, {
    session_id: 's',
    revision: 1,
    execution_generation: 1,
    operation: 'reasoning',
    effort: 'high'
  })

  const second = protocol.prepare('config.set', { key: 'reasoning', session_id: 's', value: 'low' })

  expect(second).toMatchObject({ expected_revision: 1, expected_generation: 1 })
})

test('a lost session.updated broadcast leaves the next option edit acceptable to the owner', async () => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))

  // The owner's real fence: a mutation whose expected_generation is behind is refused
  // `stale_generation` (hermes_state_runtime.py::mutate_runtime_session), and the accepted
  // mutation raises both counters. Nothing here broadcasts `session.updated`.
  const state = { revision: 0, generation: 0 }

  server.on('connection', (socket: any) => {
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())
      const params = frame.params ?? {}
      let result: any
      let error: any

      if (frame.method === 'session.resume') {
        result = {
          session_id: 's',
          revision: state.revision,
          execution_generation: state.generation,
          running: false,
          messages: [],
          prompts: []
        }
      } else if (frame.method === 'session.mutate' && params.operation === 'reasoning') {
        if (params.expected_revision !== state.revision) {
          error = { code: 4001, message: 'revision_conflict', data: { reason: 'revision_conflict' } }
        } else if (params.expected_generation !== state.generation) {
          error = { code: 4001, message: 'stale_generation', data: { reason: 'stale_generation' } }
        } else {
          state.revision += 1
          state.generation += 1
          result = {
            session_id: 's',
            revision: state.revision,
            execution_generation: state.generation,
            operation: 'reasoning',
            effort: params.payload.effort
          }
        }
      } else {
        result = { value: 'medium' }
      }

      socket.send(
        JSON.stringify(
          error ? { jsonrpc: '2.0', id: frame.id, error } : { jsonrpc: '2.0', id: frame.id, result }
        )
      )
    })
  })

  const client = new HermesGateway()

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?native_dial=fixture&ticket=one-use`)
    await client.request('session.resume', { session_id: 's' })

    await expect(
      client.request('config.set', { key: 'reasoning', session_id: 's', value: 'high' })
    ).resolves.toMatchObject({ value: 'high' })
    await expect(
      client.request('config.set', { key: 'reasoning', session_id: 's', value: 'low' })
    ).resolves.toMatchObject({ value: 'low' })
  } finally {
    client.close()

    for (const socket of server.clients) {
      socket.terminate()
    }

    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
