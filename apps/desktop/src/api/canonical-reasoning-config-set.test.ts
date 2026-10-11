// @vitest-environment node
import { expect, test } from 'vitest'

import { HermesGateway } from './client'

// The canonical config.set key set (tui_gateway/contracts/canonical_projections.py::
// CanonicalConfigSetParams); the real owner answers any other key with 4001 invalid_params.
const CANONICAL_CONFIG_KEYS = new Set(['busy', 'verbose', 'yolo', 'model'])

test('typed /reasoning on a canonical dial is refused by name and never sent as a bare invalid_params', async () => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const sent: any[] = []
  server.on('connection', (socket: any) => {
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())
      sent.push(frame)
      const refused = frame.method === 'config.set' && !CANONICAL_CONFIG_KEYS.has(frame.params.key)

      socket.send(
        JSON.stringify(
          refused
            ? {
                jsonrpc: '2.0',
                id: frame.id,
                error: { code: 4001, message: 'invalid_params', data: { reason: 'invalid_params', fields: ['key'] } }
              }
            : { jsonrpc: '2.0', id: frame.id, result: { value: 'medium' } }
        )
      )
    })
  })
  const client = new HermesGateway()

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?native_dial=fixture&ticket=one-use`)

    // reasoningSlashParams('high --global', 's') — the Desktop /reasoning write.
    for (const params of [
      { key: 'reasoning', session_id: 's', value: 'high' },
      { key: 'reasoning', session_id: 's', value: 'high', scope: 'global' }
    ]) {
      await expect(client.request('config.set', params)).rejects.toThrow(/not available on the shared gateway yet/)
    }

    expect(sent.filter(frame => frame.method === 'config.set')).toEqual([])
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
