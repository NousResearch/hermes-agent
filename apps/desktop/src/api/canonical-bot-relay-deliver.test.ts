// @vitest-environment node
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { expect, test } from 'vitest'

import { HermesGateway } from './client'

// The canonical dispatcher refuses any key outside CanonicalBotDeliverParams (4001
// invalid_params, handler never called). Read that closed set from the Python contract so the
// fixture owner refuses exactly what the real one does.
function canonicalDeliverKeys(): Set<string> {
  const source = readFileSync(join(process.cwd(), '..', '..', 'tui_gateway/contracts/canonical_projections.py'), 'utf8')
  const body = source.match(/class CanonicalBotDeliverParams\(Params\):\n((?: {4}\w+:.*\n)+)/)

  expect(body, 'CanonicalBotDeliverParams must exist in canonical_projections.py').toBeTruthy()

  return new Set([...body![1].matchAll(/^ {4}(\w+):/gm)].map(match => match[1]))
}

test('a Bot Relay delivery on a canonical dial carries only keys the owner admits, sender as a bot author', async () => {
  const allowed = canonicalDeliverKeys()
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const sent: any[] = []
  server.on('connection', (socket: any) => {
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())
      sent.push(frame)
      const fields = Object.keys(frame.params).filter(key => !allowed.has(key))

      socket.send(
        JSON.stringify(
          fields.length
            ? {
                jsonrpc: '2.0',
                id: frame.id,
                error: { code: 4001, message: 'invalid_params', data: { reason: 'invalid_params', fields } }
              }
            : {
                jsonrpc: '2.0',
                id: frame.id,
                result: { status: 'settled', delivery_id: frame.params.id, admission_id: 'a1', reply: 'ok' }
              }
        )
      )
    })
  })
  const client = new HermesGateway()

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?native_dial=fixture&ticket=one-use`)
    // Exactly what hermes-bots/relay.ts::relayDeliverParams sends.
    await expect(
      client.request('bot_relay.deliver', {
        id: 'env-1',
        profile: 'ops',
        message: 'Message from 🤖 Scout (@scout): status?',
        from_profile: 'scout',
        from_handle: 'scout',
        from_connection: 'laptop'
      })
    ).resolves.toMatchObject({ status: 'settled', delivery_id: 'env-1' })
    expect(sent.at(-1).params).toEqual({
      id: 'env-1',
      profile: 'ops',
      message: 'Message from 🤖 Scout (@scout@laptop): status?',
      author: { id: 'bot:laptop/scout', name: 'scout', is_bot: true }
    })
  } finally {
    client.close()

    for (const socket of server.clients) {
      socket.terminate()
    }

    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
