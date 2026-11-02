// @vitest-environment node
import { expect, test } from 'vitest'

import { HermesGateway } from './client'

// The owner asks a multi-question clarify as ONE request (the card's single form) only for a turn
// whose submit declared `clarify_batch`; an older owner's closed submit params refuse the key, so the
// Desktop declares it only where `client.capabilities` advertised `canonical_clarify_batch`.
async function submitThrough(capabilities: Record<string, unknown> | null, dial = 'native_dial=fixture&ticket=t') {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const submits: Record<string, unknown>[] = []
  let advertised = 0

  server.on('connection', (socket: any) => {
    socket.send(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'gateway.ready', payload: {} } }))
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())
      const reply = (result: unknown) => socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }))

      if (frame.method === 'client.capabilities') {
        if (frame.params.server_requests) { advertised++ }

        return capabilities
          ? reply({ server_requests: ['clarify'], ...capabilities })
          : socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, error: { code: -32601, message: 'unknown method' } }))
      }

      if (frame.method === 'session.resume') {
        return reply({ session_id: 's', revision: 1, execution_generation: 1, replay_epoch: 'e', last_sequence: 0, prompts: [] })
      }

      if (frame.method === 'prompt.submit') { submits.push(frame.params) }

      reply({ admission_id: `a${submits.length}`, ref: { session_id: 's' }, sequence: 1, status: 'queued', outcome: null,
        authority_epoch: 1, execution_generation: 1 })
    })
  })
  const client = new HermesGateway()

  try {
    await client.connect(`ws://127.0.0.1:${(server.address() as { port: number }).port}/api/ws?${dial}`)
    await expect.poll(() => advertised).toBe(1)
    // Let the capability answer land before the turn, as the app's boot does.
    await new Promise(resolve => setTimeout(resolve, 20))
    await client.request('prompt.submit', { session_id: 's', text: 'one' })
    await client.request('prompt.submit', { session_id: 's', text: 'two' })

    return submits
  } finally {
    client.close()

    for (const socket of server.clients) { socket.terminate() }

    await new Promise<void>(resolve => server.close(() => resolve()))
  }
}

test('every turn declares clarify_batch to an owner advertising canonical_clarify_batch', async () => {
  const submits = await submitThrough({ declines_not_shown: true, canonical_clarify_batch: true })
  expect(submits.map(params => params.clarify_batch)).toEqual([true, true])
})

test('an older owner (no field, or no client.capabilities at all) never sees the key', async () => {
  for (const capabilities of [{ declines_not_shown: true }, null]) {
    const submits = await submitThrough(capabilities)
    expect(submits).toHaveLength(2)
    expect(submits.some(params => 'clarify_batch' in params)).toBe(false)
  }
})
