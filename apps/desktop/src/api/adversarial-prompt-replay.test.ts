import { JsonRpcGatewayClient } from '@hermes/shared'
import { afterEach, expect, test, vi } from 'vitest'

import { HermesGateway } from './client'

afterEach(() => vi.restoreAllMocks())

test('an unanswered prompt is replayed after its response was lost and the socket reconnects', async () => {
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()

  const pending = { kind: 'approval', prompt_id: 'same-prompt', execution_generation: 2,
    command: 'build', choices: ['once', 'deny'] }

  vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async method => {
    if (method === 'approval.respond') {throw new Error('Gateway connection closed')}

    return { session_id: 's', revision: 1, execution_generation: 2, prompts: [pending] } as never
  })
  const client = new HermesGateway()
  const shown: any[] = []
  client.onRequest(request => { shown.push(request) })
  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=first')
  await client.request('session.resume', { session_id: 's' })
  shown[0].respond({ choice: 'once' })
  await Promise.resolve()
  await Promise.resolve()
  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=second')
  await client.request('session.resume', { session_id: 's' })
  expect(shown).toHaveLength(2)
  expect(shown[1].replayed).toBe(true)
  client.close()
})

test('failed answers recover an authoritative pending card once and callbacks cannot cross socket generations', async () => {
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()
  const prompt = { kind: 'approval', prompt_id: 'pending', execution_generation: 2, command: 'build', choices: ['once'] }

  const wire = vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async method => {
    if (method === 'approval.respond') { throw new Error('temporary transport failure') }

    return { session_id: 's', revision: 1, execution_generation: 2, prompts: [prompt] } as never
  })

  const client = new HermesGateway()
  const shown: any[] = []
  client.onRequest(request => { shown.push(request) })
  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=first')
  await client.request('session.resume', { session_id: 's', profile: 'sibling' })
  shown[0].respond({ choice: 'once' })
  await vi.waitFor(() => expect(shown).toHaveLength(2))
  await client.request('session.resume', { session_id: 's', profile: 'sibling' })
  expect(shown).toHaveLength(2)
  expect(wire.mock.calls.filter(([method]) => method === 'approval.respond')).toHaveLength(1)
  const old = shown[1]
  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=second')
  await client.request('session.resume', { session_id: 's', profile: 'sibling' })
  const before = wire.mock.calls.length
  old.respond({ choice: 'once' })
  await Promise.resolve()
  expect(wire).toHaveBeenCalledTimes(before)
  expect(shown).toHaveLength(3)
  client.close()
})
