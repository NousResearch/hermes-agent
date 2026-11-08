import { JsonRpcGatewayClient } from '@hermes/shared'
import { afterEach, expect, it, vi } from 'vitest'

import { HermesGateway } from './client'

afterEach(() => vi.restoreAllMocks())

it('detaches the canonical subscription and reattaches before a later submit', async () => {
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()

  const wire = vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async method => {
    if (method === 'session.resume') { return { session_id: 's', stored_session_id: 's', subscription_id: 'viewer', revision: 1 } }

    if (method === 'prompt.submit') { return { ref: { session_id: 's' }, admission_id: 'turn', status: 'queued' } }

    return { session_id: 's', subscription_id: 'viewer', detached: true }
  })

  const client = new HermesGateway()
  await client.connect('ws://127.0.0.1/api/ws?native_dial=fixture&ticket=one-use')
  await client.request('session.resume', { session_id: 's', profile: 'a' })
  await client.request('session.detach', { session_id: 's', profile: 'a' })
  expect(wire).toHaveBeenLastCalledWith('session.detach', { session_id: 's', profile: 'a', subscription_id: 'viewer' }, expect.any(Number), undefined)
  wire.mockClear()
  await client.request('prompt.submit', { session_id: 's', profile: 'a', text: 'hello' })
  expect(wire.mock.calls.map(call => call[0])).toEqual(['session.resume', 'prompt.submit'])
})

it('a late detach receipt cannot clear a reopened viewer even if the owner reused its subscription id', async () => {
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()
  let finishDetach!: (result: unknown) => void

  const wire = vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async method => {
    if (method === 'session.detach') { return new Promise(resolve => { finishDetach = resolve }) }

    if (method === 'session.resume') { return { session_id: 's', stored_session_id: 's', subscription_id: 'reused', revision: 1 } }

    return { ref: { session_id: 's' }, admission_id: 'turn', status: 'queued' }
  })

  const client = new HermesGateway()
  await client.connect('ws://127.0.0.1/api/ws?native_dial=fixture&ticket=one-use')
  await client.request('session.resume', { session_id: 's', profile: 'a' })
  const detached = client.request('session.detach', { session_id: 's', profile: 'a' })
  await vi.waitFor(() => expect(finishDetach).toBeTypeOf('function'))
  await client.request('session.resume', { session_id: 's', profile: 'a' })
  finishDetach({ detached: true })
  await detached
  wire.mockClear()
  await client.request('prompt.submit', { session_id: 's', profile: 'a', text: 'hello' })
  expect(wire.mock.calls.map(call => call[0])).toEqual(['prompt.submit'])
})
