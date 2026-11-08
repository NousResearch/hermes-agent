import { JsonRpcGatewayClient } from '@hermes/shared'
import { afterEach, expect, it, vi } from 'vitest'

import { HermesGateway } from './client'

afterEach(() => vi.restoreAllMocks())

it('canonical closeout attaches its exact owner and preserves the durable request identity on the existing method', async () => {
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()

  const wire = vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async method => {
    if (method === 'session.resume') {
      return { session_id: 'old', stored_session_id: 'old', subscription_id: 'viewer', revision: 1 }
    }

    return { closeout_id: 'durable', status: 'reviewing' }
  })

  const client = new HermesGateway()
  await client.connect('ws://127.0.0.1/api/ws?native_dial=fixture&ticket=one-use')
  await client.request('session.closeout', {
    session_id: 'old',
    profile: 'work',
    request_id: 'retained',
    intent: 'archive'
  })
  expect(wire.mock.calls.map(call => call[0])).toEqual(['session.resume', 'session.closeout'])
  expect(wire.mock.calls[1]?.[1]).toEqual({
    session_id: 'old',
    profile: 'work',
    request_id: 'retained',
    intent: 'archive'
  })
})

it('legacy closeout keeps the same request identity without a plain-archive fallback', async () => {
  const wire = vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockResolvedValue({ status: 'reviewing' })
  const client = new HermesGateway()
  await client.request('session.closeout', {
    session_id: 'runtime',
    profile: 'work',
    request_id: 'retained',
    intent: 'new'
  })
  expect(wire).toHaveBeenCalledExactlyOnceWith(
    'session.closeout',
    { session_id: 'runtime', profile: 'work', request_id: 'retained', intent: 'new' },
    expect.any(Number),
    undefined
  )
})
