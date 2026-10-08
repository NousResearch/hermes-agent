import { JsonRpcGatewayClient } from '@hermes/shared'
import { afterEach, expect, test, vi } from 'vitest'

import { HermesGateway } from './client'

afterEach(() => { vi.restoreAllMocks() })

test.each([60, 120])('compress follow-up spends only the remaining caller budget after %i ms', async elapsed => {
  let now = 1_000
  vi.spyOn(Date, 'now').mockImplementation(() => now)
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()
  const calls: Array<{ method: string; timeout?: number; signal?: AbortSignal }> = []
  const snapshot = { session_id: 's', revision: 1, execution_generation: 2, messages: [], info: {} }
  vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async (method, _params, timeout, signal) => {
    calls.push({ method, timeout, signal })

    if (method === 'session.mutate') {
      now += elapsed

      return { session_id: 's', revision: 2, operation: 'compress', message_count: 0 } as never
    }

    return snapshot as never
  })
  const client = new HermesGateway()
  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=fixture')
  await client.request('session.resume', { session_id: 's', profile: 'sibling' })
  calls.length = 0
  const controller = new AbortController()
  const compression = client.request('session.compress', { session_id: 's', profile: 'sibling' }, 100, controller.signal)

  if (elapsed < 100) {
    await expect(compression).resolves.toHaveProperty('messages')
    expect(calls).toEqual([
      { method: 'session.mutate', timeout: 100, signal: controller.signal },
      { method: 'session.resume', timeout: 40, signal: controller.signal }
    ])
  } else {
    await expect(compression).rejects.toThrow('timed out')
    expect(calls).toHaveLength(1)
  }

  client.close()
})
