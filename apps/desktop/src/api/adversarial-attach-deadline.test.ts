import { JsonRpcGatewayClient } from '@hermes/shared'
import { afterEach, expect, test, vi } from 'vitest'

import { HermesGateway } from './client'

afterEach(() => { vi.restoreAllMocks() })

test('the prerequisite attachment shares the caller deadline and cancellation', async () => {
  let now = 1_000
  vi.spyOn(Date, 'now').mockImplementation(() => now)
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()
  const calls: Array<{ method: string; timeout?: number; signal?: AbortSignal }> = []
  vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async (method, _params, timeout, signal) => {
    calls.push({ method, timeout, signal })

    if (method === 'session.resume') {
      now += 60

      return { session_id: 's', revision: 1, execution_generation: 2, messages: [], info: {} } as never
    }

    return { ref: { session_id: 's' }, admission_id: 'a', status: 'queued' } as never
  })
  const client = new HermesGateway()
  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=fixture')
  const controller = new AbortController()

  try {
    await client.request('prompt.submit', { session_id: 's', submission_id: 'one', text: 'hello' }, 100, controller.signal)
    expect(calls).toEqual([
      { method: 'session.resume', timeout: 100, signal: controller.signal },
      { method: 'prompt.submit', timeout: 40, signal: controller.signal }
    ])
  } finally { client.close() }
})



test('a prerequisite attachment resolves the launch profile alias before preparing a mutation', async () => {
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()
  const calls: Array<{ method: string; params: unknown }> = []
  vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async (method, params) => {
    calls.push({ method, params })

    if (method === 'session.resume') {
      return { session_id: 's', revision: 3, execution_generation: 2, replay_epoch: 'named-owner',
        messages: [], info: { profile_name: 'work', profile_id: '/profiles/work' } } as never
    }

    return { session_id: 's', revision: 4, operation: 'branch', branched_session_id: 'child', copied_messages: 0 } as never
  })
  const client = new HermesGateway()
  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=fixture')

  try {
    await expect(client.request('session.branch_stored', { parent_session_id: 's' })).resolves.toMatchObject({ session_id: 'child' })
    expect(calls[1]).toMatchObject({ method: 'session.mutate', params: { profile: 'work', expected_revision: 3 } })
  } finally { client.close() }
})
