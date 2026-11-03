import { JsonRpcGatewayClient, JsonRpcGatewayError } from '@hermes/shared'
import { afterEach, expect, test, vi } from 'vitest'

import { HermesGateway } from './client'

afterEach(() => { vi.restoreAllMocks() })

async function attached(mutate: (params: Record<string, unknown>) => unknown) {
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []
  vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async (method, params) => {
    calls.push({ method, params: params as Record<string, unknown> })

    if (method === 'session.mutate') { return mutate(params as Record<string, unknown>) as never }

    return { session_id: 's', revision: 4, execution_generation: 2, messages: [], info: {} } as never
  })
  const client = new HermesGateway()
  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=fixture')
  await client.request('session.resume', { session_id: 's' })
  calls.length = 0

  return { calls, client }
}

const pick = { session_id: 's', key: 'model', value: 'picked --provider custom' }

test('a composer pick is the session-scoped mutate; a busy session answers deferred for the next turn', async () => {
  const { calls, client } = await attached(params => ({
    session_id: 's', operation: 'model', status: 'deferred', model: (params.payload as { model: string }).model, revision: 4
  }))

  await expect(client.request('config.set', pick)).resolves.toMatchObject({ deferred: true, ok: true })
  expect(calls).toEqual([{ method: 'session.mutate', params: expect.objectContaining({
    operation: 'model', payload: { model: 'picked', provider: 'custom' }, expected_revision: 4, expected_generation: 2 }) }])
})

test('a fence refusal naming fresh fences is retried once at them, never looped', async () => {
  const stale = () => { throw new JsonRpcGatewayError('stale_generation', { code: 4001, data: { reason: 'stale_generation', revision: 6, execution_generation: 3 } }) }
  let attempts = 0

  const { calls, client } = await attached(params => {
    attempts += 1

    return attempts === 1 ? stale() : { session_id: 's', operation: 'model', model: 'picked', revision: (params.expected_revision as number) + 1 }
  })

  await expect(client.request('config.set', pick)).resolves.toMatchObject({ model: 'picked' })
  expect(calls.map(call => [call.params.expected_revision, call.params.expected_generation])).toEqual([[4, 2], [6, 3]])
  expect(calls[1].params.request_id).not.toBe(calls[0].params.request_id)

  attempts = 0
  calls.length = 0
  const { calls: looping, client: refused } = await attached(stale)
  await expect(refused.request('config.set', pick)).rejects.toThrow('stale_generation')
  expect(looping).toHaveLength(2)
})

test('an older pick refused after a newer pick committed is not retried and never rewinds the fences', async () => {
  let refuseA: (error: unknown) => void = () => {}

  const { calls, client } = await attached(params => {
    const model = (params.payload as { model: string }).model

    if (model === 'a' && calls.filter(call => (call.params.payload as { model: string }).model === 'a').length === 1) {
      return new Promise((_resolve, reject) => { refuseA = reject })
    }

    return { session_id: 's', operation: 'model', model, revision: 7, execution_generation: 4 }
  })

  const a = client.request('config.set', { ...pick, value: 'a --provider custom' })
  await vi.waitFor(() => expect(calls).toHaveLength(1))
  await expect(client.request('config.set', { ...pick, value: 'b --provider custom' })).resolves.toMatchObject({ model: 'b' })
  // A's stale-fence reply arrives after B committed, naming the fences B moved past.
  refuseA(new JsonRpcGatewayError('revision_conflict', { code: 4001, data: { reason: 'revision_conflict', revision: 5, execution_generation: 3 } }))
  await expect(a).rejects.toThrow('revision_conflict')
  expect(calls.map(call => (call.params.payload as { model: string }).model)).toEqual(['a', 'b'])

  await client.request('config.set', { ...pick, value: 'c --provider custom' })
  // B's receipt moved the revision to 7; A's late refusal naming 5 must not rewind it.
  expect(calls[2].params).toMatchObject({ expected_revision: 7 })
})

test("a guarded next-turn pick's confirmed resend carries the owner's token from its session.updated ask", async () => {
  const { calls, client } = await attached(params => ({ session_id: 's', operation: 'model', model: (params.payload as { model: string }).model, revision: 5 }))
  const protocol = (client as unknown as { protocol: { event: (event: Record<string, unknown>) => void } }).protocol

  protocol.event({ type: 'session.updated', session_id: 's', payload: {
    session_id: 's', operation: 'model', status: 'confirmation_required', confirm: 'owner-token', payload: { model: 'guarded', provider: 'custom' } } })
  await client.request('config.set', { ...pick, value: 'guarded --provider custom --session', confirm_expensive_model: true })

  expect(calls[0].params.payload).toEqual({ model: 'guarded', provider: 'custom', confirm: 'owner-token' })
})

test("a deferred pick's confirmation resends its whole selection (reasoning included) with the owner's token", async () => {
  const { calls, client } = await attached(params => ({ session_id: 's', operation: 'model', model: (params.payload as { model: string }).model, revision: 5 }))

  await client.request('session.model', { session_id: 's', payload: { model: 'guarded', provider: 'custom', reasoning: 'high' }, confirm: 'owner-token' })

  expect(calls).toEqual([{ method: 'session.mutate', params: expect.objectContaining({
    operation: 'model', payload: { model: 'guarded', provider: 'custom', reasoning: 'high', confirm: 'owner-token' }, expected_generation: 2 }) }])
})
