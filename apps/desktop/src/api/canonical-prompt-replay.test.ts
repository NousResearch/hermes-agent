import { JsonRpcGatewayClient } from '@hermes/shared'
import { afterEach, expect, test, vi } from 'vitest'

import { HermesGateway } from './client'

afterEach(() => vi.restoreAllMocks())

test.each(['approval', 'clarify'] as const)('%s remains replayable when no renderer was listening', async kind => {
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()
  vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockResolvedValue({
    session_id: 'session', revision: 1, execution_generation: 2,
    prompts: [{ kind, prompt_id: 'pending', execution_generation: 2,
      command: 'build', choices: ['once'], question: 'Continue?' }]
  } as never)
  const client = new HermesGateway()

  try {
    await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=first')
    await client.request('session.resume', { session_id: 'session', profile: 'alpha' })
    const shown = vi.fn()
    client.onRequest(shown)
    await client.request('session.resume', { session_id: 'session', profile: 'alpha' })
    await client.request('session.resume', { session_id: 'session', profile: 'alpha' })
    expect(shown).toHaveBeenCalledTimes(1)
    expect(shown.mock.calls[0][0]).toMatchObject({ id: 'pending', method: kind, replayed: true })
  } finally {
    client.close()
  }
})

test.each(['approval', 'clarify'] as const)('%s remains replayable when every renderer declined it', async kind => {
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()
  vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockResolvedValue({
    session_id: 'session', revision: 1, execution_generation: 2,
    prompts: [{ kind, prompt_id: 'pending', execution_generation: 2,
      command: 'build', choices: ['once'], question: 'Continue?' }]
  } as never)
  const client = new HermesGateway()
  const decline = vi.fn(() => false)
  client.onRequest(decline)

  try {
    await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=first')
    await client.request('session.resume', { session_id: 'session', profile: 'alpha' })
    const shown = vi.fn()
    client.onRequest(shown)
    await client.request('session.resume', { session_id: 'session', profile: 'alpha' })
    await client.request('session.resume', { session_id: 'session', profile: 'alpha' })
    expect(decline).toHaveBeenCalledTimes(2)
    expect(shown).toHaveBeenCalledTimes(1)
  } finally {
    client.close()
  }
})
