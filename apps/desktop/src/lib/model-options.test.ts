import { afterEach, describe, expect, it, vi } from 'vitest'

import { getGlobalModelOptions } from '@/hermes'

import { requestModelOptions } from './model-options'

const globalOptions = { model: 'hermes-4', provider: 'nous', providers: [] }

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: vi.fn(() => Promise.resolve(globalOptions))
}))

describe('requestModelOptions', () => {
  afterEach(() => {
    vi.clearAllMocks()
  })

  it('uses the connected gateway even before a session exists', async () => {
    const gatewayPayload = { model: 'BeastMode', provider: 'moa', providers: [] }

    const gateway = {
      request: vi.fn(() => Promise.resolve(gatewayPayload))
    }

    await expect(requestModelOptions({ gateway: gateway as never, sessionId: null })).resolves.toBe(gatewayPayload)

    expect(gateway.request).toHaveBeenCalledWith('model.options', { explicit_only: true })
    expect(getGlobalModelOptions).not.toHaveBeenCalled()
  })

  it('merges the global gateway catalog into session-scoped results', async () => {
    const gatewayGlobal = {
      model: 'gpt-5.4',
      provider: 'openai-codex',
      providers: [
        { models: ['gpt-5.4'], name: 'OpenAI Codex', slug: 'openai-codex' },
        { models: ['MiniMax M3'], name: 'MiniMax', slug: 'minimax' }
      ]
    }
    const gatewayScoped = {
      model: 'gpt-5.4',
      provider: 'openai-codex',
      providers: [{ models: ['MiniMax M3'], name: 'MiniMax', slug: 'minimax' }]
    }
    const gateway = {
      request: vi
        .fn()
        .mockResolvedValueOnce(gatewayGlobal)
        .mockResolvedValueOnce(gatewayScoped)
    }

    await expect(requestModelOptions({ gateway: gateway as never, refresh: true, sessionId: 'session-1' })).resolves
      .toEqual({
        ...gatewayScoped,
        providers: [
          { models: ['gpt-5.4'], name: 'OpenAI Codex', slug: 'openai-codex' },
          { models: ['MiniMax M3'], name: 'MiniMax', slug: 'minimax' }
        ]
      })

    expect(gateway.request).toHaveBeenNthCalledWith(1, 'model.options', {
      explicit_only: true,
      refresh: true
    })
    expect(gateway.request).toHaveBeenNthCalledWith(2, 'model.options', {
      explicit_only: true,
      refresh: true,
      session_id: 'session-1'
    })
  })

  it('falls back to REST when no gateway is connected', async () => {
    await requestModelOptions({ refresh: true })

    expect(getGlobalModelOptions).toHaveBeenCalledWith({ explicitOnly: true, refresh: true })
  })
})
