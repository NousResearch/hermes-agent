import { afterEach, describe, expect, it, vi } from 'vitest'

import { getGlobalModelOptions } from '@/hermes'

import { manualPickRemoved, modelOptionsQueryKey, requestModelOptions } from './model-options'

const globalOptions = { model: 'hermes-4', provider: 'nous', providers: [] }

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: vi.fn(() => Promise.resolve(globalOptions))
}))

describe('requestModelOptions', () => {
  afterEach(() => {
    vi.clearAllMocks()
  })

  it('uses the connected gateway even before a session exists', async () => {
    const gatewayPayload = {
      model: 'BeastMode',
      provider: 'moa',
      providers: [{ models: ['BeastMode'], name: 'Mixture of Agents', slug: 'moa' }]
    }

    const gateway = {
      request: vi.fn(() => Promise.resolve(gatewayPayload))
    }

    await expect(requestModelOptions({ gateway: gateway as never, sessionId: null })).resolves.toBe(gatewayPayload)

    expect(gateway.request).toHaveBeenCalledWith('model.options', { explicit_only: true })
    expect(getGlobalModelOptions).not.toHaveBeenCalled()
  })

  it('recovers an empty gateway catalog through profile-scoped REST without replacing the session selection', async () => {
    const gatewayPayload = { model: 'hermes-local', provider: 'hermes-local' }

    const restPayload = {
      model: 'profile-default',
      provider: 'openai-codex',
      providers: [{ models: ['hermes-local'], name: 'Hermes Local vLLM', slug: 'hermes-local' }]
    }

    const gateway = {
      request: vi.fn(() => Promise.resolve(gatewayPayload))
    }

    vi.mocked(getGlobalModelOptions).mockResolvedValueOnce(restPayload)

    await expect(requestModelOptions({ gateway: gateway as never, sessionId: 'session-1' })).resolves.toEqual({
      ...restPayload,
      model: 'hermes-local',
      provider: 'hermes-local'
    })

    expect(getGlobalModelOptions).toHaveBeenCalledWith({ explicitOnly: true })
  })

  it('recovers through profile-scoped REST when the gateway catalog request fails', async () => {
    const restPayload = {
      model: 'hermes-local',
      provider: 'hermes-local',
      providers: [{ models: ['hermes-local'], name: 'Hermes Local vLLM', slug: 'hermes-local' }]
    }

    const gateway = {
      request: vi.fn(() => Promise.reject(new Error('gateway request unavailable')))
    }

    vi.mocked(getGlobalModelOptions).mockResolvedValueOnce(restPayload)

    await expect(requestModelOptions({ gateway: gateway as never, sessionId: 'session-1' })).resolves.toEqual(
      restPayload
    )
    expect(getGlobalModelOptions).toHaveBeenCalledWith({ explicitOnly: true })
  })

  it('preserves the gateway error when its REST recovery path also fails', async () => {
    const gatewayError = new Error('gateway request unavailable')

    const gateway = {
      request: vi.fn(() => Promise.reject(gatewayError))
    }

    vi.mocked(getGlobalModelOptions).mockRejectedValueOnce(new Error('REST request unavailable'))

    await expect(requestModelOptions({ gateway: gateway as never })).rejects.toBe(gatewayError)
  })

  it('keeps the gateway result when both catalog paths have no selectable models', async () => {
    const gatewayPayload = { model: 'hermes-local', provider: 'hermes-local', providers: [] }

    const gateway = {
      request: vi.fn(() => Promise.resolve(gatewayPayload))
    }

    await expect(requestModelOptions({ gateway: gateway as never })).resolves.toBe(gatewayPayload)
  })

  it('passes the active session id and refresh flag through the gateway', async () => {
    const gateway = {
      request: vi.fn(() => Promise.resolve(globalOptions))
    }

    await requestModelOptions({ gateway: gateway as never, refresh: true, sessionId: 'session-1' })

    expect(gateway.request).toHaveBeenCalledWith('model.options', {
      explicit_only: true,
      refresh: true,
      session_id: 'session-1'
    })
    expect(getGlobalModelOptions).toHaveBeenCalledWith({ explicitOnly: true, refresh: true })
  })

  it('falls back to REST when no gateway is connected', async () => {
    await requestModelOptions({ refresh: true })

    expect(getGlobalModelOptions).toHaveBeenCalledWith({ explicitOnly: true, refresh: true })
  })

  it('prefers an owner-routed request over the ambient gateway socket', async () => {
    const gatewayPayload = {
      model: 'chrome-model',
      provider: 'nous',
      providers: [{ models: ['chrome-model'], name: 'Nous', slug: 'nous' }]
    }
    const routedPayload = {
      model: 'berry-model',
      provider: 'openai',
      providers: [{ models: ['berry-model'], name: 'OpenAI', slug: 'openai' }]
    }
    const gateway = {
      request: vi.fn(() => Promise.resolve(gatewayPayload))
    }
    const request = vi.fn(() => Promise.resolve(routedPayload))

    await expect(
      requestModelOptions({ gateway: gateway as never, request, sessionId: 'tile-1' })
    ).resolves.toBe(routedPayload)

    expect(request).toHaveBeenCalledWith('model.options', { explicit_only: true, session_id: 'tile-1' })
    expect(gateway.request).not.toHaveBeenCalled()
  })

  it('scopes REST recovery to the catalog owner profile', async () => {
    const restPayload = {
      model: 'berry-local',
      provider: 'hermes-local',
      providers: [{ models: ['berry-local'], name: 'Hermes Local', slug: 'hermes-local' }]
    }
    const request = vi.fn(() => Promise.reject(new Error('gateway request unavailable')))

    vi.mocked(getGlobalModelOptions).mockResolvedValueOnce(restPayload)

    await expect(requestModelOptions({ profile: 'berry', request, sessionId: 'tile-1' })).resolves.toEqual(
      restPayload
    )
    expect(getGlobalModelOptions).toHaveBeenCalledWith({ explicitOnly: true }, 'berry')
  })
})

describe('modelOptionsQueryKey', () => {
  it('isolates new-chat catalogs by active gateway profile', () => {
    expect(modelOptionsQueryKey('default')).toEqual(['model-options', 'default', 'global'])
    expect(modelOptionsQueryKey('compass')).toEqual(['model-options', 'compass', 'global'])
    expect(modelOptionsQueryKey('default')).not.toEqual(modelOptionsQueryKey('compass'))
  })

  it('keeps session catalogs inside the owning profile namespace', () => {
    expect(modelOptionsQueryKey(' compass ', 'session-1')).toEqual(['model-options', 'compass', 'session-1'])
  })
})

describe('manualPickRemoved', () => {
  const providers = [
    { name: 'OpenRouter', slug: 'openrouter', models: ['owl-alpha', 'gpt-5.5'] },
    { name: 'Nous', slug: 'nous', models: [] } // present but unconfigured / re-auth
  ]

  it('flags a pick whose model was dropped from a populated provider', () => {
    expect(manualPickRemoved(providers, 'openrouter', 'nemotron-removed')).toBe(true)
  })

  it('keeps a pick that is still in the catalog', () => {
    expect(manualPickRemoved(providers, 'openrouter', 'gpt-5.5')).toBe(false)
  })

  it('matches the provider by name as well as slug', () => {
    expect(manualPickRemoved(providers, 'OpenRouter', 'gpt-5.5')).toBe(false)
    expect(manualPickRemoved(providers, 'OpenRouter', 'gone')).toBe(true)
  })

  it('never clobbers when the provider is absent (ambiguous / deauth)', () => {
    expect(manualPickRemoved(providers, 'anthropic', 'claude-sonnet-4.6')).toBe(false)
  })

  it('never clobbers when the provider has an empty model list (re-auth)', () => {
    expect(manualPickRemoved(providers, 'nous', 'hermes-4')).toBe(false)
  })

  it('never clobbers on a not-yet-loaded or empty catalog', () => {
    expect(manualPickRemoved(undefined, 'openrouter', 'gpt-5.5')).toBe(false)
    expect(manualPickRemoved([], 'openrouter', 'gpt-5.5')).toBe(false)
  })

  it('never clobbers when there is no pick', () => {
    expect(manualPickRemoved(providers, '', '')).toBe(false)
  })
})
