import { QueryClient } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getGlobalModelOptions } from '@/hermes'

import {
  catalogProviderMatches,
  firstSelectableCatalogModel,
  manualPickRemoved,
  modelOptionsQueryKey,
  reconcileSelectionAfterCatalogRefresh,
  requestModelOptions,
  selectionInCatalog
} from './model-options'

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

  it('passes the catalog owner profile through the shared gateway RPC', async () => {
    const gateway = {
      request: vi.fn(() => Promise.resolve(globalOptions))
    }

    await requestModelOptions({ gateway: gateway as never, profile: 'fred-work' })

    expect(gateway.request).toHaveBeenCalledWith('model.options', {
      explicit_only: true,
      profile: 'fred-work'
    })
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

    const request = vi.fn(() => Promise.resolve(routedPayload)) as unknown as <T>(
      method: string,
      params?: Record<string, unknown>
    ) => Promise<T>

    await expect(requestModelOptions({ gateway: gateway as never, request, sessionId: 'tile-1' })).resolves.toBe(
      routedPayload
    )

    expect(request).toHaveBeenCalledWith('model.options', { explicit_only: true, session_id: 'tile-1' })
    expect(gateway.request).not.toHaveBeenCalled()
  })

  it('does not recover an owner-routed failure through the ambient REST connection', async () => {
    const ownerError = new Error('owner gateway unavailable')
    const request = vi.fn(() => Promise.reject(ownerError))

    await expect(requestModelOptions({ profile: 'berry', request, sessionId: 'tile-1' })).rejects.toBe(ownerError)
    expect(getGlobalModelOptions).not.toHaveBeenCalled()
  })

  it('keeps an empty owner-routed catalog instead of replacing it from ambient REST', async () => {
    const ownerPayload = { model: 'berry-local', provider: 'hermes-local', providers: [] }

    const request = vi.fn(() => Promise.resolve(ownerPayload)) as unknown as <T>(
      method: string,
      params?: Record<string, unknown>
    ) => Promise<T>

    await expect(requestModelOptions({ profile: 'berry', request, sessionId: 'tile-1' })).resolves.toBe(ownerPayload)
    expect(getGlobalModelOptions).not.toHaveBeenCalled()
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

  it('isolates identical profile and session names across registry connections', () => {
    const sourceAKey = modelOptionsQueryKey('default', 'session-1', 'source-a')
    const sourceBKey = modelOptionsQueryKey('default', 'session-1', 'source-b')
    const queryClient = new QueryClient()

    expect(sourceAKey).toEqual(['model-options', 'default', 'session-1', 'owner', 'source-a'])
    queryClient.setQueryData(sourceAKey, { providers: [{ models: ['a/model'], slug: 'a' }] })
    queryClient.setQueryData(sourceBKey, { providers: [{ models: ['b/model'], slug: 'b' }] })

    expect(queryClient.getQueryData(sourceAKey)).toMatchObject({ providers: [{ models: ['a/model'] }] })
    expect(queryClient.getQueryData(sourceBKey)).toMatchObject({ providers: [{ models: ['b/model'] }] })
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

describe('catalogProviderMatches', () => {
  const cloudflare = {
    aliases: ['custom:cloudflare', 'cloudflare'],
    models: ['@cf/meta/llama-3.3-70b-instruct-fp8-fast'],
    name: 'Cloudflare',
    slug: 'cloudflare'
  }

  it('matches slug, display name, and custom-provider aliases', () => {
    expect(catalogProviderMatches(cloudflare, 'cloudflare')).toBe(true)
    expect(catalogProviderMatches(cloudflare, 'Cloudflare')).toBe(true)
    expect(catalogProviderMatches(cloudflare, 'custom:cloudflare')).toBe(true)
    expect(catalogProviderMatches(cloudflare, 'openrouter')).toBe(false)
  })
})

describe('reconcileSelectionAfterCatalogRefresh', () => {
  const zhipu = { name: '智谱2', slug: 'zhipu', models: ['glm-4.5-air', 'glm-5-turbo'] }

  const bytea = {
    name: '字节A',
    slug: 'byteplus',
    models: ['deepseek-v4-flash', 'doubao-seed-2.0-pro']
  }

  const moa = { name: 'Mixture of Agents', slug: 'moa', models: ['default'] }

  const openrouter = {
    models: ['glm-4.5-air', 'gpt-5.5'],
    name: 'OpenRouter',
    slug: 'openrouter'
  }

  it('switches to the first new-group model when the current pick is gone', () => {
    expect(selectionInCatalog([bytea], 'glm-4.5-air', 'zhipu')).toBe(false)
    expect(firstSelectableCatalogModel([moa, bytea])).toEqual({
      model: 'deepseek-v4-flash',
      provider: 'byteplus'
    })
    expect(reconcileSelectionAfterCatalogRefresh('glm-4.5-air', [moa, bytea], 'zhipu')).toEqual({
      model: 'deepseek-v4-flash',
      provider: 'byteplus'
    })
  })

  it('keeps the current pick when it is still in the refreshed catalog', () => {
    expect(reconcileSelectionAfterCatalogRefresh('glm-4.5-air', [zhipu, moa], 'zhipu')).toBeNull()
  })

  it('keeps the current provider when the same model id exists on another provider', () => {
    expect(selectionInCatalog([openrouter, zhipu], 'glm-4.5-air', 'zhipu')).toBe(true)
    expect(selectionInCatalog([openrouter, zhipu], 'glm-4.5-air', 'openrouter')).toBe(true)
    expect(reconcileSelectionAfterCatalogRefresh('glm-4.5-air', [openrouter, zhipu], 'zhipu')).toBeNull()
  })

  it('keeps a custom provider pair when OpenRouter lists the same model id', () => {
    const model = '@cf/meta/llama-3.3-70b-instruct-fp8-fast'

    const cloudflare = {
      aliases: ['custom:cloudflare', 'cloudflare'],
      models: [model],
      name: 'Cloudflare',
      slug: 'cloudflare'
    }

    const openrouterCf = { models: [model, 'gpt-5.5'], name: 'OpenRouter', slug: 'openrouter' }

    expect(selectionInCatalog([openrouterCf, cloudflare], model, 'custom:cloudflare')).toBe(true)
    expect(reconcileSelectionAfterCatalogRefresh(model, [openrouterCf, cloudflare], 'custom:cloudflare')).toBeNull()
  })

  it('does not jump to OpenRouter when the current provider is missing but still lists the same model id', () => {
    expect(reconcileSelectionAfterCatalogRefresh('glm-4.5-air', [openrouter, moa], 'zhipu')).toBeNull()
  })

  it('keeps the pick when the current provider is present but unconfigured', () => {
    const emptyZhipu = { models: [], name: '智谱2', slug: 'zhipu' }

    expect(reconcileSelectionAfterCatalogRefresh('glm-4.5-air', [emptyZhipu, openrouter], 'zhipu')).toBeNull()
  })

  it('falls back when the current provider is populated and dropped the model', () => {
    const zhipuWithoutAir = { models: ['glm-5-turbo'], name: '智谱2', slug: 'zhipu' }

    expect(reconcileSelectionAfterCatalogRefresh('glm-4.5-air', [zhipuWithoutAir, openrouter], 'zhipu')).toEqual({
      model: 'glm-5-turbo',
      provider: 'zhipu'
    })
  })

  it('does not wipe the pick when the refreshed catalog has no selectable models', () => {
    expect(reconcileSelectionAfterCatalogRefresh('glm-4.5-air', [moa], 'zhipu')).toBeNull()
    expect(reconcileSelectionAfterCatalogRefresh('glm-4.5-air', [], 'zhipu')).toBeNull()
    expect(reconcileSelectionAfterCatalogRefresh('glm-4.5-air', undefined, 'zhipu')).toBeNull()
  })
})
