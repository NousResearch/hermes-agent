// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  deleteCustomEndpoint,
  getCustomEndpoints,
  getGlobalModelOptions,
  saveCustomEndpoint,
  setCustomEndpointEnabled
} from '@/hermes'

import { $visibleModels, emptyProviderSentinelKey } from '@/store/model-visibility'

import { useProviderConfig } from './use-provider-config'

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: vi.fn().mockResolvedValue({ providers: [] }),
  getCustomEndpoints: vi.fn(),
  saveCustomEndpoint: vi.fn().mockResolvedValue({ ok: true, endpoints: [] }),
  deleteCustomEndpoint: vi.fn().mockResolvedValue({ ok: true, endpoints: [] }),
  setCustomEndpointEnabled: vi.fn().mockResolvedValue({ ok: true, endpoints: [] }),
  discoverProviderModels: vi.fn().mockResolvedValue({ models: [] }),
  testCustomProviderConnection: vi.fn().mockResolvedValue({ ok: true }),
  setEnvVar: vi.fn().mockResolvedValue({ ok: true })
}))

const Wrapper = ({ children }: { children: React.ReactNode }) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

const endpointsResponse = (endpoints: unknown[] = []) => ({
  endpoints,
  current: { provider: '', model: '', base_url: '' }
})

const labEndpoint = {
  id: 'lab',
  name: 'Lab',
  base_url: 'https://lab/v1',
  model: 'a',
  models: ['a'],
  discover_models: true,
  has_api_key: false,
  enabled: true
}

describe('useProviderConfig', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('loads custom providers from the REST endpoints API', async () => {
    vi.mocked(getCustomEndpoints).mockResolvedValue(endpointsResponse([labEndpoint]) as any)

    const { result } = renderHook(() => useProviderConfig(), { wrapper: Wrapper })

    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(getCustomEndpoints).toHaveBeenCalled()
    expect(result.current.customProviders).toHaveLength(1)
    expect(result.current.customProviders[0].name).toBe('lab')
    expect(result.current.customProviders[0].base_url).toBe('https://lab/v1')
  })

  it('saveCustomProvider calls saveCustomEndpoint with the api_key (never a config write)', async () => {
    vi.mocked(getCustomEndpoints).mockResolvedValue(endpointsResponse([labEndpoint]) as any)

    const { result } = renderHook(() => useProviderConfig(), { wrapper: Wrapper })
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    await result.current.saveCustomProvider({
      name: 'New One',
      base_url: 'https://new/v1',
      api_key: 'sk-new',
      api_mode: 'chat_completions',
      models: [{ id: 'x' }]
    })

    expect(saveCustomEndpoint).toHaveBeenCalledTimes(1)
    const update = vi.mocked(saveCustomEndpoint).mock.calls[0][0]
    expect(update.id).toBe('new-one')
    expect(update.base_url).toBe('https://new/v1')
    expect(update.api_key).toBe('sk-new')
    expect(update.api_mode).toBe('chat_completions')
    expect(update.models).toEqual(['x'])
    // Re-probe + invalidate after save.
    expect(getGlobalModelOptions).toHaveBeenCalledWith({
      includeUnconfigured: true,
      explicitOnly: false,
      refresh: true
    })
  })

  it('saveCustomProvider omits api_key when the form submits an empty one', async () => {
    vi.mocked(getCustomEndpoints).mockResolvedValue(endpointsResponse([labEndpoint]) as any)

    const { result } = renderHook(() => useProviderConfig(), { wrapper: Wrapper })
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    await result.current.saveCustomProvider({
      name: 'Lab',
      base_url: 'https://lab/v2',
      api_key: '',
      api_mode: 'chat_completions',
      models: [{ id: 'a' }]
    })

    const update = vi.mocked(saveCustomEndpoint).mock.calls[0][0]
    expect(update.api_key).toBeUndefined()
  })

  it('setEnabled toggles a custom provider via setCustomEndpointEnabled', async () => {
    vi.mocked(getCustomEndpoints).mockResolvedValue(endpointsResponse([labEndpoint]) as any)

    const { result } = renderHook(() => useProviderConfig(), { wrapper: Wrapper })
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    await result.current.setEnabled('custom:lab', false)

    expect(setCustomEndpointEnabled).toHaveBeenCalledWith('lab', false)
  })

  it('deleteCustomProvider calls deleteCustomEndpoint by id', async () => {
    vi.mocked(getCustomEndpoints).mockResolvedValue(endpointsResponse([labEndpoint]) as any)

    const { result } = renderHook(() => useProviderConfig(), { wrapper: Wrapper })
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    await result.current.deleteCustomProvider('custom:lab')

    expect(deleteCustomEndpoint).toHaveBeenCalledWith('lab')
  })

  it('a new provider starts with all models hidden (hide-all sentinel)', async () => {
    vi.mocked(getCustomEndpoints).mockResolvedValue(endpointsResponse([]) as any)
    $visibleModels.set(null)

    const { result } = renderHook(() => useProviderConfig(), { wrapper: Wrapper })
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    await result.current.saveCustomProvider({
      name: 'Fresh Provider',
      base_url: 'https://fresh/v1',
      api_mode: 'chat_completions',
      models: [{ id: 'a' }, { id: 'b' }]
    })

    const stored = $visibleModels.get()
    expect(stored?.has(emptyProviderSentinelKey('custom:fresh-provider'))).toBe(true)
  })

  it('editing an existing provider does not write the hide-all sentinel', async () => {
    vi.mocked(getCustomEndpoints).mockResolvedValue(endpointsResponse([labEndpoint]) as any)
    $visibleModels.set(null)

    const { result } = renderHook(() => useProviderConfig(), { wrapper: Wrapper })
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    await result.current.saveCustomProvider({
      name: 'Lab',
      base_url: 'https://lab/v2',
      api_mode: 'chat_completions',
      models: [{ id: 'a' }]
    })

    expect($visibleModels.get()?.has(emptyProviderSentinelKey('custom:lab'))).toBeFalsy()
  })

  describe('refreshCatalog', () => {
    it('calls getGlobalModelOptions with refresh: true', async () => {
      vi.mocked(getCustomEndpoints).mockResolvedValue(endpointsResponse([]) as any)

      const { result } = renderHook(() => useProviderConfig(), { wrapper: Wrapper })
      await waitFor(() => expect(result.current.isLoading).toBe(false))

      vi.mocked(getGlobalModelOptions).mockClear()
      await result.current.refreshCatalog()

      expect(getGlobalModelOptions).toHaveBeenCalledWith({
        includeUnconfigured: true,
        explicitOnly: false,
        refresh: true
      })
    })

    it('propagates errors from the RPC call', async () => {
      vi.mocked(getCustomEndpoints).mockResolvedValue(endpointsResponse([]) as any)

      const { result } = renderHook(() => useProviderConfig(), { wrapper: Wrapper })
      await waitFor(() => expect(result.current.isLoading).toBe(false))

      vi.mocked(getGlobalModelOptions).mockRejectedValueOnce(new Error('backend down'))

      await expect(result.current.refreshCatalog()).rejects.toThrow('backend down')
    })
  })
})
