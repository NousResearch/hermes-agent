// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { getGlobalModelOptions } from '@/hermes'

import { useProviderModelCatalog } from './use-provider-catalog'

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: vi.fn()
}))

const Wrapper = ({ children }: { children: React.ReactNode }) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

describe('useProviderModelCatalog', () => {
  it('requests unconfigured providers and explicit models, then drops empty providers', async () => {
    vi.mocked(getGlobalModelOptions).mockResolvedValue({
      providers: [
        { slug: 'openai', name: 'OpenAI', models: ['gpt-4', 'gpt-3.5'] },
        { slug: 'ghost', name: 'Ghost', models: [] }
      ]
    })

    const { result } = renderHook(() => useProviderModelCatalog(), { wrapper: Wrapper })

    expect(getGlobalModelOptions).toHaveBeenCalledWith({
      includeUnconfigured: true,
      explicitOnly: false
    })
    expect(result.current.isPending).toBe(true)

    await waitFor(() => expect(result.current.isPending).toBe(false))

    expect(result.current.providers).toHaveLength(1)
    expect(result.current.providers[0].slug).toBe('openai')
  })

  it('exposes an error flag and empty list when the request rejects', async () => {
    vi.mocked(getGlobalModelOptions).mockRejectedValue(new Error('boom'))

    const { result } = renderHook(() => useProviderModelCatalog(), { wrapper: Wrapper })

    await waitFor(() => expect(result.current.isError).toBe(true))
    expect(result.current.providers).toEqual([])
  })

  it('keeps custom providers even when they have no models yet', async () => {
    vi.mocked(getGlobalModelOptions).mockResolvedValue({
      providers: [
        { slug: 'openai', name: 'OpenAI', models: ['gpt-4'] },
        { slug: 'custom:lab', name: 'Lab', models: [], is_user_defined: true }
      ]
    })

    const { result } = renderHook(() => useProviderModelCatalog(), { wrapper: Wrapper })

    await waitFor(() => expect(result.current.isPending).toBe(false))

    const slugs = result.current.providers.map(p => p.slug)
    expect(slugs).toContain('openai')
    expect(slugs).toContain('custom:lab')
  })

  it('keeps an unconfigured api_key provider with a key_env so it can be activated', async () => {
    vi.mocked(getGlobalModelOptions).mockResolvedValue({
      providers: [
        { slug: 'openai', name: 'OpenAI', models: ['gpt-4'] },
        {
          slug: 'deepseek',
          name: 'DeepSeek',
          models: [],
          auth_type: 'api_key',
          key_env: 'DEEPSEEK_API_KEY',
          authenticated: false
        }
      ]
    })

    const { result } = renderHook(() => useProviderModelCatalog(), { wrapper: Wrapper })

    await waitFor(() => expect(result.current.isPending).toBe(false))

    const slugs = result.current.providers.map(p => p.slug)
    expect(slugs).toContain('deepseek')
  })

  it('drops an unconfigured oauth provider with no models (not inline-configurable)', async () => {
    vi.mocked(getGlobalModelOptions).mockResolvedValue({
      providers: [
        { slug: 'openai', name: 'OpenAI', models: ['gpt-4'] },
        { slug: 'google', name: 'Google', models: [], auth_type: 'oauth_google', key_env: 'GOOGLE_API_KEY', authenticated: false }
      ]
    })

    const { result } = renderHook(() => useProviderModelCatalog(), { wrapper: Wrapper })

    await waitFor(() => expect(result.current.isPending).toBe(false))

    const slugs = result.current.providers.map(p => p.slug)
    expect(slugs).toContain('openai')
    expect(slugs).not.toContain('google')
  })

  it('drops a model-less provider that is neither user-defined nor configurable', async () => {
    vi.mocked(getGlobalModelOptions).mockResolvedValue({
      providers: [
        { slug: 'openai', name: 'OpenAI', models: ['gpt-4'] },
        { slug: 'bare', name: 'Bare', models: [] }
      ]
    })

    const { result } = renderHook(() => useProviderModelCatalog(), { wrapper: Wrapper })

    await waitFor(() => expect(result.current.isPending).toBe(false))

    const slugs = result.current.providers.map(p => p.slug)
    expect(slugs).toContain('openai')
    expect(slugs).not.toContain('bare')
  })
})
