import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesModule from '@/hermes'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import { $settingsOwner } from '@/store/settings-scope'

const api = vi.hoisted(() => ({
  getEnvVars: vi.fn(),
  getHermesConfigRecord: vi.fn(),
  getHermesConfigSchema: vi.fn()
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesModule>()),
  ...api
}))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: vi.fn() })
}))

import { useSettingsSearchCatalog } from './use-settings-search'

function connection(baseUrl: string, token: string) {
  return {
    authMode: 'token',
    baseUrl,
    connectionId: 'gateway',
    headers: { 'Cf-Access-Client-Id': token },
    mode: 'remote',
    token
  } as never
}

beforeEach(() => {
  $activeGatewayProfile.set('alpha')
  $connection.set(connection('https://one.example', 'token-one'))
  api.getEnvVars.mockResolvedValue({})
  api.getHermesConfigRecord.mockResolvedValue({})
  api.getHermesConfigSchema.mockResolvedValue({ category_order: [], fields: {} })
})

afterEach(() => {
  cleanup()
  $connection.set(null)
  vi.clearAllMocks()
})

describe('useSettingsSearchCatalog owner isolation', () => {
  it('does not fall back to ambient config when no Settings owner is available', () => {
    $connection.set(null)
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    )

    renderHook(() => useSettingsSearchCatalog(true), { wrapper })

    expect(api.getHermesConfigRecord).not.toHaveBeenCalled()
    expect(api.getHermesConfigSchema).not.toHaveBeenCalled()
    expect(api.getEnvVars).not.toHaveBeenCalled()
  })

  it('scopes config, schema and env requests and cache rows to the selected Settings owner', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    )

    const firstOwner = $settingsOwner.get()

    expect(firstOwner).toBeTruthy()
    renderHook(() => useSettingsSearchCatalog(true), { wrapper })

    await waitFor(() => {
      expect(api.getHermesConfigRecord).toHaveBeenCalledWith(firstOwner)
      expect(api.getHermesConfigSchema).toHaveBeenCalledWith(firstOwner)
      expect(api.getEnvVars).toHaveBeenCalledWith(firstOwner)
    })

    $connection.set(connection('https://two.example', 'token-two'))
    const secondOwner = $settingsOwner.get()

    expect(secondOwner).toBeTruthy()
    expect(secondOwner).not.toBe(firstOwner)
    await waitFor(() => {
      expect(api.getHermesConfigRecord).toHaveBeenCalledWith(secondOwner)
      expect(api.getHermesConfigSchema).toHaveBeenCalledWith(secondOwner)
      expect(api.getEnvVars).toHaveBeenCalledWith(secondOwner)
    })

    const keys = client.getQueryCache().getAll().map(query => query.queryKey)
    expect(keys.filter(key => key[0] === 'hermes-config-schema')).toHaveLength(2)
    expect(keys.filter(key => key[0] === 'desktop-settings-search-env-vars')).toHaveLength(2)
    expect(JSON.stringify(keys)).not.toContain('token-one')
    expect(JSON.stringify(keys)).not.toContain('token-two')
  })
})
