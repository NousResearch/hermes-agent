import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, renderHook, waitFor } from '@testing-library/react'
import { createElement, type PropsWithChildren } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const hermesMocks = vi.hoisted(() => ({
  getEnvVars: vi.fn(async () => ({})),
  getHermesConfigRecord: vi.fn(async () => ({ config: {} })),
  getHermesConfigSchema: vi.fn(async () => ({ fields: [] }))
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ...hermesMocks
}))

const gatewayRequestMock = vi.hoisted(() => vi.fn())

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: gatewayRequestMock })
}))

const loadAgentPluginsMock = vi.hoisted(() => vi.fn(async () => undefined))

vi.mock('@/store/agent-plugins', () => ({
  $agentPlugins: { get: () => [], listen: () => () => undefined, subscribe: () => () => undefined },
  isDesktopRelevantPlugin: () => true,
  loadAgentPlugins: loadAgentPluginsMock
}))

import { $gatewayState } from '@/store/session'
import { $settingsScopeOverride } from '@/store/settings-scope'

import { useSettingsSearchCatalog } from './use-settings-search'

// Regression coverage for: the deep ⌘K settings-search catalog (config
// fields, credentials, agent plugins) ignored the shared "Applies to" scope
// override and always searched the active profile — so scoping Settings to
// another profile via the shared selector made that profile's credentials/
// fields/plugins unfindable via search, contradicting the search-page
// removal's own justification ("⌘K already finds credentials").

function wrapper({ children }: PropsWithChildren) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return createElement(QueryClientProvider, { client }, children)
}

beforeEach(() => {
  hermesMocks.getEnvVars.mockClear()
  hermesMocks.getHermesConfigRecord.mockClear()
  hermesMocks.getHermesConfigSchema.mockClear()
  loadAgentPluginsMock.mockClear()
  $settingsScopeOverride.set(null)
  $gatewayState.set('open')
})

afterEach(() => {
  $settingsScopeOverride.set(null)
  vi.clearAllMocks()
})

describe('useSettingsSearchCatalog — settings-scope override', () => {
  it('fetches config/schema/env for the active profile when no scope override is set', async () => {
    renderHook(() => useSettingsSearchCatalog(true), { wrapper })

    await waitFor(() => expect(hermesMocks.getEnvVars).toHaveBeenCalled())

    expect(hermesMocks.getEnvVars).toHaveBeenCalledWith(null)
    expect(hermesMocks.getHermesConfigSchema).toHaveBeenCalledWith(null)
    // useHermesConfigRecord folds null -> undefined before its own fetch call
    // ("null/undefined both mean 'no override'"); that's its established
    // contract, not something this fix changes.
    expect(hermesMocks.getHermesConfigRecord).toHaveBeenCalledWith(undefined)
    await waitFor(() => expect(loadAgentPluginsMock).toHaveBeenCalledWith(gatewayRequestMock, null))
  })

  it('fetches config/schema/env for the scoped profile when Settings is scoped to another one', async () => {
    $settingsScopeOverride.set('work')

    renderHook(() => useSettingsSearchCatalog(true), { wrapper })

    await waitFor(() => expect(hermesMocks.getEnvVars).toHaveBeenCalled())

    expect(hermesMocks.getEnvVars).toHaveBeenCalledWith('work')
    expect(hermesMocks.getHermesConfigSchema).toHaveBeenCalledWith('work')
    expect(hermesMocks.getHermesConfigRecord).toHaveBeenCalledWith('work')
    await waitFor(() => expect(loadAgentPluginsMock).toHaveBeenCalledWith(gatewayRequestMock, 'work'))
  })

  it('refetches config/schema/env for the new profile when the scope override changes after mount', async () => {
    renderHook(() => useSettingsSearchCatalog(true), { wrapper })

    await waitFor(() => expect(hermesMocks.getEnvVars).toHaveBeenCalledWith(null))

    hermesMocks.getEnvVars.mockClear()
    hermesMocks.getHermesConfigSchema.mockClear()
    hermesMocks.getHermesConfigRecord.mockClear()
    loadAgentPluginsMock.mockClear()

    act(() => {
      $settingsScopeOverride.set('work')
    })

    await waitFor(() => expect(hermesMocks.getEnvVars).toHaveBeenCalledWith('work'))
    expect(hermesMocks.getHermesConfigSchema).toHaveBeenCalledWith('work')
    expect(hermesMocks.getHermesConfigRecord).toHaveBeenCalledWith('work')
    await waitFor(() => expect(loadAgentPluginsMock).toHaveBeenCalledWith(gatewayRequestMock, 'work'))
  })
})
