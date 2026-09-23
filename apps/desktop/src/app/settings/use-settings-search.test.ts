// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import { createElement, type PropsWithChildren } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'

import { filterSettingsSearchEntries } from './settings-search'
import { useSettingsSearchCatalog } from './use-settings-search'

vi.mock('@/i18n', () => ({ useI18n: () => ({ t: en }) }))

vi.mock('@/hermes', () => ({
  getEnvVars: vi.fn(() => Promise.resolve({})),
  getHermesConfigSchema: vi.fn(() => Promise.resolve({ fields: {} }))
}))

vi.mock('../hooks/use-config-record', () => ({
  useHermesConfigRecord: () => ({ data: undefined, isError: false, isFetching: false })
}))

vi.mock('../hooks/use-on-profile-switch', () => ({ useOnProfileSwitch: vi.fn() }))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: vi.fn() })
}))

function wrapper({ children }: PropsWithChildren) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return createElement(QueryClientProvider, { client }, children)
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('settings search catalog copy', () => {
  it('ranks the titlebar settings entry under the cogwheel keyword', async () => {
    const { result } = renderHook(() => useSettingsSearchCatalog(true), { wrapper })

    const entries = await waitFor(() => {
      if (result.current.appearanceEntries.length === 0) {
        throw new Error('catalog not ready')
      }

      return result.current.appearanceEntries
    })

    const appActions = entries.find(entry => entry.id === 'setting:appearance.app-actions')

    expect(appActions).toBeDefined()

    expect(filterSettingsSearchEntries(entries, 'cogwheel').map(entry => entry.id)).toContain(
      'setting:appearance.app-actions'
    )
  })

  it('keeps titlebar wording matching for plain settings searches', async () => {
    const { result } = renderHook(() => useSettingsSearchCatalog(true), { wrapper })

    const entries = await waitFor(() => {
      if (result.current.appearanceEntries.length === 0) {
        throw new Error('catalog not ready')
      }

      return result.current.appearanceEntries
    })

    expect(filterSettingsSearchEntries(entries, 'titlebar settings').map(entry => entry.id)).toContain(
      'setting:appearance.app-actions'
    )
  })
})
