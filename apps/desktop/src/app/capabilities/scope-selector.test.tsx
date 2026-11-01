import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { getProfiles } from '@/hermes'
import type { ProfileInfo } from '@/types/hermes'

vi.mock('@/store/gateway', () => ({ $gateway: atom(null), activeGatewayConnectionId: () => null }))
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getProfiles: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ reloadOpenStarmapGraph: vi.fn(), resetStarmapGraph: vi.fn() }))

const { $activeGatewayProfile } = await import('@/store/profile')
const { useCapabilityScope } = await import('./scope-selector')

const profile = (name: string, extra: Partial<ProfileInfo> = {}): ProfileInfo => ({
  name,
  is_default: name === 'default',
  path: '',
  has_env: false,
  model: null,
  provider: null,
  skill_count: 0,
  ...extra
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

it('shows rich single-connection agent names while capability scopes keep canonical profile keys (#146434)', async () => {
  $activeGatewayProfile.set('default')

  const profiles = [
    profile('default'),
    profile('research-agent', { bot_title: '  中文研究助手  ', display_name: 'Research' }),
    profile('writing-agent', { display_name: '  Writer  ' }),
    profile('legacy-agent', { bot_title: ' \t ', display_name: '  ' })
  ]

  vi.mocked(getProfiles).mockResolvedValue({ profiles })

  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )

  try {
    const { result } = renderHook(() => useCapabilityScope({}), { wrapper })
    await waitFor(() => expect(result.current.options.length).toBe(profiles.length))

    expect(result.current.options).toEqual([
      { key: 'default', label: 'Hermes (default)', value: 'default' },
      { key: 'research-agent', label: '中文研究助手', value: 'research-agent' },
      { key: 'writing-agent', label: 'Writer', value: 'writing-agent' },
      { key: 'legacy-agent', label: 'legacy-agent', value: 'legacy-agent' }
    ])
    act(() => result.current.onChange('research-agent'))
    expect(result.current.value).toBe('research-agent')
    expect(result.current.profile).toBe('research-agent')
    expect(result.current.key).toBe('research-agent')
    expect(result.current.label).toBe('中文研究助手')
    expect(profiles[1].name).toBe('research-agent')
  } finally {
    client.clear()
  }
})
