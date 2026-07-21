import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const getHermesConfigRecord = vi.fn()
const getHermesConfigSchema = vi.fn()

vi.mock('@/hermes', () => ({
  getElevenLabsVoices: vi.fn().mockResolvedValue({ available: false, voices: [] }),
  getHermesConfigRecord: () => getHermesConfigRecord(),
  getHermesConfigSchema: () => getHermesConfigSchema(),
  saveHermesConfig: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { saving: 'Saving' },
      settings: {
        config: {
          autosaveFailed: 'Could not save settings',
          emptyDesc: 'No settings',
          emptyTitle: 'No settings',
          failedLoad: 'Could not load settings',
          invalidJson: 'Invalid JSON',
          loading: 'Loading settings'
        }
      },
      skills: { refresh: 'Refresh' }
    }
  })
}))

vi.mock('@/store/keep-awake', async () => {
  const { atom } = await import('nanostores')

  return { $keepAwake: atom(false), setKeepAwake: vi.fn() }
})

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

vi.mock('@/store/profile', async () => {
  const { atom } = await import('nanostores')

  return { $activeGatewayProfile: atom('default') }
})

vi.mock('react-router-dom', () => ({
  useSearchParams: () => [new URLSearchParams(), vi.fn()]
}))

vi.mock('../overlays/panel', () => ({
  PanelEmpty: ({ title }: { title: string }) => <div>{title}</div>
}))

vi.mock('./config-field', () => ({
  ConfigField: ({ value }: { value: unknown }) => <div data-testid="config-value">{String(value)}</div>
}))

vi.mock('./helpers', () => ({
  enumOptionsFor: vi.fn(),
  getNested: (config: Record<string, unknown>, key: string) => config[key],
  isExternalMemoryProvider: () => false,
  sectionFieldEntries: (_schema: unknown, config: Record<string, unknown>) =>
    new Map([['chat', config.timezone === undefined ? [] : [['timezone', { type: 'string' }]]]]),
  setNested: (config: Record<string, unknown>, key: string, value: unknown) => ({ ...config, [key]: value })
}))

vi.mock('./memory/connect', () => ({ MemoryConnect: () => null }))
vi.mock('./memory/provider-config-panel', () => ({ ProviderConfigPanel: () => null }))
vi.mock('./model-settings', () => ({ ModelSettings: () => null, ModelSettingsSkeleton: () => null }))
vi.mock('./primitives', () => ({
  EmptyState: () => null,
  LoadingState: ({ label }: { label: string }) => <div>{label}</div>,
  SettingsContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  ToggleRow: () => null
}))

beforeEach(async () => {
  vi.clearAllMocks()
  const { $activeGatewayProfile } = await import('@/store/profile')
  $activeGatewayProfile.set('default')
  getHermesConfigSchema.mockResolvedValue({ fields: { timezone: { type: 'string' } } })
})

afterEach(() => cleanup())

describe('ConfigSettings profile switching', () => {
  it('reseeds after an identical config refetch instead of loading forever', async () => {
    let resolveRefetch: ((value: { timezone: string }) => void) | undefined

    const refetchResult = new Promise<{ timezone: string }>(resolve => {
      resolveRefetch = resolve
    })

    getHermesConfigRecord.mockResolvedValueOnce({ timezone: 'UTC' }).mockReturnValueOnce(refetchResult)

    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { ConfigSettings } = await import('./config-settings')
    const { $activeGatewayProfile } = await import('@/store/profile')

    render(
      <QueryClientProvider client={queryClient}>
        <ConfigSettings activeSectionId="chat" importInputRef={{ current: null }} />
      </QueryClientProvider>
    )

    expect((await screen.findByTestId('config-value')).textContent).toBe('UTC')

    let refetch: Promise<void> | undefined

    act(() => {
      $activeGatewayProfile.set('work')
      refetch = queryClient.invalidateQueries({ queryKey: ['hermes-config-record'] })
    })

    expect(await screen.findByText('Loading settings')).toBeTruthy()

    await new Promise(resolve => setTimeout(resolve, 5))
    resolveRefetch?.({ timezone: 'UTC' })
    await refetch

    expect((await screen.findByTestId('config-value')).textContent).toBe('UTC')
  })
})
