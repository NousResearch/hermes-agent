import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MoaConfigResponse } from '@/types/hermes'

const { getAuxiliaryModels, getGlobalModelInfo, getGlobalModelOptions, getMoaModels, setModelAssignment } = vi.hoisted(() => ({
  getAuxiliaryModels: vi.fn(),
  getGlobalModelInfo: vi.fn(),
  getGlobalModelOptions: vi.fn(),
  getMoaModels: vi.fn(),
  setModelAssignment: vi.fn()
}))

let profileSwitchHandler: (() => void) | null = null

vi.mock('@/hermes', () => ({
  createMoaModelsSaveRequest: (body: MoaConfigResponse) => () => Promise.resolve(body),
  getAuxiliaryModels: () => getAuxiliaryModels(),
  getGlobalModelInfo: () => getGlobalModelInfo(),
  getGlobalModelOptions: () => getGlobalModelOptions(),
  getMoaModels: () => getMoaModels(),
  getRecommendedDefaultModel: vi.fn(),
  saveHermesConfig: vi.fn(),
  saveMoaModels: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setEnvVar: vi.fn(),
  setModelAssignment: (body: unknown) => setModelAssignment(body)
}))
vi.mock('@/store/onboarding', () => ({
  startManualLocalEndpoint: vi.fn(),
  startManualOnboarding: vi.fn(),
  startManualProviderOAuth: vi.fn()
}))
vi.mock('../hooks/use-config-record', () => ({
  invalidateHermesConfig: vi.fn(),
  setHermesConfigCache: vi.fn(),
  useHermesConfigRecord: () => ({ data: {} })
}))
vi.mock('../hooks/use-on-profile-switch', () => ({
  useOnProfileSwitch: (handler: () => void) => {
    profileSwitchHandler = handler
  }
}))

import { ModelSettings } from './model-settings'

const moaConfig = (name: string): MoaConfigResponse => ({
  active_preset: '',
  aggregator: { model: 'anthropic/claude-opus-4.8', provider: 'openrouter' },
  aggregator_temperature: null,
  default_preset: name,
  enabled: true,
  max_tokens: 4096,
  presets: {
    [name]: {
      aggregator: { model: 'anthropic/claude-opus-4.8', provider: 'openrouter' },
      aggregator_temperature: null,
      enabled: true,
      fanout: 'user_turn',
      max_tokens: 4096,
      reference_max_tokens: 600,
      reference_models: [{ continuity_id: `${name}-advisor`, model: 'hermes-4', provider: 'nous' }],
      reference_temperature: null
    }
  },
  reference_models: [{ continuity_id: `${name}-advisor`, model: 'hermes-4', provider: 'nous' }],
  reference_temperature: null
})

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(res => {
    resolve = res
  })

  return { promise, resolve }
}

function renderModelSettings(props: { studioOnly?: boolean } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    // The aux-task deep-link highlight reads useSearchParams, so the page
    // needs a router context in tests (the app provides HashRouter at root).
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <ModelSettings {...props} />
      </QueryClientProvider>
    </MemoryRouter>
  )
}

beforeEach(() => {
  profileSwitchHandler = null
  getGlobalModelInfo.mockResolvedValue({ model: 'hermes-4', provider: 'nous' })
  getGlobalModelOptions.mockResolvedValue({
    providers: [
      { authenticated: true, models: ['hermes-4'], name: 'Nous', slug: 'nous' },
      {
        authenticated: true,
        models: ['anthropic/claude-opus-4.8'],
        name: 'OpenRouter',
        slug: 'openrouter'
      }
    ]
  })
  getAuxiliaryModels.mockResolvedValue({
    main: { model: 'hermes-4', provider: 'nous' },
    tasks: [{ base_url: '', model: '', provider: 'auto', task: 'vision' }]
  })
  getMoaModels.mockResolvedValue(moaConfig('profile-a'))
  setModelAssignment.mockResolvedValue({ model: 'hermes-4', provider: 'nous', stale_aux: [] })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  profileSwitchHandler = null
})

describe('ModelSettings shared MoA Studio mount', () => {
  it('uses the shared compact Loader while MoA configuration is loading', async () => {
    const pending = deferred<MoaConfigResponse>()
    getMoaModels.mockReturnValueOnce(pending.promise)
    renderModelSettings({ studioOnly: true })

    await waitFor(() => expect(getMoaModels).toHaveBeenCalledOnce())
    const status = screen.getByText('Loading model configuration...').closest('[role="status"]')
    const loader = status?.querySelector('[aria-label="Loading"]')

    expect(loader).not.toBeNull()
    expect(loader?.className ?? '').toContain('size-5')
  })

  it('renders only MoA Studio in studioOnly mode', async () => {
    renderModelSettings({ studioOnly: true })

    expect(await screen.findByText('Reference 1')).toBeTruthy()
    expect(screen.getByRole('combobox', { name: 'Preset' }).textContent).toContain('profile-a')
    expect(screen.queryByText('Vision')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Apply' })).toBeNull()
  })

  it('surfaces a MoA load failure without breaking ordinary model settings', async () => {
    getMoaModels.mockRejectedValueOnce(new Error('moa endpoint unavailable'))
    renderModelSettings()

    expect(await screen.findByText('Vision')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Apply' })).toBeTruthy()
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('moa endpoint unavailable'))
  })

  it('drops an older profile MoA response after the active profile changes', async () => {
    const profileA = deferred<MoaConfigResponse>()
    getMoaModels.mockReturnValueOnce(profileA.promise).mockResolvedValueOnce(moaConfig('profile-b'))
    renderModelSettings({ studioOnly: true })

    await waitFor(() => expect(getMoaModels).toHaveBeenCalledTimes(1))
    await act(async () => {
      profileSwitchHandler?.()
    })

    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Preset' }).textContent).toContain('profile-b')
    )

    await act(async () => {
      profileA.resolve(moaConfig('profile-a'))
      await profileA.promise
    })

    expect(screen.getByRole('combobox', { name: 'Preset' }).textContent).toContain('profile-b')
    expect(screen.queryByText('profile-a')).toBeNull()
  })

  it('keeps ordinary model refreshes separate from MoA while profile switches reload both', async () => {
    renderModelSettings()

    expect(await screen.findByText('Reference 1')).toBeTruthy()
    expect(getMoaModels).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))

    await waitFor(() => expect(setModelAssignment).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(getGlobalModelInfo).toHaveBeenCalledTimes(2))
    expect(getMoaModels).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('combobox', { name: 'Preset' }).textContent).toContain('profile-a')

    await act(async () => {
      profileSwitchHandler?.()
    })

    await waitFor(() => expect(getMoaModels).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(getGlobalModelInfo).toHaveBeenCalledTimes(3))
  })
})
