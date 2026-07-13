import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MoaConfigResponse } from '@/hermes'

// Radix Select calls scrollIntoView on its items when the content opens; jsdom
// doesn't implement it (nor hasPointerCapture / releasePointerCapture), so stub
// them to let the dropdown open in tests.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const getGlobalModelInfo = vi.fn()
const getGlobalModelOptions = vi.fn()
const getAuxiliaryModels = vi.fn()
const getMoaModels = vi.fn()
const setModelAssignment = vi.fn()
const getRecommendedDefaultModel = vi.fn()
const saveMoaModels = vi.fn()
const setEnvVar = vi.fn()
const setApiRequestProfile = vi.fn()
const getHermesConfigRecord = vi.fn()
const saveHermesConfig = vi.fn()
const startManualProviderOAuth = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelInfo: () => getGlobalModelInfo(),
  getGlobalModelOptions: () => getGlobalModelOptions(),
  getAuxiliaryModels: () => getAuxiliaryModels(),
  getMoaModels: () => getMoaModels(),
  setModelAssignment: (body: unknown) => setModelAssignment(body),
  getRecommendedDefaultModel: (slug: string) => getRecommendedDefaultModel(slug),
  saveMoaModels: (body: unknown) => saveMoaModels(body),
  setEnvVar: (key: string, value: string) => setEnvVar(key, value),
  setApiRequestProfile: (profile: string) => setApiRequestProfile(profile),
  getHermesConfigRecord: () => getHermesConfigRecord(),
  saveHermesConfig: (config: unknown) => saveHermesConfig(config)
}))

vi.mock('@/store/onboarding', () => ({
  startManualProviderOAuth: (slug: string) => startManualProviderOAuth(slug)
}))

beforeEach(() => {
  saveMoaModels.mockReset()
  getGlobalModelInfo.mockResolvedValue({ provider: 'nous', model: 'hermes-4' })
  getGlobalModelOptions.mockResolvedValue({
    providers: [
      {
        name: 'Nous',
        slug: 'nous',
        models: ['hermes-4', 'hermes-4-mini'],
        authenticated: true,
        capabilities: { 'hermes-4': { reasoning: true, fast: true } }
      }
    ]
  })
  getAuxiliaryModels.mockResolvedValue({
    main: { provider: 'nous', model: 'hermes-4' },
    tasks: [{ task: 'vision', provider: 'auto', model: '', base_url: '' }]
  })
  getMoaModels.mockResolvedValue(null)
  setModelAssignment.mockResolvedValue({ provider: 'nous', model: 'hermes-4', gateway_tools: [] })
  getRecommendedDefaultModel.mockResolvedValue({ provider: 'nous', model: 'hermes-4', free_tier: null })
  saveMoaModels.mockImplementation(async body => body)
  setEnvVar.mockResolvedValue({ ok: true })
  getHermesConfigRecord.mockResolvedValue({ agent: { reasoning_effort: 'medium', service_tier: 'normal' } })
  saveHermesConfig.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.clearAllMocks()
})

async function renderModelSettings() {
  const { ModelSettings } = await import('./model-settings')
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <ModelSettings />
    </QueryClientProvider>
  )
}

function createMoaConfig(): MoaConfigResponse {
  const reference = { provider: 'nous', model: 'hermes-4' }
  const aggregator = { provider: 'nous', model: 'hermes-4-mini' }

  return {
    default_preset: 'default',
    active_preset: 'default',
    presets: {
      default: {
        aggregator: { ...aggregator },
        aggregator_temperature: 0.2,
        enabled: true,
        max_tokens: 4096,
        reference_models: [{ ...reference }],
        reference_temperature: 0.7
      }
    },
    aggregator: { ...aggregator },
    aggregator_temperature: 0.2,
    enabled: true,
    max_tokens: 4096,
    reference_models: [{ ...reference }],
    reference_temperature: 0.7
  }
}

function configureMoa() {
  getGlobalModelOptions.mockResolvedValue({
    providers: [
      {
        name: 'Nous',
        slug: 'nous',
        models: ['hermes-4', 'hermes-4-mini'],
        authenticated: true,
        capabilities: {}
      },
      {
        name: 'OpenRouter',
        slug: 'openrouter',
        models: ['anthropic/claude-sonnet-4.5'],
        authenticated: true,
        capabilities: {}
      }
    ]
  })
  getMoaModels.mockResolvedValue(createMoaConfig())
}

function getMoaRow(name: string): HTMLElement {
  const row = screen.getByText(name).parentElement

  if (!row) {
    throw new Error(`Could not find controls for ${name}`)
  }

  return row
}

function selectMoaValue(rowName: string, selectIndex: number, optionName: string) {
  const trigger = within(getMoaRow(rowName)).getAllByRole('combobox')[selectIndex]
  fireEvent.click(trigger)
  fireEvent.click(screen.getByRole('option', { name: optionName }))
}

async function flushMoaAutosave() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(601)
  })
}

function createDeferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(res => {
    resolve = res
  })

  return { promise, resolve }
}

describe('ModelSettings', () => {
  it('loads the current main model and lists configured providers only', async () => {
    await renderModelSettings()

    await waitFor(() => expect(getGlobalModelInfo).toHaveBeenCalled())
    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())

    // Open the provider Select — only configured providers should be listed.
    const triggers = await screen.findAllByRole('combobox')
    fireEvent.click(triggers[0])

    // "Nous" shows in both the trigger and the open list.
    expect((await screen.findAllByText('Nous')).length).toBeGreaterThan(0)
    expect(screen.queryByText(/DeepSeek/)).toBeNull()
  })

  it('writes the profile default speed (service_tier) when the fast switch is toggled', async () => {
    await renderModelSettings()
    await waitFor(() => expect(getHermesConfigRecord).toHaveBeenCalled())

    const fastSwitch = await screen.findByRole('switch')
    fireEvent.click(fastSwitch)

    await waitFor(() =>
      expect(saveHermesConfig).toHaveBeenCalledWith(
        expect.objectContaining({ agent: expect.objectContaining({ service_tier: 'fast' }) })
      )
    )
  })

  it('hides the reasoning/speed defaults when the main model reports no capabilities', async () => {
    getGlobalModelOptions.mockResolvedValueOnce({
      providers: [
        {
          name: 'Nous',
          slug: 'nous',
          models: ['hermes-4'],
          authenticated: true,
          capabilities: { 'hermes-4': { reasoning: false, fast: false } }
        }
      ]
    })

    await renderModelSettings()
    await waitFor(() => expect(getHermesConfigRecord).toHaveBeenCalled())

    expect(screen.queryByRole('switch')).toBeNull()
  })

  it('renders the auxiliary task rows', async () => {
    await renderModelSettings()

    expect(await screen.findByText('Vision')).toBeTruthy()
    expect(screen.getAllByText('auto · use main model').length).toBeGreaterThan(0)
  })

  it('assigns an auxiliary task to the main model via setModelAssignment', async () => {
    await renderModelSettings()

    // One "Set to main" button per task slot; the first is Vision.
    const setToMainButtons = await screen.findAllByRole('button', { name: 'Set to main' })
    fireEvent.click(setToMainButtons[0])

    await waitFor(() =>
      expect(setModelAssignment).toHaveBeenCalledWith({
        model: 'hermes-4',
        provider: 'nous',
        scope: 'auxiliary',
        task: 'vision'
      })
    )
  })

  it('warns when a main switch leaves auxiliary tasks pinned to another provider', async () => {
    setModelAssignment.mockResolvedValueOnce({
      provider: 'openrouter',
      model: 'anthropic/claude-opus-4.7',
      gateway_tools: [],
      stale_aux: [{ task: 'compression', provider: 'nous', model: 'hermes-4' }]
    })

    await renderModelSettings()
    await waitFor(() => expect(getGlobalModelInfo).toHaveBeenCalled())

    const applyButton = await screen.findByRole('button', { name: 'Apply' })
    fireEvent.click(applyButton)

    // The switch-time notice names the pinned provider and offers a reset.
    expect(await screen.findByText(/still run on/)).toBeTruthy()
    expect(screen.getByText('nous')).toBeTruthy()
  })

  it('shows a persistent banner when a loaded aux slot mismatches the main provider', async () => {
    getAuxiliaryModels.mockResolvedValueOnce({
      main: { provider: 'nous', model: 'hermes-4' },
      tasks: [{ task: 'curator', provider: 'openrouter', model: 'anthropic/claude-opus-4.7', base_url: '' }]
    })

    await renderModelSettings()

    // Banner present on load, no switch required.
    expect(await screen.findByText(/still run on/)).toBeTruthy()
  })

  describe('MoA autosave', () => {
    it('keeps a new reference draft and cancels its pending save when changing provider clears the model', async () => {
      configureMoa()
      saveMoaModels.mockResolvedValue(createMoaConfig())
      await renderModelSettings()
      await screen.findByText('Mixture of Agents')
      vi.useFakeTimers()

      fireEvent.click(screen.getByRole('button', { name: 'Add reference model' }))
      expect(screen.getByText('Reference 2')).toBeTruthy()

      selectMoaValue('Reference 2', 0, 'OpenRouter')
      expect(getMoaRow('Reference 2').textContent).toContain('openrouter ·')
      await flushMoaAutosave()

      expect(saveMoaModels).not.toHaveBeenCalled()
      expect(screen.getByText('Reference 2')).toBeTruthy()
    })

    it('saves a new reference once after its replacement provider and model are complete', async () => {
      configureMoa()
      await renderModelSettings()
      await screen.findByText('Mixture of Agents')
      vi.useFakeTimers()

      fireEvent.click(screen.getByRole('button', { name: 'Add reference model' }))
      selectMoaValue('Reference 2', 0, 'OpenRouter')
      await flushMoaAutosave()
      expect(saveMoaModels).not.toHaveBeenCalled()

      selectMoaValue('Reference 2', 1, 'anthropic/claude-sonnet-4.5')
      await flushMoaAutosave()

      expect(saveMoaModels).toHaveBeenCalledTimes(1)
      expect(saveMoaModels).toHaveBeenCalledWith(
        expect.objectContaining({
          presets: expect.objectContaining({
            default: expect.objectContaining({
              reference_models: [
                { provider: 'nous', model: 'hermes-4' },
                { provider: 'openrouter', model: 'anthropic/claude-sonnet-4.5' }
              ]
            })
          })
        })
      )
    })

    it('does not save while an aggregator provider change leaves its model empty', async () => {
      configureMoa()
      await renderModelSettings()
      await screen.findByText('Mixture of Agents')
      vi.useFakeTimers()

      selectMoaValue('Aggregator', 0, 'OpenRouter')
      expect(getMoaRow('Aggregator').textContent).toContain('openrouter ·')
      await flushMoaAutosave()

      expect(saveMoaModels).not.toHaveBeenCalled()
      expect(getMoaRow('Aggregator').textContent).toContain('openrouter ·')
    })

    it('does not let an older save response overwrite a newer local draft', async () => {
      configureMoa()
      const pendingSave = createDeferred<MoaConfigResponse>()
      saveMoaModels.mockImplementationOnce(() => pendingSave.promise)
      await renderModelSettings()
      await screen.findByText('Mixture of Agents')
      vi.useFakeTimers()

      selectMoaValue('Reference 1', 1, 'hermes-4-mini')
      await flushMoaAutosave()
      expect(saveMoaModels).toHaveBeenCalledTimes(1)

      selectMoaValue('Reference 1', 0, 'OpenRouter')
      expect(getMoaRow('Reference 1').textContent).toContain('openrouter ·')

      const olderSavedConfig = saveMoaModels.mock.calls[0][0] as MoaConfigResponse
      await act(async () => {
        pendingSave.resolve(olderSavedConfig)
        await pendingSave.promise
      })

      expect(getMoaRow('Reference 1').textContent).toContain('openrouter ·')
    })
  })
})
