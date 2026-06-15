import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

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
const setModelAssignment = vi.fn()
const getRecommendedDefaultModel = vi.fn()
const setEnvVar = vi.fn()
const startManualProviderOAuth = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelInfo: () => getGlobalModelInfo(),
  getGlobalModelOptions: () => getGlobalModelOptions(),
  getAuxiliaryModels: () => getAuxiliaryModels(),
  setModelAssignment: (body: unknown) => setModelAssignment(body),
  getRecommendedDefaultModel: (slug: string) => getRecommendedDefaultModel(slug),
  setEnvVar: (key: string, value: string) => setEnvVar(key, value)
}))

vi.mock('@/store/onboarding', () => ({
  startManualProviderOAuth: (slug: string) => startManualProviderOAuth(slug)
}))

beforeEach(() => {
  getGlobalModelInfo.mockResolvedValue({ provider: 'nous', model: 'hermes-4' })
  getGlobalModelOptions.mockResolvedValue({
    providers: [
      { name: 'Nous', slug: 'nous', models: ['hermes-4', 'hermes-4-mini'], authenticated: true },
      // An unconfigured api_key provider — surfaced by the full-universe payload.
      { name: 'DeepSeek', slug: 'deepseek', models: [], authenticated: false, auth_type: 'api_key', key_env: 'DEEPSEEK_API_KEY' }
    ]
  })
  getAuxiliaryModels.mockResolvedValue({
    main: { provider: 'nous', model: 'hermes-4' },
    tasks: [{ task: 'vision', provider: 'auto', model: '', base_url: '' }]
  })
  setModelAssignment.mockResolvedValue({ provider: 'nous', model: 'hermes-4', gateway_tools: [] })
  getRecommendedDefaultModel.mockResolvedValue({ provider: 'deepseek', model: 'deepseek-chat', free_tier: null })
  setEnvVar.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderModelSettings() {
  const { ModelSettings } = await import('./model-settings')

  return render(<ModelSettings />)
}

describe('ModelSettings', () => {
  it('loads the current main model and lists the full provider universe', async () => {
    await renderModelSettings()

    await waitFor(() => expect(getGlobalModelInfo).toHaveBeenCalled())
    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())

    // Open the provider Select — every provider from the full payload should be
    // listed, including the unconfigured one with its "set up" hint.
    const triggers = await screen.findAllByRole('combobox')
    fireEvent.click(triggers[0])

    // "Nous" shows in both the trigger and the open list; the unconfigured
    // provider + its setup hint are the unique signal of the full universe.
    expect((await screen.findAllByText('Nous')).length).toBeGreaterThan(0)
    expect(await screen.findByText(/DeepSeek/)).toBeTruthy()
    expect(await screen.findByText(/set up/)).toBeTruthy()
  })

  it('activates an unconfigured api_key provider inline by saving its key', async () => {
    await renderModelSettings()

    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())

    // Open the provider Select and pick the unconfigured provider.
    const triggers = screen.getAllByRole('combobox')
    fireEvent.click(triggers[0])
    const deepseekOption = await screen.findByText(/DeepSeek/)
    fireEvent.click(deepseekOption)

    // The inline key input appears for an api_key provider that needs setup.
    const keyInput = await screen.findByPlaceholderText(/Paste DEEPSEEK_API_KEY/)
    fireEvent.change(keyInput, { target: { value: 'sk-test-123' } })

    const activate = await screen.findByRole('button', { name: /Activate/ })
    fireEvent.click(activate)

    await waitFor(() => expect(setEnvVar).toHaveBeenCalledWith('DEEPSEEK_API_KEY', 'sk-test-123'))
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
})
