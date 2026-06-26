import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MemoryProviderConfig } from '@/types/hermes'

const getMemoryProviderConfig = vi.fn()
const saveMemoryProviderConfig = vi.fn()
const startMemoryProviderOAuth = vi.fn()
const getMemoryProviderOAuthStatus = vi.fn()

vi.mock('@/hermes', () => ({
  getMemoryProviderConfig: (provider: string) => getMemoryProviderConfig(provider),
  saveMemoryProviderConfig: (provider: string, values: unknown) => saveMemoryProviderConfig(provider, values),
  startMemoryProviderOAuth: (provider: string) => startMemoryProviderOAuth(provider),
  getMemoryProviderOAuthStatus: (provider: string) => getMemoryProviderOAuthStatus(provider)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

function hindsightSchema(overrides: Partial<MemoryProviderConfig['fields'][number]>[] = []): MemoryProviderConfig {
  const fields: MemoryProviderConfig['fields'] = [
    {
      key: 'mode',
      label: 'Mode',
      kind: 'select',
      value: 'cloud',
      description: 'How Hermes connects to Hindsight.',
      placeholder: '',
      is_set: true,
      options: [
        { value: 'cloud', label: 'Cloud', description: 'Hindsight Cloud API (lightweight, just needs an API key)' },
        { value: 'local_external', label: 'Local External', description: 'Connect to an existing Hindsight instance' }
      ]
    },
    {
      key: 'api_key',
      label: 'API key',
      kind: 'secret',
      value: '',
      description: 'Used to authenticate with the Hindsight API.',
      placeholder: 'Enter Hindsight API key',
      is_set: false,
      options: []
    },
    {
      key: 'api_url',
      label: 'API URL',
      kind: 'text',
      value: 'https://api.hindsight.vectorize.io',
      description: '',
      placeholder: '',
      is_set: true,
      options: []
    },
    { key: 'bank_id', label: 'Bank ID', kind: 'text', value: 'hermes', description: '', placeholder: '', is_set: true, options: [] },
    {
      key: 'recall_budget',
      label: 'Recall budget',
      kind: 'select',
      value: 'mid',
      description: '',
      placeholder: '',
      is_set: true,
      options: [
        { value: 'low', label: 'low', description: '' },
        { value: 'mid', label: 'mid', description: '' },
        { value: 'high', label: 'high', description: '' }
      ]
    }
  ]

  return {
    name: 'hindsight',
    label: 'Hindsight',
    fields: fields.map((field, index) => ({ ...field, ...overrides[index] }))
  }
}

function withOAuth(
  oauth: MemoryProviderConfig['oauth'],
  overrides: Partial<MemoryProviderConfig['fields'][number]>[] = []
): MemoryProviderConfig {
  return { ...hindsightSchema(overrides), oauth }
}

beforeEach(() => {
  getMemoryProviderConfig.mockResolvedValue(hindsightSchema())
  saveMemoryProviderConfig.mockResolvedValue({ ok: true })
  startMemoryProviderOAuth.mockResolvedValue({ ok: true, status: 'pending' })
  getMemoryProviderOAuthStatus.mockResolvedValue({
    supported: true,
    authenticated: true,
    org_id: 'org_1',
    flow: 'done',
    error: null
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderPanel(provider = 'hindsight') {
  const { ProviderConfigPanel } = await import('./provider-config-panel')

  return render(<ProviderConfigPanel provider={provider} />)
}

describe('ProviderConfigPanel', () => {
  it('renders the declared provider fields generically', async () => {
    await renderPanel()

    expect(await screen.findByDisplayValue('https://api.hindsight.vectorize.io')).toBeTruthy()
    expect(screen.getByDisplayValue('hermes')).toBeTruthy()
    expect(screen.getByText('Cloud')).toBeTruthy()
    expect(screen.getAllByText('Hindsight Cloud API (lightweight, just needs an API key)').length).toBeGreaterThan(0)
    expect(screen.getByText('mid')).toBeTruthy()
  })

  it('collapses and expands the fields', async () => {
    await renderPanel()

    expect(await screen.findByLabelText('API URL')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /Hindsight settings/ }))
    expect(screen.queryByLabelText('API URL')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /Hindsight settings/ }))
    expect(await screen.findByLabelText('API URL')).toBeTruthy()
  })

  it('saves edited values without requiring a secret replacement', async () => {
    await renderPanel()

    const apiUrl = await screen.findByLabelText('API URL')
    fireEvent.change(apiUrl, { target: { value: 'http://localhost:8888' } })
    fireEvent.change(screen.getByLabelText('Bank ID'), { target: { value: 'ben-bank' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() =>
      expect(saveMemoryProviderConfig).toHaveBeenCalledWith('hindsight', {
        mode: 'cloud',
        api_key: '',
        api_url: 'http://localhost:8888',
        bank_id: 'ben-bank',
        recall_budget: 'mid'
      })
    )
  })

  it('renders nothing for a provider with no declared config surface', async () => {
    getMemoryProviderConfig.mockResolvedValue({ name: 'builtin', label: 'builtin', fields: [] })

    const { container } = await renderPanel('builtin')

    await waitFor(() => expect(getMemoryProviderConfig).toHaveBeenCalledWith('builtin'))
    expect(container.querySelector('section')).toBeNull()
  })

  it('starts the browser flow and polls status until connected', async () => {
    getMemoryProviderConfig.mockResolvedValue(
      withOAuth({ supported: true, authenticated: false, org_id: '', method: '' })
    )

    await renderPanel()

    const signIn = await screen.findByRole('button', { name: /Sign in with browser/ })
    fireEvent.click(signIn)

    await waitFor(() => expect(startMemoryProviderOAuth).toHaveBeenCalledWith('hindsight'))
    // Polls the status endpoint (after a short delay) until flow === 'done'...
    await waitFor(() => expect(getMemoryProviderOAuthStatus).toHaveBeenCalledWith('hindsight'), { timeout: 4000 })
    // ...then re-reads the provider config to show the Connected state.
    await waitFor(() => expect(getMemoryProviderConfig).toHaveBeenCalledTimes(2), { timeout: 4000 })
  }, 10000)

  it('does not report success on a pre-existing token until the new flow completes', async () => {
    getMemoryProviderConfig.mockResolvedValue(
      withOAuth({ supported: true, authenticated: true, org_id: 'old', method: 'oauth' })
    )
    // A token already exists (authenticated: true) but THIS sign-in is still
    // pending on the first poll, then completes on the second.
    getMemoryProviderOAuthStatus
      .mockResolvedValueOnce({ supported: true, authenticated: true, org_id: 'old', flow: 'pending', error: null })
      .mockResolvedValue({ supported: true, authenticated: true, org_id: 'new', flow: 'done', error: null })

    await renderPanel()
    // authenticated → the button reads "Re-authenticate".
    fireEvent.click(await screen.findByRole('button', { name: /Re-authenticate/ }))

    // Refresh (2nd config read) happens only after the SECOND poll (flow: done),
    // proving it didn't short-circuit on the first pending+authenticated poll.
    await waitFor(() => expect(getMemoryProviderConfig).toHaveBeenCalledTimes(2), { timeout: 8000 })
    expect(getMemoryProviderOAuthStatus).toHaveBeenCalledTimes(2)
  }, 12000)

  it('surfaces a background sign-in error from the status poll', async () => {
    getMemoryProviderConfig.mockResolvedValue(
      withOAuth({ supported: true, authenticated: false, org_id: '', method: '' })
    )
    getMemoryProviderOAuthStatus.mockResolvedValue({
      supported: true,
      authenticated: false,
      org_id: '',
      flow: 'error',
      error: 'user declined'
    })

    await renderPanel()
    fireEvent.click(await screen.findByRole('button', { name: /Sign in with browser/ }))

    await waitFor(() => expect(getMemoryProviderOAuthStatus).toHaveBeenCalledWith('hindsight'), { timeout: 4000 })
    // On error the panel does NOT refresh the config (stays at the initial load).
    await waitFor(() => expect(getMemoryProviderConfig).toHaveBeenCalledTimes(1), { timeout: 4000 })
  }, 10000)

  it('shows a Connected pill and Re-authenticate when already authenticated', async () => {
    getMemoryProviderConfig.mockResolvedValue(
      withOAuth({ supported: true, authenticated: true, org_id: 'org_42', method: 'oauth' })
    )

    await renderPanel()

    expect(await screen.findByText(/Connected — org org_42/)).toBeTruthy()
    expect(screen.getByRole('button', { name: /Re-authenticate/ })).toBeTruthy()
  })

  it('shows the org name in the Connected pill when available', async () => {
    getMemoryProviderConfig.mockResolvedValue(
      withOAuth({ supported: true, authenticated: true, org_id: 'org_42', org_name: "Ben's Org", method: 'oauth' })
    )

    await renderPanel()

    expect(await screen.findByText(/Connected — Ben's Org/)).toBeTruthy()
    expect(screen.queryByText(/org_42/)).toBeNull()
  })

  it('hides the sign-in button when mode is not cloud', async () => {
    getMemoryProviderConfig.mockResolvedValue(
      withOAuth({ supported: true, authenticated: false, org_id: '', method: '' }, [
        { value: 'local_external' }
      ])
    )

    await renderPanel()

    await screen.findByLabelText('API URL')
    expect(screen.queryByRole('button', { name: /Sign in with browser/ })).toBeNull()
  })

  it('shows no sign-in button when the provider declares no oauth surface', async () => {
    await renderPanel()

    await screen.findByLabelText('API URL')
    expect(screen.queryByRole('button', { name: /Sign in with browser/ })).toBeNull()
  })
})
