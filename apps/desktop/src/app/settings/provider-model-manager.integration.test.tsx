// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $visibleModels, emptyProviderSentinelKey } from '@/store/model-visibility'
import type { ModelOptionProvider } from '@/types/hermes'
import { saveCustomEndpoint, setCustomEndpointEnabled } from '@/hermes'

import { ProviderModelManager } from './provider-model-manager'

const providers: ModelOptionProvider[] = [
  { slug: 'openai', name: 'OpenAI', models: ['gpt-4o'], enabled: true },
  { slug: 'custom:lab', name: 'Lab', models: ['a'], is_user_defined: true, enabled: true }
]

// Canonical REST endpoints the Provider Manager now sources custom providers from.
const endpoints = [{
  id: 'lab',
  name: 'Lab',
  base_url: 'https://lab/v1',
  model: 'a',
  models: ['a'],
  discover_models: true,
  has_api_key: false,
  enabled: true
}]

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof import('@/hermes')>()),
  getGlobalModelOptions: vi.fn(() => ({ providers })),
  getCustomEndpoints: vi.fn(() => ({ endpoints, current: { provider: '', model: '', base_url: '' } })),
  saveCustomEndpoint: vi.fn(() => Promise.resolve({ ok: true, endpoints })),
  deleteCustomEndpoint: vi.fn(() => Promise.resolve({ ok: true, endpoints })),
  setCustomEndpointEnabled: vi.fn(() => Promise.resolve({ ok: true, endpoints })),
  discoverProviderModels: vi.fn(() => Promise.resolve({ models: [{ id: 'b', name: 'B' }] }))
}))

function renderManager() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <ProviderModelManager />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('ProviderModelManager integration', () => {
  beforeEach(() => {
    $visibleModels.set(null)
    vi.mocked(saveCustomEndpoint).mockClear()
    vi.mocked(setCustomEndpointEnabled).mockClear()
  })

  it('adds a new custom provider via the dialog and hides its models by default', async () => {
    renderManager()

    // Open the add dialog from the nav header.
    fireEvent.click(screen.getByRole('button', { name: 'Add provider…' }))
    expect(screen.getByLabelText('Name')).toBeTruthy()

    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Fresh' } })
    fireEvent.change(screen.getByLabelText('Base URL'), { target: { value: 'https://fresh/v1' } })

    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(saveCustomEndpoint).toHaveBeenCalled())
    const update = vi.mocked(saveCustomEndpoint).mock.calls[0][0]
    // The stored identity is the generated id (normalized from the friendly name).
    expect(update.id).toBe('fresh')
    expect(update.base_url).toBe('https://fresh/v1')
    // New provider starts with every model hidden.
    expect($visibleModels.get()?.has(emptyProviderSentinelKey('custom:fresh'))).toBe(true)
  })

  it('toggles a custom provider’s activation via the enable endpoint', async () => {
    renderManager()

    // Select the custom provider, then flip its activation switch.
    fireEvent.click(await screen.findByText('Lab'))
    const toggle = await screen.findByLabelText('Disable provider')
    fireEvent.click(toggle)

    await waitFor(() => expect(setCustomEndpointEnabled).toHaveBeenCalledWith('lab', false))
  })

  it('opens the Edit (not Add) modal prefilled when editing a custom provider', async () => {
    renderManager()

    // Select the custom provider and click "Edit provider".
    fireEvent.click(await screen.findByText('Lab'))
    fireEvent.click(await screen.findByRole('button', { name: 'Edit provider' }))

    // The dialog title (heading) must be "Edit provider" (not "Add provider"),
    // and the Base URL field must be prefilled from the endpoint — regression
    // test for the bug where a custom provider's Edit button opened the Add modal.
    expect(await screen.findByRole('heading', { name: 'Edit provider' })).toBeTruthy()
    const baseUrlField = screen.getByLabelText('Base URL') as HTMLInputElement
    expect(baseUrlField.value).toBe('https://lab/v1')
  })

  it('discovers models and persists them via saveCustomEndpoint', async () => {
    renderManager()

    fireEvent.click(await screen.findByText('Lab'))
    // Lab already has 1 model → the button reads "Update list" (not "Discover models").
    fireEvent.click(await screen.findByRole('button', { name: 'Update list' }))

    await waitFor(() => expect(saveCustomEndpoint).toHaveBeenCalled())
    const update = vi.mocked(saveCustomEndpoint).mock.calls[0][0]
    // Discovered model 'b' is merged into the models list.
    expect(update.models).toContain('b')
    expect(update.models).toContain('a')
    // Provider starts hidden (sentinel) since it was default-visible.
    expect($visibleModels.get()?.has(emptyProviderSentinelKey('custom:lab'))).toBe(true)
  })

  it('manually adds a model and marks it active', async () => {
    renderManager()

    fireEvent.click(await screen.findByText('Lab'))
    fireEvent.click(await screen.findByRole('button', { name: 'Add model' }))

    fireEvent.change(await screen.findByLabelText('Model ID'), { target: { value: 'c' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(saveCustomEndpoint).toHaveBeenCalled())
    const update = vi.mocked(saveCustomEndpoint).mock.calls[0][0]
    expect(update.models).toContain('c')
    // Manually added model is active (visible).
    expect($visibleModels.get()?.has('custom:lab::c')).toBe(true)
  })
})
