import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { getCustomEndpoints, getEnvVars, getGlobalModelOptions, setEnvVar } from '@/hermes'
import { $visibleModels } from '@/store/model-visibility'
import type { ModelOptionProvider } from '@/types/hermes'

import { ProviderModelManager } from './provider-model-manager'

const providers: ModelOptionProvider[] = [
  { slug: 'openai', name: 'OpenAI', models: ['gpt-4o', 'gpt-4o-mini'] },
  { slug: 'anthropic', name: 'Anthropic', models: ['claude-3-5-sonnet'] },
  // An unconfigured-but-configurable built-in provider: no models yet, api_key
  // auth flow, and a key_env to write the API key into. The widened catalog
  // filter keeps it so the manager can surface it in the Unconfigured group.
  {
    slug: 'deepseek',
    name: 'DeepSeek',
    models: [],
    auth_type: 'api_key',
    key_env: 'DEEPSEEK_API_KEY',
    authenticated: false
  }
]

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof import('@/hermes')>()),
  getGlobalModelOptions: vi.fn(() => ({ providers })),
  getCustomEndpoints: vi.fn(() => ({ endpoints: [], current: { provider: '', model: '', base_url: '' } })),
  getEnvVars: vi.fn(() => ({})),
  saveCustomEndpoint: vi.fn(() => Promise.resolve({ ok: true, endpoints: [] })),
  deleteCustomEndpoint: vi.fn(() => Promise.resolve({ ok: true, endpoints: [] })),
  setCustomEndpointEnabled: vi.fn(() => Promise.resolve({ ok: true, endpoints: [] })),
  setEnvVar: vi.fn(() => ({ ok: true }))
}))

function renderManager(initialEntries: string[] = ['/']) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={initialEntries}>
        <ProviderModelManager />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('ProviderModelManager', () => {
  beforeEach(() => {
    $visibleModels.set(null)
  })

  it('shows the first provider’s models by default', async () => {
    renderManager()

    // The catalog loads asynchronously (useProviderModelCatalog), so wait for it.
    // The provider name appears in both the nav row and the right-pane header,
    // so scope the assertion to the nav entry (the first match).
    await screen.findAllByText('OpenAI')
    const switches = await screen.findAllByRole('switch')
    // OpenAI exposes two models → two model toggles + one provider activation toggle.
    expect(switches).toHaveLength(3)
  })

  it('reopens the Add provider dialog (not Edit credentials) after editing built-in credentials', async () => {
    renderManager()
    await screen.findAllByText('OpenAI')

    // Open the built-in "Edit credentials" dialog, then close it. The dialog
    // blocks Escape/outside-click dismissal, so dismiss via Cancel.
    fireEvent.click(screen.getByRole('button', { name: 'Edit credentials' }))
    expect(screen.getByRole('heading', { name: 'Edit credentials' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())

    // Reopen via "Add provider…" — must show the Add dialog, not the stale
    // Edit-credentials dialog (regression: editMode stayed 'builtin').
    fireEvent.click(screen.getByRole('button', { name: 'Add provider…' }))
    expect(screen.getByRole('heading', { name: 'Add provider…' })).toBeTruthy()
    expect(screen.queryByRole('heading', { name: 'Edit credentials' })).toBeNull()
  })

  it('switches the right pane when a provider is clicked', async () => {
    renderManager()

    // The provider name appears in both the nav row and the right-pane header,
    // so scope the click to the nav entry (the first match).
    const navItems = await screen.findAllByText('Anthropic')
    fireEvent.click(navItems[0])

    await waitFor(() => expect(screen.getAllByRole('switch')).toHaveLength(2))
    const option = navItems[0].closest('[role="option"]')
    expect(option?.getAttribute('aria-selected')).toBe('true')
  })

  it('deep-links the selected provider via ?pmprovider=', async () => {
    renderManager(['/?pmprovider=anthropic'])

    await waitFor(() => expect(screen.getAllByRole('switch')).toHaveLength(2))
    const navItems = screen.getAllByText('Anthropic')
    const option = navItems[0].closest('[role="option"]')
    expect(option?.getAttribute('aria-selected')).toBe('true')
  })

  it('applies the standard settings horizontal gutter and bottom padding', async () => {
    renderManager()
    await screen.findAllByText('OpenAI')

    // The outermost flex container carries PAGE_INSET_X + pb-6 so the manager
    // matches the spacing of every other settings section. Walk up from a nav
    // option to the unique pb-6 container (inner elements don't carry pb-6).
    const navOption = screen.getByRole('option', { name: /OpenAI/ })
    const container = navOption.closest('.pb-6')
    expect(container?.className).toContain('px-[clamp(1.25rem,4vw,4rem)]')
    expect(container?.className).toContain('pb-6')
  })

  it('renders a page header with the title and description above the content', async () => {
    renderManager()
    await screen.findAllByText('OpenAI')

    // The header is a semantic <header> banner carrying the page title and a
    // short description, divided from the two-column body by a bottom border.
    const header = screen.getByRole('banner')
    expect(header.className).toContain('border-b')
    expect(header.textContent).toContain('Provider Manager')
    expect(header.textContent).toContain('Enable providers')
  })

  describe('discover / update for all providers', () => {
    it('shows the "Update list" button for a built-in provider with models', async () => {
      renderManager()
      await screen.findAllByText('OpenAI')

      // OpenAI has models → "Update list"
      expect(screen.getByRole('button', { name: /update list/i })).toBeTruthy()
    })

    it('does NOT show "Add model" for a built-in provider', async () => {
      renderManager()
      await screen.findAllByText('OpenAI')

      expect(screen.queryByRole('button', { name: /add model/i })).toBeNull()
    })

    it('calls getGlobalModelOptions with refresh:true when clicking "Update list" on a built-in provider', async () => {
      renderManager()
      await screen.findAllByText('OpenAI')

      vi.mocked(getGlobalModelOptions).mockClear()
      fireEvent.click(screen.getByRole('button', { name: /update list/i }))

      await waitFor(() =>
        expect(getGlobalModelOptions).toHaveBeenCalledWith(
          expect.objectContaining({ refresh: true })
        )
      )
    })

    it('shows the "Model list refreshed" banner after a successful built-in refresh', async () => {
      renderManager()
      await screen.findAllByText('OpenAI')

      fireEvent.click(screen.getByRole('button', { name: /update list/i }))

      await waitFor(() =>
        expect(screen.getByText(/model list refreshed/i)).toBeTruthy()
      )
    })
  })

  describe('unconfigured provider configure panel', () => {
    it('shows the configure panel (not the model list) for an unconfigured api_key provider', async () => {
      renderManager(['/?pmprovider=deepseek'])

      // The configure panel renders the description + an API key field, and does
      // NOT render the model list's "Update list" / "Discover models" button.
      await waitFor(() => expect(screen.getByText(/Enter your API key/i)).toBeTruthy())
      expect(screen.getByLabelText('API key')).toBeTruthy()
      expect(screen.getByRole('button', { name: /Save & discover models/i })).toBeTruthy()
      expect(screen.queryByRole('button', { name: /update list/i })).toBeNull()
    })

    it('shows the normal model list for a configured provider', async () => {
      renderManager(['/?pmprovider=openai'])

      await screen.findAllByText('OpenAI')
      // Configured provider → model list with an "Update list" button, no panel.
      expect(screen.getByRole('button', { name: /update list/i })).toBeTruthy()
      expect(screen.queryByText(/Enter your API key/i)).toBeNull()
    })

    it('persists the API key via setEnvVar and re-probes the catalog on save', async () => {
      renderManager(['/?pmprovider=deepseek'])

      await waitFor(() => expect(screen.getByText(/Enter your API key/i)).toBeTruthy())

      fireEvent.change(screen.getByLabelText('API key'), { target: { value: 'sk-deep-123' } })
      vi.mocked(getGlobalModelOptions).mockClear()
      fireEvent.click(screen.getByRole('button', { name: /Save & discover models/i }))

      // The key is written to the provider's key_env env var…
      await waitFor(() => expect(setEnvVar).toHaveBeenCalledWith('DEEPSEEK_API_KEY', 'sk-deep-123'))
      // …and the catalog is re-probed with refresh:true so models are discovered
      // and the provider flips to configured.
      await waitFor(() =>
        expect(getGlobalModelOptions).toHaveBeenCalledWith(expect.objectContaining({ refresh: true }))
      )
    })
  })

  describe('add-provider id generation (collision set from endpoints)', () => {
    it('uniquifies the generated id against existing custom providers', async () => {
      // Seed an existing custom provider "lab" via the REST endpoints API.
      vi.mocked(getCustomEndpoints).mockReturnValue({
        endpoints: [
          {
            id: 'lab',
            name: 'lab',
            base_url: 'https://lab/v1',
            model: '',
            models: [],
            discover_models: true,
            has_api_key: false,
            enabled: true
          }
        ],
        current: { provider: '', model: '', base_url: '' }
      } as never)

      renderManager()
      await screen.findAllByText('OpenAI')

      // Open the Add provider dialog.
      fireEvent.click(screen.getByRole('button', { name: /add provider/i }))

      // Type a friendly name that collides with the existing "lab".
      fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Lab' } })

      // The read-only Provider ID preview resolves the collision to "lab-2".
      expect(screen.getByLabelText('Provider ID').textContent).toBe('lab-2')
    })
  })
})
