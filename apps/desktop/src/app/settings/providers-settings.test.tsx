import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ConfirmHost } from '@/components/confirm-host'
import { $confirmRequest } from '@/store/confirm'
import type { EnvVarInfo, OAuthProvider } from '@/types/hermes'

const listOAuthProviders = vi.fn()
const disconnectOAuthProvider = vi.fn()
const getEnvVars = vi.fn()
const setEnvVar = vi.fn()
const startManualProviderOAuth = vi.fn()
const startManualLocalEndpoint = vi.fn()
const onboarding = atom({ manual: false })

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: atom('alpha'),
  $profiles: atom([]),
  refreshProfiles: async () => {},
  normalizeProfileKey: (p: string | null) => p || 'default'
}))

vi.mock('@/hermes', () => ({
  setApiRequestProfile: vi.fn(),
  getProfiles: async () => ({ profiles: (await import('@/store/profile')).$profiles.get() }),
  setEnvVar: (key: string, value: string, profile?: string) => setEnvVar(key, value, profile),
  disconnectOAuthProvider: (...args: unknown[]) => disconnectOAuthProvider(...args),
  getEnvVars: (...args: unknown[]) => getEnvVars(...args),
  listOAuthProviders: (...args: unknown[]) => listOAuthProviders(...args)
}))

vi.mock('@/store/onboarding', () => ({
  $desktopOnboarding: onboarding,
  startManualProviderOAuth: (...args: unknown[]) => startManualProviderOAuth(...args),
  startManualLocalEndpoint: (reason: null | string) => startManualLocalEndpoint(reason)
}))

function provider(id: string, loggedIn: boolean, patch: Partial<OAuthProvider> = {}): OAuthProvider {
  return {
    cli_command: `hermes auth add ${id}`,
    disconnectable: true,
    docs_url: '',
    flow: 'device_code',
    id,
    name: id === 'nous' ? 'Nous Portal' : 'MiniMax',
    status: {
      logged_in: loggedIn
    },
    ...patch
  }
}

// One `/api/env` row (an EnvVarInfo) for the API-keys view. Mirrors the
// `provider()` factory above: a valid base + per-test overrides, typed against
// the real response shape so it can't drift from EnvVarInfo.
function keyVar(patch: Partial<EnvVarInfo> = {}): EnvVarInfo {
  return {
    advanced: false,
    category: 'provider',
    description: '',
    is_password: true,
    is_set: false,
    provider: '',
    provider_label: '',
    redacted_value: null,
    tools: [],
    url: '',
    ...patch
  }
}

beforeEach(() => {
  onboarding.set({ manual: false })
  getEnvVars.mockResolvedValue({})
  disconnectOAuthProvider.mockResolvedValue({ ok: true, provider: 'nous' })
  listOAuthProviders.mockResolvedValue({
    providers: [provider('nous', true), provider('minimax-oauth', false)]
  })
})

afterEach(() => {
  cleanup()
  $confirmRequest.set(null)
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

// Removal goes through confirm() from @/store/confirm, so the host has to be
// mounted for the prompt to render — same as in the real app shell.
async function renderProvidersSettings() {
  const { ProvidersSettings } = await import('./providers-settings')
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      <>
        <ProvidersSettings onClose={vi.fn()} onViewChange={vi.fn()} view="accounts" />
        <ConfirmHost />
      </>
    )
  })

  return result!
}

describe('ProvidersSettings', () => {
  it('reads and saves API keys for the shared Settings target and reloads when it changes', async () => {
    const { $settingsScopeOverride } = await import('@/store/settings-scope')
    const { $activeGatewayProfile, $profiles } = await import('@/store/profile')
    $activeGatewayProfile.set('profile-a')
    $settingsScopeOverride.set('profile-b')
    $profiles.set(
      ['profile-a', 'profile-b'].map(name => ({
        name,
        has_env: false,
        is_default: false,
        model: null,
        path: '',
        provider: null,
        skill_count: 0
      }))
    )
    getEnvVars.mockResolvedValue({ WIDGET_API_KEY: keyVar({ provider: 'widget', provider_label: 'Widget' }) })
    const { ProvidersSettings } = await import('./providers-settings')

    try {
      const { container } = render(<ProvidersSettings onClose={vi.fn()} onViewChange={vi.fn()} view="keys" />)
      await screen.findByText('Widget')
      expect(getEnvVars).toHaveBeenLastCalledWith('profile-b')
      expect(screen.getByText('Applies to')).toBeTruthy()
      const input = container.querySelector('input[type="password"]')!
      fireEvent.focus(input)
      fireEvent.change(input, { target: { value: 'fixture-key' } })
      fireEvent.click(screen.getByRole('button', { name: 'Save' }))
      await waitFor(() => expect(setEnvVar).toHaveBeenCalledWith('WIDGET_API_KEY', 'fixture-key', 'profile-b'))
      fireEvent.click(screen.getByRole('button', { name: 'profile-a' }))
      await waitFor(() => expect(getEnvVars).toHaveBeenLastCalledWith(undefined))
    } finally {
      cleanup()
      $settingsScopeOverride.set(null)
      $activeGatewayProfile.set('default')
      $profiles.set([])
    }
  })

  it('uses the settings target for account reads, removal and sign-in', async () => {
    const { $settingsScopeOverride } = await import('@/store/settings-scope')
    $settingsScopeOverride.set('beta')

    try {
      await renderProvidersSettings()
      expect(getEnvVars).toHaveBeenCalledWith('beta')
      expect(listOAuthProviders).toHaveBeenCalledWith('beta')
      fireEvent.click(await screen.findByText('Nous Portal'))
      expect(startManualProviderOAuth).toHaveBeenCalledWith('nous', 'beta')
      fireEvent.click(await screen.findByRole('button', { name: 'Remove Nous Portal' }))
      fireEvent.click(await screen.findByRole('button', { name: 'Disconnect' }))
      await waitFor(() => expect(disconnectOAuthProvider).toHaveBeenCalledWith('nous', 'beta'))
    } finally {
      $settingsScopeOverride.set(null)
    }
  })

  it('disconnects a connected provider account and refreshes the accounts list', async () => {
    await renderProvidersSettings()

    const remove = await screen.findByRole('button', { name: 'Remove Nous Portal' })
    await act(async () => {
      fireEvent.click(remove)
    })

    // Removal is confirmed first — nothing has been disconnected yet.
    expect(await screen.findByRole('dialog')).toBeTruthy()
    expect(disconnectOAuthProvider).not.toHaveBeenCalled()

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Disconnect' }))
    })

    await waitFor(() => expect(disconnectOAuthProvider).toHaveBeenCalledWith('nous', undefined))
    expect(listOAuthProviders).toHaveBeenCalledTimes(2)
  })

  it('leaves the account connected when the removal prompt is dismissed', async () => {
    await renderProvidersSettings()

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: 'Remove Nous Portal' }))
    })

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: 'Cancel' }))
    })

    expect(disconnectOAuthProvider).not.toHaveBeenCalled()
  })

  it('keeps provider selection separate from account removal', async () => {
    await renderProvidersSettings()

    await act(async () => {
      fireEvent.click(await screen.findByText('Nous Portal'))
    })

    expect(startManualProviderOAuth).toHaveBeenCalledWith('nous', undefined)
    expect(disconnectOAuthProvider).not.toHaveBeenCalled()
  })

  it('does not offer removal for externally managed providers', async () => {
    listOAuthProviders.mockResolvedValue({
      providers: [
        provider('qwen-oauth', true, {
          cli_command: 'hermes auth add qwen-oauth',
          disconnect_hint: "Use `hermes auth add qwen-oauth` or that provider's CLI to remove it.",
          disconnectable: false,
          flow: 'external',
          name: 'Qwen (via Qwen CLI)'
        })
      ]
    })

    await renderProvidersSettings()

    expect(await screen.findByText('Qwen Code')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Remove Qwen Code' })).toBeNull()
    expect(screen.getByText(/managed by its own CLI/)).toBeTruthy()
  })

  it('renders a Keys card for a backend-tagged provider with no PROVIDER_GROUPS prefix', async () => {
    // A provider the backend catalog tags (provider/provider_label) but that has
    // no desktop PROVIDER_GROUPS prefix row must still render its own card —
    // this is the GUI/CLI drift fix: membership comes from the backend, not
    // from the hand-maintained prefix list.
    getEnvVars.mockResolvedValue({
      WIDGETAI_API_KEY: keyVar({
        provider: 'widgetai',
        provider_label: 'WidgetAI',
        url: 'https://widgetai.example/keys'
      })
    })
    listOAuthProviders.mockResolvedValue({ providers: [] })

    const { ProvidersSettings } = await import('./providers-settings')
    await act(async () => {
      render(<ProvidersSettings onClose={vi.fn()} onViewChange={vi.fn()} view="keys" />)
    })

    expect(await screen.findByText('WidgetAI')).toBeTruthy()
  })

  it('orders API-key providers by priority then name, and filters them via search', async () => {
    // These three providers have no curated PROVIDER_GROUPS priority, so they
    // share the default priority and fall back to alphabetical among themselves
    // (Acme, Middle, Zebra) — exercising the name tiebreak of the priority sort.
    getEnvVars.mockResolvedValue({
      ZEBRA_API_KEY: keyVar({ provider: 'zebra', provider_label: 'Zebra' }),
      ACME_API_KEY: keyVar({ provider: 'acme', provider_label: 'Acme' }),
      MIDDLE_API_KEY: keyVar({ provider: 'middle', provider_label: 'Middle' })
    })
    listOAuthProviders.mockResolvedValue({ providers: [] })

    const { ProvidersSettings } = await import('./providers-settings')
    render(<ProvidersSettings onClose={vi.fn()} onViewChange={vi.fn()} view="keys" />)

    // Equal priority → alphabetical tiebreak: Acme, Middle, Zebra.
    await screen.findByText('Acme')
    const labels = screen.getAllByText(/Acme|Middle|Zebra/).map(el => el.textContent)
    expect(labels).toEqual(['Acme', 'Middle', 'Zebra'])

    // Typing narrows the list to matching providers only.
    const search = screen.getByPlaceholderText('Search providers…')
    await act(async () => {
      fireEvent.change(search, { target: { value: 'mid' } })
    })

    await waitFor(() => expect(screen.queryByText('Acme')).toBeNull())
    expect(screen.getByText('Middle')).toBeTruthy()
    expect(screen.queryByText('Zebra')).toBeNull()

    // A non-matching query shows the empty-state copy.
    await act(async () => {
      fireEvent.change(search, { target: { value: 'nonesuch-xyz' } })
    })
    expect(await screen.findByText('No providers match your search.')).toBeTruthy()
  })

  it('offers a Local / custom endpoint entry in the API-keys tab that opens the custom-endpoint flow', async () => {
    // Regression: the composer pill and the providers "have an API key"
    // affordance both dead-end on the env-var-driven key catalog, which never
    // lists a custom endpoint — so without this row there is no reachable
    // Desktop GUI path to add one. See issue #62817.
    getEnvVars.mockResolvedValue({})
    listOAuthProviders.mockResolvedValue({ providers: [] })

    const { ProvidersSettings } = await import('./providers-settings')
    render(<ProvidersSettings onClose={vi.fn()} onViewChange={vi.fn()} view="keys" />)

    const row = await screen.findByText('Local / custom endpoint')
    expect(screen.getByText(/OpenAI-compatible endpoint/)).toBeTruthy()

    fireEvent.click(row)

    await waitFor(() => expect(startManualLocalEndpoint).toHaveBeenCalledWith(null))
  })
})
