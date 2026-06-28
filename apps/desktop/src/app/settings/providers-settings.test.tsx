import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { EnvVarInfo, OAuthProvider } from '@/types/hermes'

const listOAuthProviders = vi.fn()
const disconnectOAuthProvider = vi.fn()
const getEnvVars = vi.fn()
const startManualProviderOAuth = vi.fn()
const onboarding = atom({ manual: false })

vi.mock('@/hermes', () => ({
  disconnectOAuthProvider: (providerId: string) => disconnectOAuthProvider(providerId),
  getEnvVars: () => getEnvVars(),
  listOAuthProviders: () => listOAuthProviders()
}))

vi.mock('@/store/onboarding', () => ({
  $desktopOnboarding: onboarding,
  startManualProviderOAuth: (providerId: string) => startManualProviderOAuth(providerId)
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
  vi.spyOn(window, 'confirm').mockReturnValue(true)
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

async function renderProvidersSettings() {
  const { ProvidersSettings } = await import('./providers-settings')

  return render(<ProvidersSettings onClose={vi.fn()} onViewChange={vi.fn()} view="accounts" />)
}

describe('ProvidersSettings', () => {
  it('disconnects a connected provider account and refreshes the accounts list', async () => {
    await renderProvidersSettings()

    const remove = await screen.findByRole('button', { name: 'Remove Nous Portal' })
    fireEvent.click(remove)

    await waitFor(() => expect(disconnectOAuthProvider).toHaveBeenCalledWith('nous'))
    expect(listOAuthProviders).toHaveBeenCalledTimes(2)
  }, 15_000)

  it('keeps provider selection separate from account removal', async () => {
    await renderProvidersSettings()

    fireEvent.click(await screen.findByText('Nous Portal'))

    expect(startManualProviderOAuth).toHaveBeenCalledWith('nous')
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
    render(<ProvidersSettings onClose={vi.fn()} onViewChange={vi.fn()} view="keys" />)

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
    fireEvent.change(search, { target: { value: 'mid' } })

    await waitFor(() => expect(screen.queryByText('Acme')).toBeNull())
    expect(screen.getByText('Middle')).toBeTruthy()
    expect(screen.queryByText('Zebra')).toBeNull()

    // A non-matching query shows the empty-state copy.
    fireEvent.change(search, { target: { value: 'nonesuch-xyz' } })
    expect(await screen.findByText('No providers match your search.')).toBeTruthy()
  })
})
