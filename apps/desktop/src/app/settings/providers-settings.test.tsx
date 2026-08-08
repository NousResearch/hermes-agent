import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { EnvVarInfo, OAuthProvider } from '@/types/hermes'

const listOAuthProviders = vi.fn()
const disconnectOAuthProvider = vi.fn()
const activateCredentialPoolEntry = vi.fn()
const getEnvVars = vi.fn()
const getCredentialPool = vi.fn()
const startManualProviderOAuth = vi.fn()
const startManualLocalEndpoint = vi.fn()
const onboarding = atom({ manual: false })

vi.mock('@/hermes', () => ({
  activateCredentialPoolEntry: (provider: string, index: number) => activateCredentialPoolEntry(provider, index),
  disconnectOAuthProvider: (providerId: string) => disconnectOAuthProvider(providerId),
  getCredentialPool: () => getCredentialPool(),
  getEnvVars: () => getEnvVars(),
  listOAuthProviders: () => listOAuthProviders()
}))

vi.mock('@/store/onboarding', () => ({
  $desktopOnboarding: onboarding,
  startManualProviderOAuth: (providerId: string) => startManualProviderOAuth(providerId),
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
  getCredentialPool.mockResolvedValue({ providers: [] })
  disconnectOAuthProvider.mockResolvedValue({ ok: true, provider: 'nous' })
  activateCredentialPoolEntry.mockResolvedValue({ ok: true, provider: 'copilot' })
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
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(<ProvidersSettings onClose={vi.fn()} onViewChange={vi.fn()} view="accounts" />)
  })

  return result!
}

describe('ProvidersSettings', () => {
  it('disconnects a connected provider account and refreshes the accounts list', async () => {
    await renderProvidersSettings()

    const remove = await screen.findByRole('button', { name: 'Remove Nous Portal' })
    await act(async () => {
      fireEvent.click(remove)
    })

    await waitFor(() => expect(disconnectOAuthProvider).toHaveBeenCalledWith('nous'))
    expect(listOAuthProviders).toHaveBeenCalledTimes(2)
  })

  it('keeps provider selection separate from account removal', async () => {
    await renderProvidersSettings()

    await act(async () => {
      fireEvent.click(await screen.findByText('Nous Portal'))
    })

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

  it('shows a per-credential pool status only for a provider with >1 stored credential', async () => {
    // Regression: multi-credential providers (e.g. a personal + a shared
    // Copilot key) had no Desktop UI to see which stored credential is
    // actually active. Copilot's key lives in the API-keys view, keyed by the
    // backend's raw provider id ("copilot"), which is what /api/credentials/pool
    // groups by too — see issue #80828.
    getEnvVars.mockResolvedValue({
      COPILOT_GITHUB_TOKEN: keyVar({ provider: 'copilot', provider_label: 'GitHub Copilot', is_set: true }),
      ACME_API_KEY: keyVar({ provider: 'acme', provider_label: 'Acme', is_set: true })
    })
    getCredentialPool.mockResolvedValue({
      providers: [
        {
          provider: 'copilot',
          entries: [
            { index: 1, label: 'wadefengx', priority: 0, request_count: 12, token_preview: 'sk-…abcd', has_refresh: false, last_status: 'ok' },
            { index: 2, label: 'shared-fallback', priority: 1, request_count: 0, token_preview: 'sk-…wxyz', has_refresh: false, last_status: 'exhausted' }
          ]
        }
      ]
    })
    listOAuthProviders.mockResolvedValue({ providers: [] })

    const { ProvidersSettings } = await import('./providers-settings')
    render(<ProvidersSettings onClose={vi.fn()} onViewChange={vi.fn()} view="keys" />)

    expect(await screen.findByText('wadefengx')).toBeTruthy()
    expect(screen.getByText('shared-fallback')).toBeTruthy()
    expect(screen.getByText('exhausted')).toBeTruthy()
    // Acme has no pool entries in this test → no status list rendered for it.
    expect(screen.queryByText('sk-…wxyz')).toBeNull()
  })

  it('lets the user click "Use this" to switch which stored credential is active', async () => {
    // The Desktop-only path for a user who won't touch the CLI (issue #80828
    // follow-up): clicking the button on the non-current entry calls the new
    // activate endpoint and re-fetches the pool so the status list updates.
    getEnvVars.mockResolvedValue({
      COPILOT_GITHUB_TOKEN: keyVar({ provider: 'copilot', provider_label: 'GitHub Copilot', is_set: true })
    })
    getCredentialPool.mockResolvedValue({
      providers: [
        {
          provider: 'copilot',
          entries: [
            { index: 1, label: 'wadefengx', priority: 0, request_count: 12, token_preview: 'sk-…abcd', has_refresh: false, last_status: 'ok' },
            { index: 2, label: 'shared-fallback', priority: 1, request_count: 0, token_preview: 'sk-…wxyz', has_refresh: false, last_status: 'ok' }
          ]
        }
      ]
    })
    listOAuthProviders.mockResolvedValue({ providers: [] })

    const { ProvidersSettings } = await import('./providers-settings')
    render(<ProvidersSettings onClose={vi.fn()} onViewChange={vi.fn()} view="keys" />)

    await screen.findByText('shared-fallback')
    // Only the non-current entry (index 2) gets a switch button.
    expect(screen.queryAllByRole('button', { name: 'Use this' })).toHaveLength(1)

    fireEvent.click(screen.getByRole('button', { name: 'Use this' }))

    await waitFor(() => expect(activateCredentialPoolEntry).toHaveBeenCalledWith('copilot', 2))
    expect(getCredentialPool).toHaveBeenCalledTimes(2)
  })
})
