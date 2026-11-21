import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, useNavigate } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ConfirmHost } from '@/components/confirm-host'
import { settleConfirm } from '@/store/confirm'
import { stubResizeObserver } from '@/test/jsdom'

import { envVar } from './test-utils'

const getEnvVars = vi.fn()
const deleteEnvVar = vi.fn()

stubResizeObserver()

vi.mock('@/hermes', () => ({
  deleteEnvVar: (...args: unknown[]) => deleteEnvVar(...args),
  getEnvVars: (profile?: null | string) => getEnvVars(profile),
  getProfiles: async () => ({ profiles: [] }),
  revealEnvVar: vi.fn(),
  setApiRequestProfile: () => undefined,
  setEnvVar: vi.fn()
}))

// Load once at module scope so no test's 15s budget pays the heavy transform
// + import (the first-test timeout flake under CI load).
const { KeysSettings } = await import('./keys-settings')
const { $settingsScopeOverride } = await import('@/store/settings-scope')

beforeEach(() => {
  getEnvVars.mockResolvedValue({})
  deleteEnvVar.mockResolvedValue({ ok: true })
  Object.defineProperty(Element.prototype, 'scrollIntoView', {
    configurable: true,
    value: vi.fn()
  })
})

afterEach(() => {
  cleanup()
  settleConfirm(false)
  vi.clearAllMocks()
})

async function renderKeysSettings(view: 'settings' | 'tools', route = '/settings') {
  await act(async () => {
    render(
      <MemoryRouter initialEntries={[route]}>
        <KeysSettings view={view} />
        <ConfirmHost />
      </MemoryRouter>
    )
  })
}

function DeepLinkButton({ target }: { target: string }) {
  const navigate = useNavigate()

  return (
    <button onClick={() => navigate(`/settings?tab=keys&key=${target}`)} type="button">
      Open key
    </button>
  )
}

describe('KeysSettings', () => {
  it('removes a saved key without editing it and keeps it when confirmation is cancelled', async () => {
    $settingsScopeOverride.set('profile-b')
    getEnvVars.mockResolvedValue({
      WIDGET_API_KEY: envVar('tool', { is_set: true, redacted_value: '••••1234' })
    })

    try {
      await renderKeysSettings('tools')

      const remove = await screen.findByRole('button', { name: 'Remove' })
      expect(screen.getByDisplayValue('••••1234').hasAttribute('readonly')).toBe(true)
      expect(screen.queryByRole('button', { name: 'Save' })).toBeNull()
      fireEvent.click(remove)

      const cancelled = await screen.findByRole('dialog')
      expect(deleteEnvVar).not.toHaveBeenCalled()
      fireEvent.click(within(cancelled).getByRole('button', { name: 'Cancel' }))
      expect(deleteEnvVar).not.toHaveBeenCalled()
      expect(screen.getByDisplayValue('••••1234')).toBeTruthy()

      fireEvent.click(screen.getByRole('button', { name: 'Remove' }))
      const confirmed = await screen.findByRole('dialog')
      fireEvent.click(within(confirmed).getByRole('button', { name: 'Confirm' }))

      await waitFor(() => expect(deleteEnvVar).toHaveBeenCalledWith('WIDGET_API_KEY', 'profile-b'))
      await waitFor(() => expect(screen.queryByRole('button', { name: 'Remove' })).toBeNull())
      expect(screen.queryByDisplayValue('••••1234')).toBeNull()
    } finally {
      cleanup()
      $settingsScopeOverride.set(null)
    }
  })

  it('fetches env vars for the displayed profile (the concrete key, never null) when unscoped', async () => {
    // #90549 class: getEnvVars(null) targets the primary profile's env store,
    // so a non-default profile's Keys page would read (and edit) the wrong
    // profile. #118432: `undefined` is equally wrong — profileScoped() then
    // drops `?profile=` entirely and the backend falls back to the home it was
    // LAUNCHED under, which need not be the profile this page displays. Send
    // the concrete key the page names.
    await renderKeysSettings('tools')

    await waitFor(() => expect(getEnvVars).toHaveBeenCalledWith('default'))
  })

  it('lists tools and excludes settings / channel-managed credentials', async () => {
    getEnvVars.mockResolvedValue({
      BRAVE_SEARCH_API_KEY: envVar('tool', { description: 'Search the web with Brave.' }),
      FIRECRAWL_API_KEY: envVar('tool', { description: 'Crawl and extract websites.' }),
      GATEWAY_PROXY: envVar('setting', { description: 'Gateway reverse proxy.' }),
      TELEGRAM_BOT_TOKEN: envVar('messaging', {
        channel_managed: true,
        description: 'Telegram bot token.'
      })
    })

    await renderKeysSettings('tools')

    expect(screen.getByText('BRAVE SEARCH')).toBeTruthy()
    expect(screen.getByText('FIRECRAWL')).toBeTruthy()
    expect(screen.queryByText('GATEWAY PROXY')).toBeNull()
    expect(screen.queryByText('TELEGRAM BOT')).toBeNull()
    expect(screen.queryByRole('combobox')).toBeNull()
  })

  it('lists settings rows and excludes tools / channel-managed credentials', async () => {
    getEnvVars.mockResolvedValue({
      API_SERVER_TOKEN: envVar('setting', { description: 'Protect the local API server.' }),
      GATEWAY_PROXY: envVar('messaging', { description: 'Gateway reverse proxy address.' }),
      TELEGRAM_BOT_TOKEN: envVar('messaging', {
        channel_managed: true,
        description: 'Telegram bot token.'
      }),
      BRAVE_SEARCH_API_KEY: envVar('tool', { description: 'Search the web with Brave.' })
    })

    await renderKeysSettings('settings')

    expect(screen.getByText('API SERVER')).toBeTruthy()
    expect(screen.getByText('GATEWAY PROXY')).toBeTruthy()
    expect(screen.queryByText('TELEGRAM BOT')).toBeNull()
    expect(screen.queryByText('BRAVE SEARCH')).toBeNull()
  })

  it('expands and highlights a deep-linked credential card', async () => {
    getEnvVars.mockResolvedValue({
      BRAVE_SEARCH_API_KEY: envVar('tool', { description: 'Search the web with Brave.' }),
      FIRECRAWL_API_KEY: envVar('tool', { description: 'Crawl and extract websites.' })
    })

    render(
      <MemoryRouter initialEntries={['/settings?tab=keys']}>
        <KeysSettings view="tools" />
        <DeepLinkButton target="FIRECRAWL_API_KEY" />
      </MemoryRouter>
    )

    expect(await screen.findByText('BRAVE SEARCH')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Open key' }))

    await waitFor(() => {
      const target = globalThis.document.getElementById('credential-key-FIRECRAWL_API_KEY')
      expect(target?.classList).toContain('setting-field-highlight')
    })
    expect(screen.getByText('Crawl and extract websites.')).toBeTruthy()
  })

  it('drops an unsaved credential edit when the settings target switches profile', async () => {
    // Regression: `vars` is re-fetched when the shared "Applies to" target
    // changes, but the in-flight edit map was not reset with it. A value typed
    // while targeting profile-b survived the switch to profile-c, where the
    // still-live Save would persist it into the WRONG profile.
    $settingsScopeOverride.set('profile-b')
    getEnvVars.mockResolvedValue({
      WIDGET_API_KEY: envVar('tool', { description: 'Widget key.', is_set: true, redacted_value: '••••••' })
    })

    try {
      const { container } = render(
        <MemoryRouter initialEntries={['/settings']}>
          <KeysSettings view="tools" />
        </MemoryRouter>
      )

      expect(await screen.findByText('WIDGET')).toBeTruthy()
      await waitFor(() => expect(getEnvVars).toHaveBeenCalledWith('profile-b'))

      // Open the field and type a value without saving it.
      fireEvent.focus(container.querySelector('input[readonly]') as HTMLInputElement)
      fireEvent.change(container.querySelector('input[type="password"]') as HTMLInputElement, {
        target: { value: 'typed-secret' }
      })

      expect(screen.getByDisplayValue('typed-secret')).toBeTruthy()

      // Re-target Settings at another profile. This is where the leak
      // manifested: the draft stayed live, so the (still-rendered) Save would
      // dispatch it through setEnvVar against the NEW target.
      await act(async () => {
        $settingsScopeOverride.set('profile-c')
      })
      await waitFor(() => expect(getEnvVars).toHaveBeenCalledWith('profile-c'))

      // The draft belonged to the previous target: it is gone, and so is the
      // Save control that would have dispatched it — no path is left that can
      // write the stale value into the profile now being targeted.
      expect(screen.queryByDisplayValue('typed-secret')).toBeNull()
      expect(screen.queryByRole('button', { name: 'Save' })).toBeNull()
    } finally {
      cleanup()
      $settingsScopeOverride.set(null)
    }
  })
})
