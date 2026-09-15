import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useNavigate } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'

import type { KeysView } from './keys-settings'
import { envVar } from './test-utils'

const getEnvVars = vi.fn()
const setEnvVar = vi.fn()

stubResizeObserver()

vi.mock('@/hermes', () => ({
  deleteEnvVar: vi.fn(),
  getEnvVars: (profile?: null | string) => getEnvVars(profile),
  revealEnvVar: vi.fn(),
  setApiRequestProfile: () => undefined,
  setEnvVar: (key: string, value: string, profile?: string) => setEnvVar(key, value, profile)
}))

beforeEach(() => {
  getEnvVars.mockResolvedValue({})
  Object.defineProperty(Element.prototype, 'scrollIntoView', {
    configurable: true,
    value: vi.fn()
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderKeysSettings(view: KeysView, route = '/settings') {
  const { KeysSettings } = await import('./keys-settings')

  await act(async () => {
    render(
      <MemoryRouter initialEntries={[route]}>
        <KeysSettings view={view} />
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
  it('fetches env vars for the active profile (undefined, never null) when unscoped', async () => {
    // #90549 class: getEnvVars(null) targets the primary profile's env store,
    // so a non-default profile's Keys page would read (and edit) the wrong
    // profile. Unscoped must send undefined so the active profile applies.
    await renderKeysSettings('tools')

    await waitFor(() => expect(getEnvVars).toHaveBeenCalledWith(undefined))
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

  it('lists custom rows and excludes every catalog category', async () => {
    getEnvVars.mockResolvedValue({
      MY_SERVICE_API_KEY: envVar('custom'),
      BRAVE_SEARCH_API_KEY: envVar('tool'),
      API_SERVER_TOKEN: envVar('setting'),
      GATEWAY_PROXY: envVar('messaging')
    })

    await renderKeysSettings('custom')

    expect(screen.getByText('MY SERVICE')).toBeTruthy()
    expect(screen.queryByText('BRAVE SEARCH')).toBeNull()
    expect(screen.queryByText('API SERVER')).toBeNull()
    expect(screen.queryByText('GATEWAY PROXY')).toBeNull()
  })

  it('rejects an invalid custom key name and keeps Add disabled', async () => {
    await renderKeysSettings('custom')

    // The label names the field and IS the accessible name — the action text
    // ("Add a custom key") sits in the caption above, so visible text and
    // accessible name agree instead of an aria-label overriding the label
    // (WCAG 2.5.3 Label in Name).
    const input = screen.getByLabelText('Variable name')
    const add = screen.getByRole('button', { name: 'Add' })

    expect(add.hasAttribute('disabled')).toBe(true)
    expect(screen.queryByText(/start with a letter or underscore/)).toBeNull()

    fireEvent.change(input, { target: { value: '2bad name!' } })

    expect(screen.getByText(/start with a letter or underscore/)).toBeTruthy()
    expect(add.hasAttribute('disabled')).toBe(true)
    // The reason is wired to the control, not merely painted beside it.
    expect(input.getAttribute('aria-invalid')).toBe('true')
    expect(input.getAttribute('aria-describedby')).toBe('settings-custom-key-feedback')
  })

  it('explains why Add is disabled when the variable name already exists', async () => {
    getEnvVars.mockResolvedValue({ WEATHER_API_KEY: envVar('custom', { is_set: true }) })
    await renderKeysSettings('custom')

    const input = screen.getByLabelText('Variable name')
    const add = screen.getByRole('button', { name: 'Add' })

    fireEvent.change(input, { target: { value: 'weather_api_key' } })

    expect(add.hasAttribute('disabled')).toBe(true)
    expect(screen.getByText('A variable with this name already exists.')).toBeTruthy()
    expect(input.getAttribute('aria-invalid')).toBe('true')
    expect(input.getAttribute('aria-describedby')).toBe('settings-custom-key-feedback')
  })

  it('adds a valid custom key: normalises case, registers the row, and opens it for editing', async () => {
    await renderKeysSettings('custom')

    fireEvent.change(screen.getByLabelText('Variable name'), { target: { value: 'my_service' } })

    const add = screen.getByRole('button', { name: 'Add' })

    expect(add.hasAttribute('disabled')).toBe(false)

    fireEvent.click(add)

    // The new row appears in the list, and the row's value editor is OPEN for
    // the new key: the editable (not read-only) paste-value input is mounted
    // and focused. Asserting on the row wrapper id would prove nothing — it
    // renders whether or not the editor is open.
    expect(await screen.findByText('MY SERVICE')).toBeTruthy()

    const editor = (await screen.findByPlaceholderText('Paste MY SERVICE key')) as HTMLInputElement

    expect(editor.readOnly).toBe(false)
    expect(globalThis.document.activeElement).toBe(editor)
    // The form resets.
    expect((screen.getByLabelText('Variable name') as HTMLInputElement).value).toBe('')
    // The pending row is local-only until a value is saved through the normal
    // credential save path — nothing hits the wire just for adding a name.
    expect(getEnvVars).toHaveBeenCalledTimes(1)
  })

  it('persists a new custom key through setEnvVar on save', async () => {
    setEnvVar.mockResolvedValue({ ok: true })
    await renderKeysSettings('custom')

    fireEvent.change(screen.getByLabelText('Variable name'), { target: { value: 'my_service' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add' }))

    // Type the value into the freshly-opened editor and save it.
    const editor = (await screen.findByPlaceholderText('Paste MY SERVICE key')) as HTMLInputElement

    fireEvent.change(editor, { target: { value: 'secret-value' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    // The central claim: the value reaches the real setEnvVar seam with the
    // unscoped (undefined = active profile) request-scope profile.
    await waitFor(() => expect(setEnvVar).toHaveBeenCalledWith('MY_SERVICE', 'secret-value', undefined))
  })

  it('expands and highlights a deep-linked credential card', async () => {
    getEnvVars.mockResolvedValue({
      BRAVE_SEARCH_API_KEY: envVar('tool', { description: 'Search the web with Brave.' }),
      FIRECRAWL_API_KEY: envVar('tool', { description: 'Crawl and extract websites.' })
    })

    const { KeysSettings } = await import('./keys-settings')

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
    const { $settingsScopeOverride } = await import('@/store/settings-scope')

    $settingsScopeOverride.set('profile-b')
    getEnvVars.mockResolvedValue({
      WIDGET_API_KEY: envVar('tool', { description: 'Widget key.', is_set: true, redacted_value: '••••••' })
    })

    try {
      const { KeysSettings } = await import('./keys-settings')

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
