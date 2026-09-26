import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { type ComponentProps, useState } from 'react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $pluginRecords } from '@/contrib/plugins-store'
import { queryClient } from '@/lib/query-client'
import { $agentPlugins, $agentPluginsStatus, type AgentPluginRow } from '@/store/agent-plugins'
import { $confirmRequest, settleConfirm } from '@/store/confirm'
import { $pluginInstallRequest, closePluginInstallRequest } from '@/store/plugin-install-request'
import { $connection } from '@/store/session'

import { PageSearchShell } from '../../page-search-shell'
import { parseCatalog } from '../catalog/catalog-data'
import { $catalogCardView } from '../catalog/store'

import { PluginsTab } from './plugins-tab'

const requestGateway = vi.fn(async (_method: string, _params?: Record<string, unknown>): Promise<unknown> => ({
  plugins: $agentPlugins.get()
}))

// The shell owns search; exercise the same controlled composition as CapabilitiesView.
function PluginsHarness({ query: initialQuery, ...props }: ComponentProps<typeof PluginsTab>) {
  const [query, setQuery] = useState(initialQuery ?? '')

  return (
    <PageSearchShell onSearchChange={setQuery} searchHidden searchPlaceholder="Search plugins" searchValue={query}>
      <PluginsTab {...props} onQueryChange={setQuery} query={query} />
    </PageSearchShell>
  )
}

function renderPlugins(props: ComponentProps<typeof PluginsTab>) {
  return render(
    <QueryClientProvider client={queryClient}>
      <PluginsHarness {...props} />
    </QueryClientProvider>
  )
}

const weatherEntry = {
  name: 'weather-plugin',
  repo: 'https://github.com/example/weather-plugin',
  sha: 'a'.repeat(40),
  subdir: '',
  tier: 'community',
  category: 'weather',
  description: 'Local weather forecasts'
}

function seedCatalog(entries = [weatherEntry]) {
  queryClient.setQueryData(['public-catalog', 'plugins'], parseCatalog('plugins', entries))
}

async function selectCatalogEntry(name: string) {
  fireEvent.click((await screen.findAllByRole('button', { name }))[0])
  expect(screen.getAllByRole('heading', { name }).length).toBeGreaterThan(0)
}

const connectionFixture = {
  baseUrl: 'http://localhost',
  isFullscreen: false,
  logs: [],
  nativeOverlayWidth: 0,
  token: '',
  windowButtonPosition: null,
  wsUrl: ''
}

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

const uninstallDiskPlugin = vi.fn(async (_id: string) => ({ ok: true }))
const setEnvVar = vi.fn(async (..._args: unknown[]) => ({}))

vi.mock('@/api/config', () => ({ setEnvVar: (...args: unknown[]) => setEnvVar(...args) }))

vi.mock('@/contrib/runtime-loader', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  uninstallDiskPlugin: (id: string) => uninstallDiskPlugin(id)
}))

beforeEach(() => {
  $pluginRecords.set({})
  $agentPlugins.set([])
  $agentPluginsStatus.set('ready')
  $catalogCardView.set(false)
  seedCatalog([])
  closePluginInstallRequest()
  requestGateway.mockReset()
  requestGateway.mockImplementation(async () => ({ plugins: $agentPlugins.get() }))
  setEnvVar.mockClear()
})

afterEach(() => {
  cleanup()
  settleConfirm(false)
  $connection.set(null)
  queryClient.clear()
  vi.unstubAllGlobals()
})

describe('PluginsTab', () => {
  it('opens a non-first installed deep-link target from Browse and keeps that selection after consuming the link', async () => {
    Element.prototype.scrollIntoView = vi.fn()
    $agentPlugins.set(
      ['alpha', 'omega'].map(name => ({
        key: name,
        name,
        description: `${name} plugin`,
        source: 'git',
        status: 'enabled',
        version: '1'
      }))
    )
    seedCatalog()

    function Location() {
      return <output data-testid="route">{useLocation().search}</output>
    }

    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={['/capabilities?tab=plugins&plugin=omega']}>
          <Location />
          <PluginsHarness profile="workbot" query="nothing-matches" />
        </MemoryRouter>
      </QueryClientProvider>
    )
    await screen.findByRole('switch', { name: 'Agent: omega' })
    await waitFor(() => expect(screen.getByTestId('route').textContent).toBe('?tab=plugins'))
    expect(screen.getByRole<HTMLInputElement>('textbox', { name: 'Search plugins' }).value).toBe('')
    expect(screen.getByRole('switch', { name: 'Agent: omega' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'alpha', pressed: false }))
    expect(screen.getByRole('switch', { name: 'Agent: alpha' })).toBeTruthy()
  })

  it('renders declared server pills and the unavailable sentence under the description', () => {
    $agentPlugins.set([
      {
        description: 'A test plugin',
        key: 'demo-plugin',
        name: 'demo-plugin',
        servers: [
          { name: 'ready-server', sentence: '', state: 'connected' },
          {
            name: 'setup-server',
            sentence: 'Example App is not installed. Install Example App, then try again.',
            state: 'missing_app'
          }
        ],
        source: 'git',
        status: 'enabled',
        version: '1.0.0'
      }
    ])

    renderPlugins({ profile: 'workbot' })

    expect(screen.getByTestId('server-pill-ready-server')).toBeTruthy()
    expect(screen.getByTestId('server-pill-setup-server')).toBeTruthy()
    expect(screen.getByText('Example App is not installed. Install Example App, then try again.')).toBeTruthy()
    expect(screen.getByRole('switch', { name: 'Agent: demo-plugin' }).getAttribute('aria-checked')).toBe('true')
  })

  it('hides bundled plugins (managed from their own surfaces)', () => {
    $agentPlugins.set([
      {
        description: '',
        key: 'image_gen/fal',
        name: 'fal',
        source: 'bundled',
        status: 'enabled',
        version: ''
      }
    ])

    renderPlugins({ profile: null })

    expect(screen.queryByText('fal')).toBeNull()
    expect(screen.queryByRole('row')).toBeNull()
    expect(screen.getByText('No matches')).toBeTruthy()
  })

  // A desktop half can only be copied out of a backend that runs on THIS
  // machine; against a remote one the reconcile is a structural no-op, so the
  // row must say so instead of pending forever (#114079).
  it('marks a remote-backend desktop half unavailable instead of forever copying', () => {
    $connection.set({ ...connectionFixture, mode: 'remote' })
    $agentPlugins.set([
      {
        description: '',
        has_desktop_half: true,
        key: 'nous-prices',
        name: 'nous-prices',
        source: 'catalog',
        status: 'enabled',
        version: '1'
      }
    ])

    renderPlugins({ profile: null })

    const detail = within(screen.getByRole('row', { name: /^nous-prices/ }))
    expect(detail.getByText('unavailable (remote backend)')).toBeTruthy()
    expect(detail.queryByText('copying…')).toBeNull()
  })

  it('keeps the pending desktop-half state on a local backend', () => {
    $agentPlugins.set([
      {
        description: '',
        has_desktop_half: true,
        key: 'nous-prices',
        name: 'nous-prices',
        source: 'catalog',
        status: 'enabled',
        version: '1'
      }
    ])

    renderPlugins({ profile: null })

    const detail = within(screen.getByRole('row', { name: /^nous-prices/ }))
    expect(detail.getByText('copying…')).toBeTruthy()
    expect(detail.queryByText('unavailable (remote backend)')).toBeNull()
  })

  it('renders a unified package as ONE row with a Desktop switch and an Agent switch', () => {
    $pluginRecords.set({
      media: { id: 'media', name: 'Media Studio', kind: 'disk', status: 'loaded', packageName: 'hermes-media-studio' }
    })
    $agentPlugins.set([
      {
        description: '',
        key: 'hermes-media-studio',
        name: 'hermes-media-studio',
        source: 'git',
        status: 'disabled',
        version: '1'
      }
    ])

    renderPlugins({ profile: 'workbot', scopeLabel: 'workbot' })

    expect(screen.getAllByTestId(/^plugin-row-/)).toHaveLength(1)
    expect(screen.getByText('Agent + Desktop')).toBeTruthy()
    expect(screen.getByRole('switch', { name: 'Desktop: Media Studio' }).getAttribute('aria-checked')).toBe('true')
    expect(screen.getByRole('switch', { name: 'Agent: Media Studio' }).getAttribute('aria-checked')).toBe('false')
    expect(screen.getAllByText('Agent in workbot').length).toBeGreaterThan(0)
  })

  it('offers "Install here" for a desktop half whose agent half is not in the selected profile', async () => {
    $pluginRecords.set({
      media: {
        id: 'media',
        name: 'Media Studio',
        kind: 'disk',
        status: 'loaded',
        packageName: 'hermes-media-studio',
        packageOrigin: { repo: 'https://github.com/NousResearch/hermes-media-studio.git', sha: 'abc' }
      }
    })

    renderPlugins({ profile: 'workbot', scopeLabel: 'workbot' })

    expect(screen.queryByRole('switch', { name: /^Agent:/ })).toBeNull()
    screen.getByRole('button', { name: 'Install here' }).click()
    // Pre-filled from the package marker: repo + pinned sha, agent half only.
    await waitFor(() => {
      expect($pluginInstallRequest.get()).toMatchObject({
        legacyHint: 'agent',
        profile: 'workbot',
        repo: 'https://github.com/NousResearch/hermes-media-studio.git',
        sha: 'abc'
      })
    })
  })

  it('disables "Install here" when the package has no known origin (hand-copied folder)', () => {
    $pluginRecords.set({
      media: { id: 'media', name: 'Media Studio', kind: 'disk', status: 'loaded', packageName: 'hermes-media-studio' }
    })

    renderPlugins({ profile: 'workbot', scopeLabel: 'workbot' })

    expect((screen.getByRole('button', { name: 'Install here' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('loads the plugin list scoped to the selected profile', () => {
    renderPlugins({ profile: 'workbot' })

    expect(requestGateway).toHaveBeenCalledWith(
      'plugins.manage',
      expect.objectContaining({ action: 'list', profile: 'workbot' })
    )
  })

  it.each([false, true])('opens the scoped native install dialog from the selected entry (cards=%s)', async cards => {
    $catalogCardView.set(cards)
    seedCatalog([
      { ...weatherEntry, name: 'other-plugin', repo: 'https://github.com/example/other-plugin' },
      weatherEntry
    ])
    renderPlugins({ profile: 'workbot' })

    await selectCatalogEntry(weatherEntry.name)
    expect($pluginInstallRequest.get()).toBeNull()
    fireEvent.click(screen.getByRole('switch', { name: `Add ${weatherEntry.name}` }))

    expect($pluginInstallRequest.get()).toMatchObject({
      catalogName: weatherEntry.name,
      repo: weatherEntry.repo,
      profile: 'workbot',
      sha: weatherEntry.sha
    })
  })

  it('shows installed plugins in list/detail mode and filters them through the parent search', () => {
    $pluginRecords.set({
      clock: { id: 'clock', name: 'Clock', kind: 'disk', status: 'loaded' },
      weather: { id: 'weather', name: 'Weather', kind: 'disk', status: 'loaded' }
    })
    renderPlugins({ profile: null })

    expect(screen.getByRole('button', { name: 'Clock', pressed: true })).toBeTruthy()
    expect(screen.getByRole('switch', { name: 'Desktop: Clock' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Weather', pressed: false }))
    expect(screen.getByRole('switch', { name: 'Desktop: Weather' })).toBeTruthy()
    expect(screen.queryByRole('dialog')).toBeNull()

    fireEvent.change(screen.getByRole('textbox', { name: 'Search plugins' }), { target: { value: 'Clock' } })
    expect(screen.queryByRole('button', { name: 'Weather', pressed: false })).toBeNull()
    expect(screen.getByRole('switch', { name: 'Desktop: Clock' })).toBeTruthy()
  })

  it('toggles by canonical key through plugins.manage', async () => {
    $agentPlugins.set([
      {
        description: '',
        key: 'image_gen/legacy',
        name: 'Legacy plugin',
        source: 'user',
        status: 'disabled',
        version: '0.20.0'
      }
    ])
    requestGateway.mockImplementation(async (_method, params) =>
      params?.action === 'toggle'
        ? { ok: true, plugin: { key: 'image_gen/legacy', name: 'Legacy plugin', status: 'enabled' } }
        : { plugins: $agentPlugins.get() }
    )

    renderPlugins({ profile: null })

    screen.getByRole('switch', { name: 'Agent: Legacy plugin' }).click()

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({ action: 'toggle', key: 'image_gen/legacy', enable: true })
      )
    )
  })

  it('renders keyless rows read-only (no name-addressed toggle RPC)', () => {
    // Name-addressed toggles flip every same-named plugin across category
    // dirs — pre-contract-v6 rows must never reach the RPC.
    $agentPlugins.set([
      {
        description: 'Returned by a pre-key backend',
        name: 'Legacy plugin',
        source: 'user',
        status: 'disabled',
        version: '0.20.0'
      }
    ])

    renderPlugins({ profile: null })

    const toggle = screen.getByRole('switch', { name: 'Agent: Legacy plugin' })

    expect(toggle.hasAttribute('disabled') || toggle.getAttribute('aria-disabled') === 'true').toBe(true)

    toggle.click()

    expect(requestGateway).not.toHaveBeenCalledWith('plugins.manage', expect.objectContaining({ action: 'toggle' }))
  })

  it('appends the selected subdir for multi-plugin repos without changing the catalog pin', async () => {
    const entry = {
      ...weatherEntry,
      name: 'nested-plugin',
      repo: 'https://github.com/example/plugins-monorepo',
      subdir: 'packages/nested-plugin'
    }

    seedCatalog([entry])
    renderPlugins({ profile: null })

    await selectCatalogEntry(entry.name)
    fireEvent.click(screen.getByRole('switch', { name: `Add ${entry.name}` }))

    expect($pluginInstallRequest.get()).toMatchObject({
      catalogName: entry.name,
      repo: `${entry.repo}#${entry.subdir}`,
      profile: null,
      sha: entry.sha
    })
  })
})

describe('PluginsTab catalog UX', () => {
  it('saves declared settings from the installed detail with values and secrets on their scoped routes', async () => {
    const row: AgentPluginRow = {
      name: 'demo-settings',
      key: 'category/demo-settings',
      description: 'Configurable plugin',
      source: 'git',
      status: 'enabled',
      version: '1',
      settings_schema: [
        { key: 'retries', label: 'Retries', type: 'number', description: '', required: true, value: 3 },
        { key: 'api_key', label: 'API key', type: 'secret', description: '', required: true, env: 'DEMO_API_KEY' }
      ]
    }

    $agentPlugins.set([row])
    requestGateway.mockImplementation(async (_method, params) =>
      params?.action === 'settings'
        ? {
            ok: true,
            plugin: {
              ...row,
              settings_schema: row.settings_schema!.map(field =>
                field.key === 'retries' ? { ...field, value: 9 } : field
              )
            }
          }
        : { plugins: $agentPlugins.get() }
    )

    const profile = { profile: 'workbot', connectionId: 'remote-work' }

    await act(async () => {
      renderPlugins({ profile })
    })

    fireEvent.click(screen.getByRole('button', { name: 'Settings: demo-settings' }))
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Retries' }), { target: { value: '9' } })
    fireEvent.change(screen.getByLabelText('API key'), { target: { value: 'test-api-key' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save settings' }))

    await waitFor(() => expect(setEnvVar).toHaveBeenCalledWith('DEMO_API_KEY', 'test-api-key', profile))
    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('plugins.manage', {
        action: 'settings',
        key: row.key,
        values: { retries: 9 },
        profile: 'workbot'
      })
    )
    await waitFor(() => expect(screen.getByLabelText<HTMLInputElement>('API key').value).toBe(''))
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Save settings' }).disabled).toBe(true)
  })

  it('requires widening consent before retrying an update, and leaves a declined update untouched', async () => {
    $agentPlugins.set([
      {
        name: 'demo-weather',
        key: 'demo-weather',
        description: '',
        source: 'git',
        status: 'enabled',
        version: '1',
        catalog_sha: 'b'.repeat(40),
        update_available: true
      }
    ])
    requestGateway.mockImplementation(async (_method, params) => {
      if (params?.action !== 'update') {
        return { plugins: $agentPlugins.get() }
      }

      return params.accept_capabilities
        ? { ok: true, unchanged: false }
        : { consent_required: true, sha: 'b'.repeat(40), delta_lines: ['Tools: weather-alerts'] }
    })
    await act(async () => {
      renderPlugins({ profile: 'workbot' })
    })
    fireEvent.click(screen.getByRole('button', { name: `Update to ${'b'.repeat(8)}` }))
    await waitFor(() => expect($confirmRequest.get()?.description).toContain('Tools: weather-alerts'))
    expect($confirmRequest.get()?.title).toBe('demo-weather asks for more')
    await act(async () => {
      settleConfirm(false)
    })
    expect(requestGateway).not.toHaveBeenCalledWith(
      'plugins.manage',
      expect.objectContaining({ accept_capabilities: true })
    )
    expect(screen.getByRole('button', { name: `Update to ${'b'.repeat(8)}` })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: `Update to ${'b'.repeat(8)}` }))
    await waitFor(() => expect($confirmRequest.get()).not.toBeNull())
    await act(async () => {
      settleConfirm(true)
    })
    expect(requestGateway).toHaveBeenCalledWith('plugins.manage', {
      action: 'update',
      name: 'demo-weather',
      profile: 'workbot',
      accept_capabilities: true
    })
  })

  it('fetches the catalog once, retaining parent search across selection and filters', async () => {
    queryClient.clear()

    const fetchCatalog = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [
        weatherEntry,
        { ...weatherEntry, name: 'garden-plugin', category: 'garden', description: 'Garden planning' }
      ]
    })

    vi.stubGlobal('fetch', fetchCatalog)
    await act(async () => {
      renderPlugins({ profile: null })
    })

    await selectCatalogEntry('garden-plugin')

    const search = screen.getByRole<HTMLInputElement>('textbox', { name: 'Search plugins' })

    fireEvent.change(search, { target: { value: 'weather' } })
    expect(await screen.findByRole('heading', { name: weatherEntry.name })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'garden-plugin' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Installed', pressed: false }))
    expect(screen.queryByRole('heading', { name: weatherEntry.name })).toBeNull()
    expect(search.value).toBe('weather')
    fireEvent.click(screen.getAllByRole('button', { name: 'Clear filters' })[0])
    expect(search.value).toBe('')
    await selectCatalogEntry('garden-plugin')

    expect(fetchCatalog).toHaveBeenCalledTimes(1)
    expect(fetchCatalog.mock.calls[0][0]).toMatch(/\/docs\/api\/plugins\.json$/)
    expect(fetchCatalog.mock.calls[0][1]).toMatchObject({ credentials: 'omit' })
    expect($pluginInstallRequest.get()).toBeNull()
  })

  it('retries a failed catalog only when asked, not on tab bounce', async () => {
    queryClient.clear()

    const fetchCatalog = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 503 })
      .mockResolvedValue({ ok: true, json: async () => [weatherEntry] })

    vi.stubGlobal('fetch', fetchCatalog)
    const view = renderPlugins({ profile: null })

    expect(await screen.findByText('Catalog HTTP 503')).toBeTruthy()
    expect(fetchCatalog).toHaveBeenCalledTimes(1)
    view.unmount()
    await act(async () => {
      renderPlugins({ profile: null })
    })
    expect(fetchCatalog).toHaveBeenCalledTimes(1)
    expect(screen.getByText('Catalog HTTP 503')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))

    expect(await screen.findByRole('heading', { name: weatherEntry.name })).toBeTruthy()
    expect(fetchCatalog).toHaveBeenCalledTimes(2)
  })

  it('shows an Update chip when the catalog pin moved past the installed SHA', () => {
    $agentPlugins.set([
      {
        catalog_name: 'demo-weather',
        catalog_sha: 'b'.repeat(40),
        catalog_tier: 'community',
        description: '',
        installed_sha: 'a'.repeat(40),
        key: 'demo-weather',
        name: 'demo-weather',
        source: 'git',
        status: 'enabled',
        update_available: true,
        version: '1.0.0'
      }
    ])

    renderPlugins({ profile: null })

    expect(screen.getByRole('button', { name: `Update to ${'b'.repeat(8)}` })).toBeTruthy()
  })

  it('re-pins through plugins.manage update when the chip is clicked', async () => {
    $agentPlugins.set([
      {
        catalog_name: 'demo-weather',
        catalog_sha: 'b'.repeat(40),
        catalog_tier: 'community',
        description: '',
        installed_sha: 'a'.repeat(40),
        key: 'demo-weather',
        name: 'demo-weather',
        source: 'git',
        status: 'enabled',
        update_available: true,
        version: '1.0.0'
      }
    ])
    requestGateway.mockResolvedValue({ ok: true, unchanged: false, plugins: [] } as never)

    renderPlugins({ profile: 'workbot' })

    screen.getByRole('button', { name: `Update to ${'b'.repeat(8)}` }).click()

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({ action: 'update', name: 'demo-weather', profile: 'workbot' })
      )
    )
  })

  it('uninstalls through plugins.manage remove only after the confirm dialog is accepted', async () => {
    $agentPlugins.set([
      {
        description: '',
        key: 'demo-weather',
        name: 'demo-weather',
        source: 'git',
        status: 'enabled',
        version: '1.0.0'
      }
    ])
    requestGateway.mockImplementation(async (_method, params) =>
      params?.action === 'remove' ? { ok: true, name: 'demo-weather' } : { plugins: $agentPlugins.get() }
    )

    renderPlugins({ profile: 'workbot' })

    screen.getByRole('button', { name: 'Uninstall: demo-weather' }).click()

    // The click only asks; nothing is deleted until the destructive confirm is answered.
    await waitFor(() => expect($confirmRequest.get()?.title).toContain('demo-weather'))
    expect(screen.getByRole('row', { name: /^demo-weather/ })).toBeTruthy()
    expect(requestGateway).not.toHaveBeenCalledWith('plugins.manage', expect.objectContaining({ action: 'remove' }))

    settleConfirm(true)

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({ action: 'remove', name: 'demo-weather', profile: 'workbot' })
      )
    )
    await waitFor(() => expect(screen.queryByText('demo-weather')).toBeNull())
  })

  it('uninstalls a standalone desktop plugin through Electron only after the confirm dialog is accepted', async () => {
    $pluginRecords.set({
      clock: { id: 'clock', name: 'Clock', kind: 'disk', status: 'loaded', file: '/h/desktop-plugins/clock/plugin.js' }
    })
    uninstallDiskPlugin.mockClear()

    renderPlugins({ profile: null })

    screen.getByRole('button', { name: 'Uninstall: Clock' }).click()

    await waitFor(() => expect($confirmRequest.get()?.title).toContain('Clock'))
    expect(uninstallDiskPlugin).not.toHaveBeenCalled()

    settleConfirm(true)

    await waitFor(() => expect(uninstallDiskPlugin).toHaveBeenCalledWith('clock'))
    // Nothing goes over the gateway: this half lives in this app, not the profile.
    expect(requestGateway).not.toHaveBeenCalledWith('plugins.manage', expect.objectContaining({ action: 'remove' }))
  })

  it('offers no desktop Uninstall for a bundled plugin or a unified package half', () => {
    $pluginRecords.set({
      bots: { id: 'bots', name: 'Bot Mode', kind: 'bundled', status: 'loaded' },
      media: { id: 'media', name: 'Media Studio', kind: 'disk', status: 'loaded', packageName: 'hermes-media-studio' }
    })

    renderPlugins({ profile: null })

    fireEvent.click(screen.getByRole('button', { name: /^Bot Mode/ }))
    expect(screen.queryByRole('button', { name: /^Uninstall:/ })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /^Media Studio/ }))
    expect(screen.queryByRole('button', { name: /^Uninstall:/ })).toBeNull()
  })

  it('offers no Uninstall for a pip-installed (entrypoint) agent plugin', () => {
    $agentPlugins.set([
      { description: '', key: 'demo-tool', name: 'demo-tool', source: 'entrypoint', status: 'enabled', version: '' }
    ])

    renderPlugins({ profile: null })

    expect(screen.getByRole('row', { name: /^demo-tool/ })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Uninstall: demo-tool' })).toBeNull()
  })

  it('toggles an installed catalog card on and off in place instead of reinstalling or uninstalling it', async () => {
    $catalogCardView.set(true)
    $agentPlugins.set([
      {
        catalog_name: 'demo-weather',
        description: '',
        installed_sha: 'a'.repeat(40),
        key: 'demo-weather',
        name: 'demo-weather',
        source: 'git',
        status: 'enabled',
        version: '1.0.0'
      }
    ])
    requestGateway.mockImplementation(async (_method, params) =>
      params?.action === 'toggle'
        ? { ok: true, plugin: { key: 'demo-weather', name: 'demo-weather', status: 'disabled' } }
        : { plugins: $agentPlugins.get() }
    )
    seedCatalog([{ ...weatherEntry, name: 'demo-weather' }])
    await act(async () => {
      renderPlugins({ profile: 'workbot' })
    })

    const card = screen
      .getAllByRole('article')
      .find(article => within(article).queryByRole('button', { name: 'demo-weather' }))!
    const toggle = within(card).getByRole('switch', { name: 'demo-weather' })
    expect(toggle.getAttribute('aria-checked')).toBe('true')
    fireEvent.click(toggle)

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({ action: 'toggle', key: 'demo-weather', enable: false, profile: 'workbot' })
      )
    )
    expect($confirmRequest.get()).toBeNull()
    expect($pluginInstallRequest.get()).toBeNull()
    expect(requestGateway).not.toHaveBeenCalledWith('plugins.manage', expect.objectContaining({ action: 'remove' }))
  })
})
