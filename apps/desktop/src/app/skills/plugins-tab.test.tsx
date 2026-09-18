import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { type ComponentProps, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $pluginRecords } from '@/contrib/plugins-store'
import { queryClient } from '@/lib/query-client'
import { $agentPlugins, $agentPluginsStatus } from '@/store/agent-plugins'
import { $pluginInstallRequest, closePluginInstallRequest } from '@/store/plugin-install-request'

import { PageSearchShell } from '../page-search-shell'

import { CapabilityTabs } from './capability-tabs'
import { parseCatalog } from './catalog-data'
import { PluginActions, PluginsTab } from './plugins-tab'
import { $catalogCardView } from './store'

const requestGateway = vi.fn(async () => ({ plugins: $agentPlugins.get() }))

// SkillsView owns navigation and search; exercise that controlled contract
// with the same primitives instead of giving PluginsTab private controls.
function PluginsHarness({
  view: initialView = 'installed',
  query: initialQuery = '',
  ...props
}: ComponentProps<typeof PluginsTab>) {
  const [view, setView] = useState(initialView)
  const [query, setQuery] = useState(initialQuery)

  return (
    <PageSearchShell onSearchChange={setQuery} searchPlaceholder="Search plugins" searchValue={query}>
      <CapabilityTabs actions={<PluginActions profile={props.profile} />} onChange={setView} value={view} />
      <PluginsTab {...props} onQueryChange={setQuery} query={query} view={view} />
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
  fireEvent.click(screen.getByRole('button', { name: 'Browse' }))
  fireEvent.click(await screen.findByRole('button', { name: text => text.startsWith(name) }))
  expect(screen.getByRole('heading', { name })).toBeTruthy()
}

afterEach(() => {
  cleanup()
  queryClient.clear()
  vi.unstubAllGlobals()
})

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

describe('PluginsTab', () => {
  beforeEach(() => {
    $pluginRecords.set({})
    $agentPlugins.set([])
    $agentPluginsStatus.set('ready')
    $catalogCardView.set(false)
    closePluginInstallRequest()
    requestGateway.mockReset()
    requestGateway.mockImplementation(async () => ({ plugins: $agentPlugins.get() }))
  })

  it('lists the scoped profile agent plugins with toggles', () => {
    $agentPlugins.set([
      {
        description: 'A test plugin',
        key: 'demo-plugin',
        name: 'demo-plugin',
        source: 'git',
        status: 'enabled',
        version: '1.0.0'
      }
    ])

    renderPlugins({ profile: 'workbot' })

    expect(screen.getByRole('button', { name: /^demo-plugin/, pressed: true })).toBeTruthy()
    const detail = within(screen.getByRole('row', { name: /^demo-plugin/ }))
    expect(detail.getByRole('switch', { name: 'Agent: demo-plugin' }).getAttribute('aria-checked')).toBe('true')
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

    expect(screen.getAllByRole('button', { name: /^Media Studio/ })).toHaveLength(1)
    expect(screen.getAllByRole('row')).toHaveLength(1)
    const detail = within(screen.getByRole('row', { name: /^Media Studio/ }))
    expect(detail.getByText('Agent + Desktop')).toBeTruthy()
    expect(detail.getByRole('switch', { name: 'Desktop: Media Studio' }).getAttribute('aria-checked')).toBe('true')
    expect(detail.getByRole('switch', { name: 'Agent: Media Studio' }).getAttribute('aria-checked')).toBe('false')
    expect(detail.getByRole('cell', { name: 'Agent in workbot' })).toBeTruthy()
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

  it('opens the dual-target install modal for the selected catalog entry in the scoped profile', async () => {
    seedCatalog([
      { ...weatherEntry, name: 'other-plugin', repo: 'https://github.com/example/other-plugin' },
      weatherEntry
    ])
    renderPlugins({ profile: 'workbot' })

    await selectCatalogEntry(weatherEntry.name)
    expect($pluginInstallRequest.get()).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))

    expect($pluginInstallRequest.get()).toMatchObject({
      catalogName: weatherEntry.name,
      repo: weatherEntry.repo,
      profile: 'workbot',
      sha: weatherEntry.sha
    })
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
    requestGateway.mockResolvedValueOnce({
      ok: true,
      plugin: { key: 'image_gen/legacy', name: 'Legacy plugin', status: 'enabled' }
    } as never)

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

  it('appends the selected subdir for multi-plugin repos without changing its catalog pin', async () => {
    const entry = {
      ...weatherEntry,
      name: 'nested-plugin',
      repo: 'https://github.com/example/plugins-monorepo',
      subdir: 'packages/nested-plugin'
    }
    seedCatalog([entry])
    renderPlugins({ profile: null })

    await selectCatalogEntry(entry.name)
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))

    expect($pluginInstallRequest.get()).toMatchObject({
      catalogName: entry.name,
      repo: `${entry.repo}#${entry.subdir}`,
      profile: null,
      sha: entry.sha
    })
  })
})

describe('PluginsTab catalog UX', () => {
  beforeEach(() => {
    $agentPlugins.set([])
    $agentPluginsStatus.set('ready')
    $catalogCardView.set(false)
    closePluginInstallRequest()
    requestGateway.mockReset()
    requestGateway.mockImplementation(async () => ({ plugins: $agentPlugins.get() }))
    $pluginRecords.set({})
  })

  it('fetches only on Browse, with no extra requests for selection, search, or tab bounce', async () => {
    const fetchCatalog = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [weatherEntry, { ...weatherEntry, name: 'garden-plugin', category: 'garden', description: 'Garden planning' }]
    })
    vi.stubGlobal('fetch', fetchCatalog)
    await act(async () => { renderPlugins({ profile: null }) })

    expect(fetchCatalog).not.toHaveBeenCalled()
    expect(screen.queryByRole('heading', { name: weatherEntry.name })).toBeNull()
    await selectCatalogEntry('garden-plugin')
    const search = screen.getByRole<HTMLInputElement>('textbox', { name: 'Search plugins' })
    fireEvent.change(search, { target: { value: 'weather' } })
    expect(await screen.findByRole('heading', { name: weatherEntry.name })).toBeTruthy()
    expect(screen.queryByRole('button', { name: /^garden-plugin/ })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Installed', pressed: false }))
    expect(screen.queryByRole('heading', { name: weatherEntry.name })).toBeNull()
    // The parent retains the shared search across navigation. Clearing from
    // the empty state must propagate through onQueryChange, not local state.
    expect(search.value).toBe('weather')
    fireEvent.click(screen.getByRole('button', { name: 'Clear filters' }))
    expect(search.value).toBe('')
    await selectCatalogEntry('garden-plugin')

    expect(fetchCatalog).toHaveBeenCalledTimes(1)
    expect(fetchCatalog.mock.calls[0][0]).toMatch(/\/docs\/api\/plugins\.json$/)
    expect(fetchCatalog.mock.calls[0][1]).toMatchObject({ credentials: 'omit' })
    expect($pluginInstallRequest.get()).toBeNull()
  })

  it('offers an explicit retry after a catalog failure rather than refetching on tab bounce', async () => {
    const fetchCatalog = vi.fn()
      .mockResolvedValueOnce({ ok: false, status: 503 })
      .mockResolvedValue({ ok: true, json: async () => [weatherEntry] })
    vi.stubGlobal('fetch', fetchCatalog)
    await act(async () => { renderPlugins({ profile: null }) })
    fireEvent.click(screen.getByRole('button', { name: 'Browse' }))

    expect(await screen.findByText('Catalog HTTP 503')).toBeTruthy()
    expect(fetchCatalog).toHaveBeenCalledTimes(1)
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /^Installed/ }))
    })
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Browse' }))
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

  it('disables installation of a catalog entry that is already installed and current', async () => {
    $agentPlugins.set([
      {
        catalog_name: 'demo-weather',
        description: '',
        installed_sha: 'a'.repeat(40),
        key: 'demo-weather',
        name: 'demo-weather',
        source: 'git',
        status: 'enabled',
        update_available: false,
        version: '1.0.0'
      }
    ])

    seedCatalog([{ ...weatherEntry, name: 'demo-weather' }])
    await act(async () => { renderPlugins({ profile: null }) })
    await selectCatalogEntry('demo-weather')

    const installed = within(screen.getByRole('main')).getByRole<HTMLButtonElement>('button', { name: 'Installed' })
    expect(installed.disabled).toBe(true)
    fireEvent.click(installed)
    expect($pluginInstallRequest.get()).toBeNull()
  })

  it('offers catalog installation for an installed entry when an update is available', async () => {
    $agentPlugins.set([
      {
        catalog_name: 'demo-weather',
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

    const entry = { ...weatherEntry, name: 'demo-weather', sha: 'b'.repeat(40) }
    seedCatalog([entry])
    await act(async () => { renderPlugins({ profile: null }) })
    await selectCatalogEntry(entry.name)
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))

    expect($pluginInstallRequest.get()).toMatchObject({
      catalogName: entry.name,
      repo: entry.repo,
      profile: null,
      sha: entry.sha
    })
  })
})
