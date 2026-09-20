import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import type { ReactElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $pluginRecords } from '@/contrib/plugins-store'
import { $agentPlugins, $agentPluginsStatus, type AgentPluginRow } from '@/store/agent-plugins'
import type * as GatewayStore from '@/store/gateway'
import { $paneHeightOverride, setPaneHeightOverride } from '@/store/panes'
import { $pluginInstallRequest, closePluginInstallRequest } from '@/store/plugin-install-request'
import { $connection } from '@/store/session'

import { PluginsTab } from './plugins-tab'

// The store re-homes its cache per owner, so seeded rows must come back from the backend read.
let backendPlugins: AgentPluginRow[] = []
const requestGateway = vi.fn(async (_method?: string, _params?: unknown) => ({ plugins: backendPlugins }))

function seedAgentPlugins(rows: AgentPluginRow[]) {
  backendPlugins = rows
}

async function renderTab(ui: ReactElement) {
  let result!: ReturnType<typeof render>
  await act(async () => {
    result = render(ui)
  })

  return result
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

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  requestGatewayForAgent: (_connection: unknown, _profile: unknown, method: string, params: unknown) =>
    requestGateway(method, params)
}))

describe('PluginsTab', () => {
  beforeEach(() => {
    $pluginRecords.set({})
    backendPlugins = []
    $agentPlugins.set([])
    $agentPluginsStatus.set('idle')
    closePluginInstallRequest()
    requestGateway.mockReset()
    requestGateway.mockImplementation(async () => ({ plugins: backendPlugins }))
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
  })

  it('lists the scoped profile agent plugins with toggles', async () => {
    seedAgentPlugins([
      {
        description: 'A test plugin',
        key: 'demo-plugin',
        name: 'demo-plugin',
        source: 'git',
        status: 'enabled',
        version: '1.0.0'
      }
    ])

    await renderTab(<PluginsTab profile="workbot" />)

    expect(screen.getByText('demo-plugin')).toBeTruthy()
    expect(screen.getByRole('switch', { name: 'Agent: demo-plugin' }).getAttribute('aria-checked')).toBe('true')
  })

  it('hides bundled plugins (managed from their own surfaces)', async () => {
    seedAgentPlugins([
      {
        description: '',
        key: 'image_gen/fal',
        name: 'fal',
        source: 'bundled',
        status: 'enabled',
        version: ''
      }
    ])

    await renderTab(<PluginsTab profile={null} />)

    expect(screen.queryByText('fal')).toBeNull()
    expect(screen.getByText(/No plugins yet/)).toBeTruthy()
  })

  // A desktop half can only be copied out of a backend that runs on THIS
  // machine; against a remote one the reconcile is a structural no-op, so the
  // row must say so instead of pending forever (#114079).
  it('marks a remote-backend desktop half unavailable instead of forever copying', async () => {
    $connection.set({ ...connectionFixture, mode: 'remote' })
    seedAgentPlugins([
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

    await renderTab(<PluginsTab profile={null} />)

    const detail = within(screen.getByRole('row', { name: /^nous-prices/ }))
    expect(detail.getByText('unavailable (remote backend)')).toBeTruthy()
    expect(detail.queryByText('copying…')).toBeNull()
  })

  it('keeps the pending desktop-half state on a local backend', async () => {
    seedAgentPlugins([
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

    await renderTab(<PluginsTab profile={null} />)

    const detail = within(screen.getByRole('row', { name: /^nous-prices/ }))
    expect(detail.getByText('copying…')).toBeTruthy()
    expect(detail.queryByText('unavailable (remote backend)')).toBeNull()
  })

  it('renders a unified package as ONE row with a Desktop switch and an Agent switch', async () => {
    $pluginRecords.set({
      media: { id: 'media', name: 'Media Studio', kind: 'disk', status: 'loaded', packageName: 'hermes-media-studio' }
    })
    seedAgentPlugins([
      {
        description: '',
        key: 'hermes-media-studio',
        name: 'hermes-media-studio',
        source: 'git',
        status: 'disabled',
        version: '1'
      }
    ])

    await renderTab(<PluginsTab profile="workbot" scopeLabel="workbot" />)

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

    await renderTab(<PluginsTab profile={{ connectionId: 'homelab', profile: 'workbot' }} scopeLabel="workbot" />)

    expect(screen.queryByRole('switch', { name: /^Agent:/ })).toBeNull()
    screen.getByRole('button', { name: 'Install here' }).click()
    // Pre-filled from the package marker: repo + pinned sha, agent half only, pinned to the tab's owner.
    await waitFor(() => {
      expect($pluginInstallRequest.get()).toMatchObject({
        legacyHint: 'agent',
        target: { connectionId: 'homelab', profile: 'workbot' },
        repo: 'https://github.com/NousResearch/hermes-media-studio.git',
        sha: 'abc'
      })
    })
  })

  it('disables "Install here" when the package has no known origin (hand-copied folder)', async () => {
    $pluginRecords.set({
      media: { id: 'media', name: 'Media Studio', kind: 'disk', status: 'loaded', packageName: 'hermes-media-studio' }
    })

    await renderTab(<PluginsTab profile="workbot" scopeLabel="workbot" />)

    expect((screen.getByRole('button', { name: 'Install here' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('loads the plugin list scoped to the selected profile', async () => {
    await renderTab(<PluginsTab profile="workbot" />)

    expect(requestGateway).toHaveBeenCalledWith(
      'plugins.manage',
      expect.objectContaining({ action: 'list', profile: 'workbot' })
    )
  })

  it('opens the dual-target install modal from a catalog pick message', async () => {
    await renderTab(<PluginsTab profile="workbot" />)

    window.dispatchEvent(
      new MessageEvent('message', {
        data: {
          name: 'weather-plugin',
          repo: 'https://github.com/example/weather-plugin',
          sha: 'a'.repeat(40),
          subdir: '',
          tier: 'community',
          type: 'hermes-plugin-pick'
        },
        origin: 'https://hermes-agent.nousresearch.com'
      })
    )

    await waitFor(() => {
      const request = $pluginInstallRequest.get()

      expect(request).not.toBeNull()
      expect(request?.catalogName).toBe('weather-plugin')
      expect(request?.repo).toBe('https://github.com/example/weather-plugin')
      expect(request?.target.profile).toBe('workbot')
      expect(request?.sha).toBe('a'.repeat(40))
    })
  })

  it('ignores pick messages from foreign origins', async () => {
    await renderTab(<PluginsTab profile={null} />)

    window.dispatchEvent(
      new MessageEvent('message', {
        data: {
          name: 'evil-plugin',
          repo: 'https://github.com/evil/evil-plugin',
          type: 'hermes-plugin-pick'
        },
        origin: 'https://evil.example.com'
      })
    )

    expect($pluginInstallRequest.get()).toBeNull()
  })

  it('toggles by canonical key through plugins.manage', async () => {
    seedAgentPlugins([
      {
        description: '',
        key: 'image_gen/legacy',
        name: 'Legacy plugin',
        source: 'user',
        status: 'disabled',
        version: '0.20.0'
      }
    ])
    requestGateway.mockImplementation(async (_method, params) => {
      if ((params as { action: string }).action === 'toggle') {
        backendPlugins = backendPlugins.map(row => ({ ...row, status: 'enabled' }))

        return { ok: true, plugin: backendPlugins[0] } as never
      }

      return { plugins: backendPlugins }
    })

    await renderTab(<PluginsTab profile={null} />)

    screen.getByRole('switch', { name: 'Agent: Legacy plugin' }).click()

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({ action: 'toggle', key: 'image_gen/legacy', enable: true })
      )
    )
  })

  it('renders keyless rows read-only (no name-addressed toggle RPC)', async () => {
    // Name-addressed toggles flip every same-named plugin across category
    // dirs — pre-contract-v6 rows must never reach the RPC.
    seedAgentPlugins([
      {
        description: 'Returned by a pre-key backend',
        name: 'Legacy plugin',
        source: 'user',
        status: 'disabled',
        version: '0.20.0'
      }
    ])

    await renderTab(<PluginsTab profile={null} />)

    const toggle = screen.getByRole('switch', { name: 'Agent: Legacy plugin' })

    expect(toggle.hasAttribute('disabled') || toggle.getAttribute('aria-disabled') === 'true').toBe(true)

    toggle.click()

    expect(requestGateway).not.toHaveBeenCalledWith('plugins.manage', expect.objectContaining({ action: 'toggle' }))
  })

  it('appends the subdir fragment for multi-plugin repos', async () => {
    await renderTab(<PluginsTab profile={null} />)

    window.dispatchEvent(
      new MessageEvent('message', {
        data: {
          name: 'nested-plugin',
          repo: 'https://github.com/example/plugins-monorepo',
          subdir: 'nested-plugin',
          type: 'hermes-plugin-pick'
        },
        origin: 'https://hermes-agent.nousresearch.com'
      })
    )

    await waitFor(() => {
      expect($pluginInstallRequest.get()?.repo).toBe('https://github.com/example/plugins-monorepo#nested-plugin')
    })
  })
})

describe('PluginsTab catalog UX', () => {
  beforeEach(() => {
    backendPlugins = []
    $agentPlugins.set([])
    $agentPluginsStatus.set('idle')
    closePluginInstallRequest()
    requestGateway.mockReset()
    requestGateway.mockImplementation(async () => ({ plugins: backendPlugins }))
    setPaneHeightOverride('capabilities-plugin-catalog', undefined)
  })

  afterEach(cleanup)

  it('grows the catalog when its top-edge sash is dragged up, and resets on double-click', async () => {
    // jsdom has no layout: give the Capabilities column a real height so the
    // "never crush the lists above" clamp has something to clamp against.
    const clientHeight = vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(900)
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: 1000 })
    await renderTab(<PluginsTab profile={null} />)
    const sash = screen.getByTestId('plugin-catalog-sash')

    fireEvent.pointerDown(sash, { button: 0, clientY: 600 })
    fireEvent.pointerMove(window, { clientY: 400 })
    fireEvent.pointerUp(window)

    // Default 380px + 200px of upward drag (clamped only by window/column size).
    expect($paneHeightOverride('capabilities-plugin-catalog').get()).toBe(580)

    fireEvent.doubleClick(sash)
    expect($paneHeightOverride('capabilities-plugin-catalog').get()).toBeUndefined()
    clientHeight.mockRestore()
  })

  it('shows an Update chip when the catalog pin moved past the installed SHA', async () => {
    seedAgentPlugins([
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

    await renderTab(<PluginsTab profile={null} />)

    expect(screen.getByRole('button', { name: `Update to ${'b'.repeat(8)}` })).toBeTruthy()
  })

  it('re-pins through plugins.manage update when the chip is clicked', async () => {
    seedAgentPlugins([
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
    requestGateway.mockImplementation(async (_method, params) =>
      (params as { action: string }).action === 'update'
        ? ({ ok: true, unchanged: false } as never)
        : { plugins: backendPlugins }
    )

    await renderTab(<PluginsTab profile="workbot" />)

    screen.getByRole('button', { name: `Update to ${'b'.repeat(8)}` }).click()

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({ action: 'update', name: 'demo-weather', profile: 'workbot' })
      )
    )
  })

  it('refuses a catalog pick that is already installed and current', async () => {
    seedAgentPlugins([
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

    await renderTab(<PluginsTab profile={null} />)

    window.dispatchEvent(
      new MessageEvent('message', {
        data: {
          name: 'demo-weather',
          repo: 'https://github.com/example/demo-weather',
          type: 'hermes-plugin-pick'
        },
        origin: 'https://hermes-agent.nousresearch.com'
      })
    )

    // The modal must NOT open — the pick is refused with a toast.
    await new Promise(resolve => setTimeout(resolve, 20))
    expect($pluginInstallRequest.get()).toBeNull()
  })

  it('still opens the modal for an installed pick when an update is available', async () => {
    seedAgentPlugins([
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

    await renderTab(<PluginsTab profile={null} />)

    window.dispatchEvent(
      new MessageEvent('message', {
        data: {
          name: 'demo-weather',
          repo: 'https://github.com/example/demo-weather',
          type: 'hermes-plugin-pick'
        },
        origin: 'https://hermes-agent.nousresearch.com'
      })
    )

    await waitFor(() => expect($pluginInstallRequest.get()).not.toBeNull())
  })
})
