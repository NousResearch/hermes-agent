import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $pluginRecords } from '@/contrib/plugins-store'
import { $agentPlugins, $agentPluginsStatus } from '@/store/agent-plugins'
import { $paneHeightOverride, setPaneHeightOverride } from '@/store/panes'
import { $pluginInstallRequest, closePluginInstallRequest } from '@/store/plugin-install-request'

import { PluginsTab } from './plugins-tab'

const requestGateway = vi.fn(async () => ({ plugins: [] }))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

describe('PluginsTab', () => {
  beforeEach(() => {
    $pluginRecords.set({})
    $agentPlugins.set([])
    $agentPluginsStatus.set('ready')
    closePluginInstallRequest()
    requestGateway.mockClear()
  })

  afterEach(cleanup)

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

    render(<PluginsTab profile="workbot" />)

    expect(screen.getByText('demo-plugin')).toBeTruthy()
    expect(screen.getByRole('switch', { name: 'Agent: demo-plugin' }).getAttribute('aria-checked')).toBe('true')
  })

  it('shows curated lifecycle bundles while hiding every other bundled plugin', () => {
    $agentPlugins.set([
      {
        description: '',
        key: 'disk-cleanup',
        name: 'disk-cleanup',
        source: 'bundled',
        status: 'enabled',
        version: ''
      },
      {
        description: '',
        key: 'security-guidance',
        name: 'security-guidance',
        source: 'bundled',
        status: 'enabled',
        version: ''
      },
      {
        description: '',
        key: 'image_gen/fal',
        name: 'fal',
        source: 'bundled',
        status: 'enabled',
        version: ''
      },
      {
        description: '',
        key: 'future-bundle',
        name: 'future-bundle',
        source: 'bundled',
        status: 'enabled',
        version: ''
      },
      {
        description: '',
        key: 'model-providers/legacy',
        name: 'legacy-provider',
        source: 'user',
        status: 'enabled',
        version: ''
      }
    ])

    render(<PluginsTab profile={null} />)

    expect(screen.getByText('disk-cleanup')).toBeTruthy()
    expect(screen.getByText('security-guidance')).toBeTruthy()
    expect(screen.queryByText('fal')).toBeNull()
    expect(screen.queryByText('future-bundle')).toBeNull()
    expect(screen.queryByText('legacy-provider')).toBeNull()
    expect(screen.getByText('Changes take effect after the backend restarts.')).toBeTruthy()
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

    render(<PluginsTab profile="workbot" scopeLabel="workbot" />)

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

    render(<PluginsTab profile="workbot" scopeLabel="workbot" />)

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

    render(<PluginsTab profile="workbot" scopeLabel="workbot" />)

    expect((screen.getByRole('button', { name: 'Install here' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('loads the plugin list scoped to the selected profile', () => {
    render(<PluginsTab profile="workbot" />)

    expect(requestGateway).toHaveBeenCalledWith(
      'plugins.manage',
      expect.objectContaining({ action: 'list', profile: 'workbot' })
    )
  })

  it('opens the dual-target install modal from a catalog pick message', async () => {
    render(<PluginsTab profile="workbot" />)

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
      expect(request?.profile).toBe('workbot')
      expect(request?.sha).toBe('a'.repeat(40))
    })
  })

  it('ignores pick messages from foreign origins', () => {
    render(<PluginsTab profile={null} />)

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
    $agentPlugins.set([
      {
        description: '',
        key: 'legacy',
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

    render(<PluginsTab profile={null} />)

    screen.getByRole('switch', { name: 'Agent: Legacy plugin' }).click()

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({ action: 'toggle', key: 'legacy', enable: true })
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

    render(<PluginsTab profile={null} />)

    const toggle = screen.getByRole('switch', { name: 'Agent: Legacy plugin' })

    expect(toggle.hasAttribute('disabled') || toggle.getAttribute('aria-disabled') === 'true').toBe(true)

    toggle.click()

    expect(requestGateway).not.toHaveBeenCalledWith('plugins.manage', expect.objectContaining({ action: 'toggle' }))
  })

  it('appends the subdir fragment for multi-plugin repos', async () => {
    render(<PluginsTab profile={null} />)

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
    $agentPlugins.set([])
    $agentPluginsStatus.set('ready')
    closePluginInstallRequest()
    requestGateway.mockClear()
    setPaneHeightOverride('capabilities-plugin-catalog', undefined)
  })

  afterEach(cleanup)

  it('grows the catalog when its top-edge sash is dragged up, and resets on double-click', () => {
    // jsdom has no layout: give the Capabilities column a real height so the
    // "never crush the lists above" clamp has something to clamp against.
    const clientHeight = vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(900)
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: 1000 })
    render(<PluginsTab profile={null} />)
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

    render(<PluginsTab profile={null} />)

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

    render(<PluginsTab profile="workbot" />)

    screen.getByRole('button', { name: `Update to ${'b'.repeat(8)}` }).click()

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({ action: 'update', name: 'demo-weather', profile: 'workbot' })
      )
    )
  })

  it('refuses a catalog pick that is already installed and current', async () => {
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

    render(<PluginsTab profile={null} />)

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

    render(<PluginsTab profile={null} />)

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
