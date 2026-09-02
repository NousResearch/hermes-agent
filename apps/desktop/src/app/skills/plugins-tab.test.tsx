import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $agentPluginScope, $agentPlugins, $agentPluginsStatus } from '@/store/agent-plugins'
import { $pluginInstallRequest, closePluginInstallRequest } from '@/store/plugin-install-request'
import { $connection } from '@/store/session'

import { PluginsTab } from './plugins-tab'

const requestGateway = vi.fn(async () => ({ plugins: [] }))
const { activeGatewayConnectionId, requestGatewayForAgent } = vi.hoisted(() => ({
  activeGatewayConnectionId: vi.fn<() => null | string>(() => null),
  requestGatewayForAgent: vi.fn(async () => ({ marketplaces: [], plugins: [] }))
}))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))
vi.mock('@/store/gateway', () => ({ activeGatewayConnectionId, requestGatewayForAgent }))

describe('PluginsTab', () => {
  beforeEach(() => {
    $agentPlugins.set([])
    $agentPluginScope.set('local::default')
    $agentPluginsStatus.set('ready')
    closePluginInstallRequest()
    requestGateway.mockClear()
    requestGatewayForAgent.mockClear()
    activeGatewayConnectionId.mockReturnValue(null)
    $connection.set(null)
  })

  afterEach(cleanup)

  it('lists the scoped profile agent plugins with toggles', () => {
    $agentPluginScope.set('local::workbot')
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
    expect(screen.getByRole('switch', { name: 'demo-plugin' }).getAttribute('aria-checked')).toBe('true')
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

    render(<PluginsTab profile={null} />)

    expect(screen.queryByText('fal')).toBeNull()
    expect(screen.getByText(/No agent plugins installed/)).toBeTruthy()
  })

  it('loads the plugin list scoped to the selected profile', () => {
    render(<PluginsTab profile="workbot" />)

    expect(requestGateway).toHaveBeenCalledWith(
      'plugins.manage',
      expect.objectContaining({ action: 'list', profile: 'workbot' })
    )
  })

  it('routes an explicitly selected connection to its owning backend', async () => {
    render(<PluginsTab profile={{ connectionId: 'remote-a', profile: 'workbot' }} />)

    await waitFor(() =>
      expect(requestGatewayForAgent).toHaveBeenCalledWith(
        'remote-a',
        'workbot',
        'plugins.manage',
        expect.objectContaining({ action: 'list', profile: 'workbot' })
      )
    )
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('invalidates same-profile rows when the ambient backend changes', async () => {
    $connection.set({ connectionId: 'remote-a', mode: 'remote', profile: 'same' } as never)
    $agentPluginScope.set('remote-a::same')
    $agentPlugins.set([
      {
        description: '',
        key: 'from-a',
        name: 'from-a',
        source: 'git',
        status: 'enabled',
        version: ''
      }
    ])
    render(<PluginsTab profile="same" />)
    expect(screen.getByText('from-a')).toBeTruthy()

    act(() => $connection.set({ connectionId: 'remote-b', mode: 'remote', profile: 'same' } as never))

    await waitFor(() => expect(screen.queryByText('from-a')).toBeNull())
    await waitFor(() => expect($agentPluginScope.get()).toBe('remote-b::same'))
  })

  it('isolates an explicit local profile from the ambient remote backend', async () => {
    activeGatewayConnectionId.mockReturnValue('remote-a')
    render(<PluginsTab profile={{ connectionId: 'local', profile: 'same' }} />)

    await waitFor(() =>
      expect(requestGatewayForAgent).toHaveBeenCalledWith(
        'local',
        'same',
        'plugins.manage',
        expect.objectContaining({ action: 'list', profile: 'same' })
      )
    )
    expect($agentPluginScope.get()).toBe('local::same')
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
      expect(request?.connectionId).toBeNull()
      expect(request?.repo).toBe('https://github.com/example/weather-plugin')
      expect(request?.profile).toBe('workbot')
      expect(request?.scopeKey).toBe('local::workbot')
      expect(request?.sha).toBe('a'.repeat(40))
    })
  })

  it('captures the selected backend and cancels confirmation when that scope changes', async () => {
    const { rerender } = render(<PluginsTab profile={{ connectionId: 'remote-a', profile: 'workbot' }} />)

    window.dispatchEvent(
      new MessageEvent('message', {
        data: {
          name: 'weather-plugin',
          repo: 'https://github.com/example/weather-plugin',
          type: 'hermes-plugin-pick'
        },
        origin: 'https://hermes-agent.nousresearch.com'
      })
    )

    await waitFor(() => {
      expect($pluginInstallRequest.get()?.connectionId).toBe('remote-a')
      expect($pluginInstallRequest.get()?.scopeKey).toBe('remote-a::workbot')
    })

    rerender(<PluginsTab profile={{ connectionId: 'remote-b', profile: 'workbot' }} />)

    await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
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

    render(<PluginsTab profile={null} />)

    screen.getByRole('switch', { name: 'Legacy plugin' }).click()

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

    render(<PluginsTab profile={null} />)

    const toggle = screen.getByRole('switch', { name: 'Legacy plugin' })

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
    $agentPluginScope.set('local::default')
    $agentPluginsStatus.set('ready')
    closePluginInstallRequest()
    requestGateway.mockClear()
    requestGatewayForAgent.mockClear()
    activeGatewayConnectionId.mockReturnValue(null)
    $connection.set(null)
  })

  afterEach(cleanup)

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

  it('shows the private marketplace subtree target on its Update chip', () => {
    $agentPlugins.set([
      {
        current_tree_sha: 'c'.repeat(40),
        description: '',
        key: 'demo-private',
        marketplace_id: 'private-source',
        marketplace_name: 'Private Market',
        marketplace_plugin_name: 'demo-private',
        name: 'demo-private',
        source: 'git',
        status: 'enabled',
        update_available: true,
        version: '1.0.0'
      }
    ])

    render(<PluginsTab profile={null} />)

    expect(screen.getByRole('button', { name: `Update to ${'c'.repeat(8)}` })).toBeTruthy()
  })

  it('re-pins through plugins.manage update when the chip is clicked', async () => {
    $agentPluginScope.set('local::workbot')
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
        expect.objectContaining({ action: 'update', key: 'demo-weather', profile: 'workbot' })
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
