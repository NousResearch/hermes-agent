import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { requestGateway } = vi.hoisted(() => ({
  requestGateway: vi.fn(async () => ({ plugins: [] }))
}))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

import { $pluginRecords } from '@/contrib/plugins-store'
import { $agentPlugins, $agentPluginsStatus } from '@/store/agent-plugins'
import { $gatewayState } from '@/store/session'

import { PluginsSettings } from './plugins-settings'

const renderSettings = () =>
  render(
    <MemoryRouter>
      <PluginsSettings />
    </MemoryRouter>
  )

beforeEach(() => {
  requestGateway.mockClear()
  $pluginRecords.set({})
  $agentPlugins.set([])
  $agentPluginsStatus.set('ready')
  $gatewayState.set('idle')
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('PluginsSettings', () => {
  it('points agent-plugin management at Capabilities instead of duplicating the list', () => {
    // Agent plugins are profile-scoped and managed in Capabilities → Plugins;
    // Settings keeps desktop plugins only, plus a pointer.
    $agentPlugins.set([
      {
        description: 'Should NOT be listed here anymore',
        key: 'demo-plugin',
        name: 'demo-plugin',
        source: 'git',
        status: 'enabled',
        version: '1.0.0'
      }
    ])

    renderSettings()

    expect(screen.queryByText('demo-plugin')).toBeNull()
    expect(screen.getByText(/managed per profile in Capabilities/)).toBeTruthy()
    expect(screen.getByRole('link', { name: /Capabilities/ }).getAttribute('href')).toContain('/skills?tab=plugins')
  })

  it('flags a unified-root desktop half whose agent half is missing on this backend', () => {
    $pluginRecords.set({
      'pixel-overlay': {
        id: 'pixel-overlay',
        name: 'Pixel Overlay',
        kind: 'disk',
        status: 'loaded',
        file: '/home/user/.hermes/plugins/pixel-overlay/desktop/plugin.js'
      }
    })
    $agentPlugins.set([]) // connected backend has no agent half
    $agentPluginsStatus.set('ready')

    renderSettings()

    expect(screen.getByText('agent half missing here')).toBeTruthy()
  })

  it('does not flag when the agent half exists on the connected backend', () => {
    $pluginRecords.set({
      'pixel-overlay': {
        id: 'pixel-overlay',
        name: 'Pixel Overlay',
        kind: 'disk',
        status: 'loaded',
        file: '/home/user/.hermes/plugins/pixel-overlay/desktop/plugin.js'
      }
    })
    $agentPlugins.set([
      {
        description: '',
        key: 'pixel-overlay',
        name: 'pixel-overlay',
        source: 'user',
        status: 'enabled',
        version: '1.0.0'
      }
    ])

    renderSettings()

    expect(screen.queryByText('agent half missing here')).toBeNull()
  })

  it('does not flag standalone desktop plugins (not from the unified root)', () => {
    $pluginRecords.set({
      standalone: {
        id: 'standalone',
        name: 'Standalone Theme',
        kind: 'disk',
        status: 'loaded',
        file: '/home/user/.config/hermes-desktop/desktop-plugins/standalone/plugin.js'
      }
    })
    $agentPlugins.set([])

    renderSettings()

    expect(screen.queryByText('agent half missing here')).toBeNull()
  })

  it('loads the connected backend plugin list once the gateway opens (badge data)', () => {
    $gatewayState.set('open')

    renderSettings()

    expect(requestGateway).toHaveBeenCalledWith('plugins.manage', expect.objectContaining({ action: 'list' }))
  })
})
