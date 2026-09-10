import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $pluginRecords } from '@/contrib/plugins-store'
import { $agentPlugins, $agentPluginsStatus } from '@/store/agent-plugins'
import { $pluginInstallRequest, closePluginInstallRequest } from '@/store/plugin-install-request'

import { DesktopPluginsSection } from './desktop-plugins-section'

beforeEach(() => {
  $pluginRecords.set({})
  $agentPlugins.set([])
  $agentPluginsStatus.set('ready')
  closePluginInstallRequest()
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('DesktopPluginsSection', () => {
  it('flags a unified-root desktop half whose agent half is missing in the scoped profile', () => {
    $pluginRecords.set({
      'pixel-overlay': {
        id: 'pixel-overlay',
        name: 'Pixel Overlay',
        kind: 'disk',
        status: 'loaded',
        file: '/home/user/.hermes/plugins/pixel-overlay/desktop/plugin.js'
      }
    })
    $agentPlugins.set([]) // scoped backend has no agent half

    render(<DesktopPluginsSection profile="workbot" />)

    expect(screen.getByText('agent half missing here')).toBeTruthy()
  })

  it('does not flag when the agent half exists, nor standalone desktop plugins', () => {
    $pluginRecords.set({
      'pixel-overlay': {
        id: 'pixel-overlay',
        name: 'Pixel Overlay',
        kind: 'disk',
        status: 'loaded',
        file: '/home/user/.hermes/plugins/pixel-overlay/desktop/plugin.js'
      },
      standalone: {
        id: 'standalone',
        name: 'Standalone Theme',
        kind: 'disk',
        status: 'loaded',
        file: '/home/user/.config/hermes-desktop/desktop-plugins/standalone/plugin.js'
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

    render(<DesktopPluginsSection profile={null} />)

    expect(screen.queryByText('agent half missing here')).toBeNull()
    expect(screen.getByText('Pixel Overlay')).toBeTruthy()
    expect(screen.getByText('Standalone Theme')).toBeTruthy()
  })

  it('repairs the agent half into the SCOPED profile, not the active one', async () => {
    $pluginRecords.set({
      'pixel-overlay': {
        id: 'pixel-overlay',
        name: 'Pixel Overlay',
        kind: 'disk',
        status: 'loaded',
        file: '/home/user/.hermes/plugins/pixel-overlay/desktop/plugin.js'
      }
    })

    render(<DesktopPluginsSection profile="workbot" />)
    screen.getByRole('button', { name: 'agent half missing here' }).click()

    await vi.waitFor(() => {
      const request = $pluginInstallRequest.get()

      expect(request?.profile).toBe('workbot')
      expect(request?.legacyHint).toBe('agent')
      expect(request?.repo).toBe('pixel-overlay')
    })
  })
})
