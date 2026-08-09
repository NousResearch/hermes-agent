import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { requestGateway } = vi.hoisted(() => ({ requestGateway: vi.fn() }))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

import { $pluginRecords } from '@/contrib/plugins-store'
import {
  $agentPluginBusy,
  $agentPlugins,
  $agentPluginsError,
  $agentPluginsStatus,
  type AgentPluginRow
} from '@/store/agent-plugins'
import { $connection, $gatewayState } from '@/store/session'

import { PluginsSettings } from './plugins-settings'

const legacyRow = {
  name: 'Legacy plugin',
  version: '0.20.0',
  description: 'Returned by a pre-key backend',
  source: 'user',
  status: 'disabled'
} satisfies AgentPluginRow

beforeEach(() => {
  requestGateway.mockReset()
  $pluginRecords.set({})
  $agentPlugins.set([legacyRow])
  $agentPluginsStatus.set('ready')
  $agentPluginsError.set(null)
  $agentPluginBusy.set(null)
  $gatewayState.set('idle')
  $connection.set(null)
})

afterEach(() => {
  cleanup()
})

describe('PluginsSettings', () => {
  it('renders and searches plugin rows returned without a canonical key', () => {
    render(<PluginsSettings />)

    expect(screen.getByText('Legacy plugin')).toBeTruthy()

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'pre-key' } })

    expect(screen.getByText('Legacy plugin')).toBeTruthy()
  })

  it('uses the legacy name address when toggling a row without a key', async () => {
    requestGateway.mockResolvedValue({ ok: true, plugin: { ...legacyRow, status: 'enabled' } })

    render(<PluginsSettings />)
    fireEvent.click(screen.getByRole('switch', { name: 'Enable Legacy plugin' }))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('plugins.manage', {
        action: 'toggle',
        name: 'Legacy plugin',
        enable: true
      })
    )
  })

  it('keeps using the canonical key when the backend provides one', async () => {
    const keyedRow = { ...legacyRow, key: 'image_gen/legacy' }

    $agentPlugins.set([keyedRow])
    requestGateway.mockResolvedValue({ ok: true, plugin: { ...keyedRow, status: 'enabled' } })

    render(<PluginsSettings />)
    fireEvent.click(screen.getByRole('switch', { name: 'Enable Legacy plugin' }))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('plugins.manage', {
        action: 'toggle',
        key: 'image_gen/legacy',
        enable: true
      })
    )
  })
})
