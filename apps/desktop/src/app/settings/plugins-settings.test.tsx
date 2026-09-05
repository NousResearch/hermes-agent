import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { requestGateway, getProfiles } = vi.hoisted(() => ({
  requestGateway: vi.fn(),
  getProfiles: vi.fn<() => Promise<{ profiles: { name: string; is_default: boolean }[] }>>(async () => ({
    profiles: []
  }))
}))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getProfiles
}))

import { $pluginRecords } from '@/contrib/plugins-store'
import { queryClient } from '@/lib/query-client'
import {
  $agentPluginBusy,
  $agentPlugins,
  $agentPluginsError,
  $agentPluginsStatus,
  type AgentPluginRow
} from '@/store/agent-plugins'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { PluginsSettings } from './plugins-settings'

const legacyRow = {
  name: 'Legacy plugin',
  version: '0.20.0',
  description: 'Returned by a pre-key backend',
  source: 'user',
  status: 'disabled'
} satisfies AgentPluginRow

const renderSettings = () =>
  render(
    <QueryClientProvider client={queryClient}>
      <PluginsSettings />
    </QueryClientProvider>
  )

beforeEach(() => {
  requestGateway.mockReset()
  getProfiles.mockReset()
  getProfiles.mockResolvedValue({ profiles: [] })
  queryClient.clear()
  $pluginRecords.set({})
  $agentPlugins.set([legacyRow])
  $agentPluginsStatus.set('ready')
  $agentPluginsError.set(null)
  $agentPluginBusy.set(null)
  $gatewayState.set('idle')
  $connection.set(null)
  $activeGatewayProfile.set('default')
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('PluginsSettings', () => {
  it('states the backend restart boundary for agent plugin toggles', () => {
    renderSettings()

    expect(screen.getByText('Changes take effect after the backend restarts.', { exact: false })).toBeTruthy()
  })

  it('renders and searches plugin rows returned without a canonical key', () => {
    renderSettings()

    expect(screen.getByText('Legacy plugin')).toBeTruthy()

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'pre-key' } })

    expect(screen.getByText('Legacy plugin')).toBeTruthy()
  })

  it('renders keyless rows read-only instead of falling back to name-addressed toggles', () => {
    // Name-addressed toggles flip every same-named plugin across category
    // dirs (image_gen/fal vs video_gen/fal) — the reason toggles moved to
    // canonical keys. A pre-contract-v6 row must never reach the RPC.
    renderSettings()

    const toggle = screen.getByRole('switch', { name: 'Enable Legacy plugin' })

    expect(toggle.hasAttribute('disabled') || toggle.getAttribute('aria-disabled') === 'true').toBe(true)

    fireEvent.click(toggle)

    expect(requestGateway).not.toHaveBeenCalledWith('plugins.manage', expect.objectContaining({ action: 'toggle' }))
  })

  it('keeps duplicate-named keyless rows distinct (no React key collision)', () => {
    const sibling = {
      ...legacyRow,
      description: 'A second plugin category with the same legacy name'
    }

    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    $agentPlugins.set([legacyRow, sibling])

    renderSettings()

    expect(screen.getAllByRole('switch', { name: 'Enable Legacy plugin' })).toHaveLength(2)
    expect(screen.getByText(sibling.description)).toBeTruthy()
    expect(consoleError.mock.calls.flat().join(' ')).not.toContain('same key')
  })

  it('keeps using the canonical key when the backend provides one', async () => {
    const keyedRow = { ...legacyRow, key: 'legacy' }

    $agentPlugins.set([keyedRow])
    requestGateway.mockResolvedValue({ ok: true, plugin: { ...keyedRow, status: 'enabled' } })

    renderSettings()
    fireEvent.click(screen.getByRole('switch', { name: 'Enable Legacy plugin' }))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('plugins.manage', {
        action: 'toggle',
        key: 'legacy',
        enable: true
      })
    )
  })

  it('shows general bundled plugins while hiding plugins owned by other settings surfaces', () => {
    // General lifecycle plugins such as disk-cleanup and security-guidance are
    // opt-in and have no other Desktop control surface. Provider/platform
    // plugins remain hidden because their owning settings pages manage them.
    $agentPlugins.set([
      legacyRow,
      { ...legacyRow, name: 'disk-cleanup', key: 'disk-cleanup', source: 'bundled', status: 'enabled' },
      {
        ...legacyRow,
        name: 'security-guidance',
        key: 'security-guidance',
        source: 'bundled',
        status: 'enabled'
      },
      { ...legacyRow, name: 'browserbase', key: 'browser/browserbase', source: 'bundled' },
      { ...legacyRow, name: 'chronos', key: 'cron_providers/chronos', source: 'bundled' },
      { ...legacyRow, name: 'basic-auth', key: 'dashboard_auth/basic', source: 'bundled' },
      { ...legacyRow, name: 'deepinfra', key: 'image_gen/deepinfra', source: 'bundled' },
      { ...legacyRow, name: 'discord', key: 'platforms/discord', source: 'bundled' },
      { ...legacyRow, name: 'fal-video', key: 'video_gen/fal', source: 'bundled' },
      { ...legacyRow, name: 'exa', key: 'web/exa', source: 'bundled' },
      { ...legacyRow, name: 'spotify', key: 'spotify', source: 'bundled' },
      { ...legacyRow, name: 'google-meet', key: 'google_meet', source: 'bundled' },
      { ...legacyRow, name: 'langfuse', key: 'observability/langfuse', source: 'bundled' }
    ])

    renderSettings()

    expect(screen.getByText('Legacy plugin')).toBeTruthy()
    expect(screen.getByText('disk-cleanup')).toBeTruthy()
    expect(screen.getByText('security-guidance')).toBeTruthy()
    expect(screen.queryByText('browserbase')).toBeNull()
    expect(screen.queryByText('chronos')).toBeNull()
    expect(screen.queryByText('basic-auth')).toBeNull()
    expect(screen.queryByText('deepinfra')).toBeNull()
    expect(screen.queryByText('discord')).toBeNull()
    expect(screen.queryByText('fal-video')).toBeNull()
    expect(screen.queryByText('exa')).toBeNull()
    expect(screen.queryByText('spotify')).toBeNull()
    expect(screen.queryByText('google-meet')).toBeNull()
    expect(screen.queryByText('langfuse')).toBeNull()
    // Count pill reflects the filtered list, not the raw RPC row count.
    expect(screen.getByText('3 installed', { exact: false })).toBeTruthy()
  })

  it('hides legacy other-surface categories when the backend reports an unreliable source', () => {
    // Older backends may report source unreliably — the key-prefix fallback
    // still hides categories other surfaces own.
    $agentPlugins.set([{ ...legacyRow, name: 'deepinfra', key: 'model-providers/deepinfra', source: 'user' }])

    renderSettings()

    expect(screen.queryByText('deepinfra')).toBeNull()
  })

  it('shows no profile selector with a single profile', async () => {
    getProfiles.mockResolvedValue({ profiles: [{ name: 'default', is_default: true }] })

    renderSettings()

    await waitFor(() => expect(getProfiles).toHaveBeenCalled())
    expect(screen.queryByText('Applies to:')).toBeNull()
  })

  it('lists the active profile scope without a profile param and reloads scoped on change', async () => {
    getProfiles.mockResolvedValue({
      profiles: [
        { name: 'default', is_default: true },
        { name: 'work', is_default: false }
      ]
    })
    requestGateway.mockResolvedValue({ plugins: [legacyRow] })
    $gatewayState.set('open')

    renderSettings()

    // Active profile scope: no profile param — older backends unchanged.
    await waitFor(() => expect(requestGateway).toHaveBeenCalledWith('plugins.manage', { action: 'list' }))
    await waitFor(() => expect(screen.getByText('Applies to:')).toBeTruthy())
  })

  it('sends toggles through the selected profile scope', async () => {
    // jsdom's scrollIntoView is missing/non-functional; Radix Select calls it
    // when the dropdown opens.
    Element.prototype.scrollIntoView = vi.fn()

    const keyedRow = { ...legacyRow, key: 'legacy' }

    getProfiles.mockResolvedValue({
      profiles: [
        { name: 'default', is_default: true },
        { name: 'work', is_default: false }
      ]
    })
    requestGateway.mockImplementation(async (method: string, params?: Record<string, unknown>) => {
      if (params?.action === 'list') {
        return { plugins: [keyedRow] }
      }

      return { ok: true, plugin: { ...keyedRow, status: 'enabled' } }
    })
    $gatewayState.set('open')

    renderSettings()

    await waitFor(() => expect(screen.getByText('Applies to:')).toBeTruthy())

    // Select the non-active profile scope.
    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.click(await screen.findByText('work'))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('plugins.manage', { action: 'list', profile: 'work' })
    )

    fireEvent.click(screen.getByRole('switch', { name: 'Enable Legacy plugin' }))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('plugins.manage', {
        action: 'toggle',
        key: 'legacy',
        enable: true,
        profile: 'work'
      })
    )
  })
})
