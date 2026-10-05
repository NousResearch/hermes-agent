import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createPluginContext } from '@/contrib/plugin'
import { resolvePluginSettingsTarget } from '@/contrib/settings-pages'
import { $agentPlugins, $agentPluginsStatus, type AgentPluginRow } from '@/store/agent-plugins'

import { OverlayNav } from '../overlays/overlay-split-layout'

import { pluginSettingsNavChildren, PluginSettingsPane, usePluginSettingsEntries } from './plugin-settings'

const notesRow: AgentPluginRow = {
  description: 'Daily notes',
  key: 'notes',
  name: 'notes',
  settings_schema: [{ description: '', key: 'region', label: 'Region', required: false, type: 'string', value: 'us' }],
  source: 'user',
  status: 'enabled',
  version: '1.0.0'
}

const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>): Promise<unknown> => {
  if (method === 'plugins.manage' && params?.action === 'list') {
    return { plugins: [notesRow] }
  }

  if (method === 'plugins.manage' && params?.action === 'settings') {
    return { ok: true, plugin: notesRow }
  }

  return {}
})

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

const Icon = () => null

function Harness() {
  const entries = usePluginSettingsEntries()
  const [selection, setSelection] = useState<{ page: null | string; plugin: null | string }>({
    page: null,
    plugin: null
  })
  const target = resolvePluginSettingsTarget(entries, selection.plugin, selection.page)
  const open = (plugin: null | string, page?: string) => setSelection({ page: page ?? null, plugin })

  return (
    <>
      <OverlayNav
        groups={[
          {
            active: true,
            children: pluginSettingsNavChildren(entries, target, open),
            icon: Icon,
            id: 'plugins',
            label: 'Plugins',
            onSelect: () => open(null)
          }
        ]}
      />
      <PluginSettingsPane entries={entries} onOpen={open} target={target} />
    </>
  )
}

const railItem = (id: string) => document.querySelector<HTMLButtonElement>(`[data-tour="nav-${id}"]`)

describe('Settings ▸ Plugins', () => {
  const disposers: Array<() => void> = []

  beforeEach(() => {
    $agentPlugins.set([])
    $agentPluginsStatus.set('idle')
    requestGateway.mockClear()
  })

  afterEach(() => {
    cleanup()
    disposers.splice(0).forEach(dispose => dispose())
  })

  it('lists every plugin with settings and walks into a sub-page, WoW-AddOns style', async () => {
    const ctx = createPluginContext('weather', dispose => disposers.push(dispose))

    ctx.registerSettingsPage({
      children: [
        { id: 'units', render: () => <p>Units content</p>, title: 'Units' },
        { id: 'alerts', render: () => <p>Alerts content</p>, title: 'Alerts' }
      ],
      id: 'main',
      render: () => <p>Weather home</p>,
      title: 'Weather'
    })

    render(<Harness />)

    // The automatic config_schema page arrives with the agent plugin list.
    await waitFor(() => expect(railItem('plugins:agent:notes')).toBeTruthy())
    expect(railItem('plugins:weather:main')?.textContent).toBe('Weather')
    // Sub-pages stay folded until their plugin is selected.
    expect(railItem('plugins:weather:main:units')).toBeNull()

    fireEvent.click(railItem('plugins:weather:main')!)
    expect(screen.getByText('Weather home')).toBeTruthy()
    expect(railItem('plugins:weather:main:alerts')?.textContent).toBe('Alerts')

    fireEvent.click(railItem('plugins:weather:main:units')!)
    expect(screen.getByText('Units content')).toBeTruthy()
    expect(screen.queryByText('Weather home')).toBeNull()
  })

  it('renders a config_schema page that saves through plugins.manage settings', async () => {
    render(<Harness />)

    await waitFor(() => expect(railItem('plugins:agent:notes')).toBeTruthy())
    fireEvent.click(railItem('plugins:agent:notes')!)

    const region = screen.getByLabelText(/Region/) as HTMLInputElement

    expect(region.value).toBe('us')
    fireEvent.change(region, { target: { value: 'eu' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save settings' }))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('plugins.manage', {
        action: 'settings',
        key: 'notes',
        // Scoped to the Settings profile selector (the active profile here).
        profile: 'default',
        values: { region: 'eu' }
      })
    )
  })

  it('drops a disabled plugin’s page from the list', async () => {
    const ctx = createPluginContext('weather', dispose => disposers.push(dispose))

    ctx.registerSettingsPage({ id: 'main', render: () => <p>Weather home</p>, title: 'Weather' })
    render(<Harness />)
    expect(railItem('plugins:weather:main')).toBeTruthy()

    // Disable = the loader runs every tracked disposer.
    disposers.splice(0).forEach(dispose => dispose())
    await waitFor(() => expect(railItem('plugins:weather:main')).toBeNull())
  })

  it('shows an overview with an empty state when no plugin has settings', async () => {
    requestGateway.mockImplementationOnce(async () => ({ plugins: [] }))
    render(<Harness />)

    await waitFor(() => expect(screen.getByText('No plugin has settings yet.')).toBeTruthy())
    expect(screen.getByRole('link', { name: /Manage plugins/ }).getAttribute('href')).toBe('#/capabilities?tab=plugins')
  })
})
