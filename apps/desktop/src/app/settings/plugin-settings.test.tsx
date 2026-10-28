import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createPluginContext } from '@/contrib/plugin'
import { type PluginSettingsRoute, resolvePluginSettingsTarget } from '@/contrib/settings-pages'
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
  const [route, setRoute] = useState<null | PluginSettingsRoute>(null)
  const target = resolvePluginSettingsTarget(entries, route)
  const open = setRoute

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
    expect(railItem('plugins:desktop:weather:main')?.textContent).toBe('Weather')
    // Sub-pages stay folded until their plugin is selected.
    expect(railItem('plugins:desktop:weather:main:units')).toBeNull()

    fireEvent.click(railItem('plugins:desktop:weather:main')!)
    expect(screen.getByText('Weather home')).toBeTruthy()
    expect(railItem('plugins:desktop:weather:main:alerts')?.textContent).toBe('Alerts')

    fireEvent.click(railItem('plugins:desktop:weather:main:units')!)
    expect(screen.getByText('Units content')).toBeTruthy()
    expect(screen.queryByText('Weather home')).toBeNull()
  })

  it('renders a config_schema page that saves through plugins.manage settings', async () => {
    render(<Harness />)

    await waitFor(() => expect(railItem('plugins:agent:notes')).toBeTruthy())
    fireEvent.click(railItem('plugins:agent:notes')!)

    const region = screen.getByLabelText(/Region/) as HTMLInputElement
    const saveButton = screen.getByRole('button', { name: 'Save settings' }) as HTMLButtonElement

    expect(region.value).toBe('us')
    // The page action sits on the heading row above the fields, disabled until an edit.
    expect(saveButton.compareDocumentPosition(region) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(saveButton.disabled).toBe(true)
    fireEvent.change(region, { target: { value: 'eu' } })
    expect(saveButton.disabled).toBe(false)
    fireEvent.click(saveButton)

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
    expect(railItem('plugins:desktop:weather:main')).toBeTruthy()

    // Disable = the loader runs every tracked disposer.
    disposers.splice(0).forEach(dispose => dispose())
    await waitFor(() => expect(railItem('plugins:desktop:weather:main')).toBeNull())
  })

  it('shows an overview with an empty state when no plugin has settings', async () => {
    requestGateway.mockImplementationOnce(async () => ({ plugins: [] }))
    render(<Harness />)

    await waitFor(() => expect(screen.getByText('No plugin has settings yet.')).toBeTruthy())
    expect(screen.getByRole('link', { name: /Manage plugins/ }).getAttribute('href')).toBe('#/capabilities?tab=plugins')
  })

  it('lists installed plugins with zero declared settings, naming known config sources (#144485)', async () => {
    const hookRow: AgentPluginRow = {
      config_source: { editable: false, kind: 'config_section', path: 'config.yaml ▸ myhook' },
      description: 'Pre-tool guardrail hook',
      key: 'myhook',
      name: 'myhook',
      settings_schema: [],
      source: 'git',
      status: 'disabled',
      version: '0.4.2'
    }

    requestGateway.mockImplementation(async () => ({ plugins: [hookRow] }))
    render(<Harness />)

    await waitFor(() => expect(screen.getByTestId('plugin-settings-row-agent:myhook')).toBeTruthy())
    expect(screen.queryByText('No plugin has settings yet.')).toBeNull()
    const overviewRow = screen.getByTestId('plugin-settings-row-agent:myhook')
    expect(overviewRow.textContent).toContain('myhook')
    expect(overviewRow.textContent).toContain('0.4.2')
    expect(overviewRow.textContent).toContain('git')
    expect(overviewRow.textContent).toContain('Disabled')

    fireEvent.click(overviewRow)
    expect(screen.getByText(/declares no settings/i)).toBeTruthy()
    expect(screen.getByText('config.yaml ▸ myhook')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Save settings' })).toBeNull()
  })

  it('shows the declared config_source path on editable forms and refuses unresolvable forms (#144485)', async () => {
    const jsonPlugin: AgentPluginRow = {
      config_source: { editable: true, kind: 'json_file', path: '$HERMES_HOME/memx.json' },
      description: 'Sidecar JSON plugin',
      key: 'memx',
      name: 'memx',
      settings_schema: [{ description: '', key: 'workspace', label: 'Workspace', required: false, type: 'string', value: 'team' }],
      source: 'user',
      status: 'enabled',
      version: '1.2.0'
    }

    const brokenPlugin: AgentPluginRow = {
      config_source: { editable: false, kind: 'json_file', path: '../escape.json' },
      description: 'Broken source plugin',
      key: 'broken',
      name: 'broken',
      settings_schema: [{ description: '', key: 'workspace', label: 'Workspace', required: false, type: 'string', value: 'team' }],
      source: 'user',
      status: 'enabled',
      version: '1.0.0'
    }

    requestGateway.mockImplementation(async () => ({ plugins: [jsonPlugin, brokenPlugin] }))
    render(<Harness />)

    await waitFor(() => expect(railItem('plugins:agent:memx')).toBeTruthy())
    fireEvent.click(railItem('plugins:agent:memx')!)
    expect(screen.getByText('$HERMES_HOME/memx.json')).toBeTruthy()
    expect((screen.getByLabelText(/Workspace/) as HTMLInputElement).value).toBe('team')
    expect(screen.getByRole('button', { name: 'Save settings' })).toBeTruthy()

    // Hard rule: editable === false never renders an editable form even if fields are present.
    fireEvent.click(railItem('plugins:agent:broken')!)
    expect(screen.getByText('../escape.json')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Save settings' })).toBeNull()
    expect(screen.queryByLabelText(/Workspace/)).toBeNull()
  })

  it('deep-links provider plugins to Memory & Context or Providers (#144485)', async () => {
    const memProvider: AgentPluginRow = {
      config_source: {
        editable: false,
        kind: 'json_file',
        path: '$HERMES_HOME/supermem/config.json',
        settings_link: 'memory'
      },
      description: 'External memory backend',
      key: 'supermem',
      name: 'supermem',
      settings_schema: [],
      source: 'user',
      status: 'enabled',
      version: '2.0.0'
    }

    requestGateway.mockImplementation(async () => ({ plugins: [memProvider] }))
    render(<Harness />)

    await waitFor(() => expect(railItem('plugins:agent:supermem')).toBeTruthy())
    fireEvent.click(railItem('plugins:agent:supermem')!)

    const link = screen.getByRole('link', { name: /Memory & Context/i })
    expect(link.getAttribute('href')).toBe('#/settings?tab=config:memory&page=persistent')
    expect(screen.getByText('$HERMES_HOME/supermem/config.json')).toBeTruthy()
  })
})
