import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createPluginContext } from '@/contrib/plugin'
import { $pluginRecords, dropPlugin, publishPlugin } from '@/contrib/plugins-store'
import { pluginSettingsHref } from '@/contrib/settings-pages'
import { $agentPlugins, $agentPluginsStatus, type AgentPluginRow } from '@/store/agent-plugins'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $settingsScopeOverride } from '@/store/settings-scope'
import type { ProfileInfo } from '@/types/hermes'

import { PluginsTab } from '../capabilities/plugins/plugins-tab'
import { OverlayNav } from '../overlays/overlay-split-layout'

import { pluginSettingsNavChildren, PluginSettingsPane, usePluginSettingsRoute } from './plugin-settings'

// Review regressions for Settings ▸ Plugins (PR #133182): every scenario here
// reproduced natively against a real backend. Each test drives the REAL route
// hook under a router, so the URL contract is exercised, not a local stand-in.

const setEnvVar = vi.fn(async (..._args: unknown[]) => ({ ok: true }))

vi.mock('@/api/config', async importOriginal => ({
  ...(await importOriginal<object>()),
  setEnvVar: (...args: unknown[]) => setEnvVar(...args)
}))

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<object>()),
  refreshProfiles: async () => []
}))

const row = (destination: string): AgentPluginRow => ({
  description: '',
  key: 'review-notes',
  name: 'review-notes',
  settings_schema: [
    { description: '', key: 'destination', label: 'Destination', required: false, type: 'string', value: destination },
    {
      description: '',
      env: 'REVIEW_TOKEN',
      has_value: false,
      key: 'token',
      label: 'Token',
      required: false,
      type: 'secret'
    }
  ],
  source: 'user',
  status: 'enabled',
  version: '1.0.0'
})

type Deferred = { promise: Promise<unknown>; resolve: (value: unknown) => void }

const deferred = (): Deferred => {
  let resolve!: (value: unknown) => void
  const promise = new Promise(r => (resolve = r))

  return { promise, resolve }
}

// Per-profile plugin lists the fake backend serves; a profile listed in
// `held` answers only when the test releases it.
let lists: Record<string, AgentPluginRow[]> = {}
let held: Record<string, Deferred> = {}

const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>): Promise<unknown> => {
  const profile = String(params?.profile ?? 'default')

  if (method === 'plugins.manage' && params?.action === 'list') {
    if (held[profile]) {
      await held[profile].promise
    }

    return { plugins: lists[profile] ?? [] }
  }

  if (method === 'plugins.manage' && params?.action === 'settings') {
    return { ok: true, plugin: null }
  }

  return {}
})

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

const profile = (name: string, isDefault = false): ProfileInfo => ({ is_default: isDefault, name }) as ProfileInfo

const Icon = () => null

let currentLocation = ''

function SettingsPlugins() {
  const route = usePluginSettingsRoute(true)
  const location = useLocation()

  currentLocation = `${location.pathname}${location.search}`

  return (
    <>
      <OverlayNav
        groups={[
          {
            active: true,
            children: pluginSettingsNavChildren(route.entries, route.target, route.open),
            icon: Icon,
            id: 'plugins',
            label: 'Plugins',
            onSelect: () => route.open(null)
          }
        ]}
      />
      <PluginSettingsPane {...route} onOpen={route.open} />
    </>
  )
}

const renderSettings = (path: string) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <SettingsPlugins />
    </MemoryRouter>
  )

const chip = (name: string) =>
  screen.queryAllByRole('button', { name }).find(el => el.className.includes('rounded-full'))

const settingsCalls = () =>
  requestGateway.mock.calls.filter(([method, params]) => method === 'plugins.manage' && params?.action === 'settings')

/** Click a Capabilities ▸ Plugins gear and return the deep link it produced. */
function gearHref(props: Parameters<typeof PluginsTab>[0], label: string): string {
  window.location.hash = ''
  const view = render(<PluginsTab {...props} />)

  fireEvent.click(screen.getByRole('button', { name: label }))
  view.unmount()

  return window.location.hash.slice(1)
}

describe('Settings ▸ Plugins ownership (review of #133182)', () => {
  const disposers: Array<() => void> = []

  beforeEach(() => {
    lists = { default: [row('DEFAULT-ready')], 'review-b': [row('B-original')], 'review-empty': [] }
    held = {}
    requestGateway.mockClear()
    setEnvVar.mockClear()
    $agentPlugins.set([])
    $agentPluginsStatus.set('idle')
    $activeGatewayProfile.set('default')
    $settingsScopeOverride.set(null)
    $profiles.set([profile('default', true), profile('review-b'), profile('review-empty')])
  })

  afterEach(() => {
    cleanup()
    disposers.splice(0).forEach(dispose => dispose())
    $settingsScopeOverride.set(null)
    window.location.hash = ''
  })

  // P1: a dirty draft (secret included) must never be submitted to the profile
  // the user switched to while that profile's list was still loading.
  it('drops the old profile’s draft on a profile switch and never saves it into the new profile', async () => {
    renderSettings(pluginSettingsHref())
    await waitFor(() => expect(document.querySelector('[data-tour$="review-notes"]')).toBeTruthy())
    fireEvent.click(document.querySelector<HTMLButtonElement>('[data-tour$="review-notes"]')!)

    const destination = await screen.findByLabelText(/Destination/)

    expect((destination as HTMLInputElement).value).toBe('DEFAULT-ready')
    fireEvent.change(destination, { target: { value: 'DEFAULT-UNSAVED' } })
    fireEvent.change(screen.getByLabelText('Token'), { target: { value: 'DEFAULT-SECRET' } })

    held['review-b'] = deferred()
    fireEvent.click(chip('review-b')!)

    // B's list is still in flight: the A draft must be gone, and nothing can be saved.
    const stale = screen.queryByLabelText(/Destination/) as HTMLInputElement | null

    expect(stale?.value ?? '').not.toBe('DEFAULT-UNSAVED')
    expect((screen.queryByLabelText('Token') as HTMLInputElement | null)?.value ?? '').toBe('')

    const save = screen.queryByRole('button', { name: 'Save settings' }) as HTMLButtonElement | null

    if (save && !save.disabled) {
      fireEvent.click(save)
    }

    await act(async () => {
      held['review-b']!.resolve(undefined)
      await held['review-b']!.promise
    })

    await waitFor(() => expect((screen.getByLabelText(/Destination/) as HTMLInputElement).value).toBe('B-original'))
    expect(settingsCalls()).toEqual([])
    expect(setEnvVar).not.toHaveBeenCalled()

    // Saving now writes B's own edit to B.
    fireEvent.change(screen.getByLabelText(/Destination/), { target: { value: 'B-EDIT' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save settings' }))
    await waitFor(() => expect(settingsCalls()).toHaveLength(1))
    expect(settingsCalls()[0]![1]).toMatchObject({ profile: 'review-b', values: { destination: 'B-EDIT' } })
  })

  // P1: the Capabilities gear carries the profile Capabilities had selected,
  // and Settings opens (and saves) THAT profile — not its own scope.
  it('opens the gear’s page for the profile selected in Capabilities', async () => {
    $agentPlugins.set(lists['review-b']!)
    $agentPluginsStatus.set('ready')

    const href = gearHref({ profile: 'review-b' }, 'Settings: review-notes')

    expect(new URLSearchParams(href.split('?')[1]).get('profile')).toBe('review-b')

    $agentPlugins.set([])
    $agentPluginsStatus.set('idle')
    requestGateway.mockClear()
    renderSettings(href)

    await waitFor(() => expect((screen.getByLabelText(/Destination/) as HTMLInputElement).value).toBe('B-original'))
    expect($settingsScopeOverride.get()).toBe('review-b')
    // The one-shot hand-off is consumed: the URL no longer pins the scope.
    expect(new URLSearchParams(currentLocation.split('?')[1]).get('profile')).toBeNull()

    fireEvent.change(screen.getByLabelText(/Destination/), { target: { value: 'GEAR-FOR-B' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save settings' }))
    await waitFor(() => expect(settingsCalls()).toHaveLength(1))
    expect(settingsCalls()[0]![1]).toMatchObject({ profile: 'review-b', values: { destination: 'GEAR-FOR-B' } })
  })

  // P2: a profile without the plugin must keep the "Applies to" selector so the
  // user can switch back.
  it('keeps the profile selector on the overview when the selected profile lacks the plugin', async () => {
    renderSettings(pluginSettingsHref())
    await waitFor(() => expect(document.querySelector('[data-tour$="review-notes"]')).toBeTruthy())
    fireEvent.click(document.querySelector<HTMLButtonElement>('[data-tour$="review-notes"]')!)
    await screen.findByLabelText(/Destination/)

    fireEvent.click(chip('review-empty')!)
    await waitFor(() => expect(screen.queryByLabelText(/Destination/)).toBeNull())
    await waitFor(() => expect(screen.getByText('No plugin has settings yet.')).toBeTruthy())

    expect(chip('default')).toBeTruthy()
    fireEvent.click(chip('default')!)
    await waitFor(() => expect(document.querySelector('[data-tour$="review-notes"]')).toBeTruthy())
  })

  // P2: desktop plugin `agent` registering page `notes` must not share an
  // identity with agent plugin `notes`'s automatic page.
  it('keeps a desktop page and an agent page apart even when their ids line up', async () => {
    lists.default = [{ ...row('AGENT-NOTES'), key: 'notes', name: 'notes' }]
    publishPlugin(
      { id: 'agent', kind: 'disk', name: 'Adversarial', status: 'loaded' },
      {
        activate: () => undefined,
        deactivate: () => undefined
      }
    )
    disposers.push(() => dropPlugin('agent'))
    createPluginContext('agent', dispose => disposers.push(dispose)).registerSettingsPage({
      id: 'notes',
      render: () => <p>Desktop landing</p>,
      title: 'Desktop notes'
    })

    renderSettings(pluginSettingsHref())
    await waitFor(() => expect(screen.getAllByRole('button', { name: /notes/ }).length).toBeGreaterThan(1))

    const ids = [...document.querySelectorAll('[data-tour^="nav-plugins:"]')].map(el => el.getAttribute('data-tour'))

    expect(new Set(ids).size).toBe(ids.length)

    // The agent row's gear opens the agent form, not the desktop page.
    $agentPlugins.set(lists.default)
    $agentPluginsStatus.set('ready')
    expect($pluginRecords.get().agent).toBeTruthy()

    const href = gearHref({ profile: 'default' }, 'Settings: notes')

    cleanup()
    renderSettings(href)
    await waitFor(() => expect((screen.getByLabelText(/Destination/) as HTMLInputElement).value).toBe('AGENT-NOTES'))
    expect(screen.queryByText('Desktop landing')).toBeNull()
  })

  // P2: `config` is an ordinary sub-page id; the automatic schema form must not
  // consume it.
  it('keeps a plugin sub-page named config reachable by rail and by deep link', async () => {
    createPluginContext('weather', dispose => disposers.push(dispose)).registerSettingsPage({
      children: [{ id: 'config', render: () => <p>CONFIG CHILD CONTENT</p>, title: 'Configuration child' }],
      id: 'main',
      render: () => <p>Weather home</p>,
      title: 'Weather'
    })

    renderSettings(pluginSettingsHref('weather', 'config'))
    await waitFor(() => expect(screen.getByText('CONFIG CHILD CONTENT')).toBeTruthy())
    expect(screen.queryByText('Weather home')).toBeNull()
    expect(screen.getByRole('button', { name: 'Configuration child' })).toBeTruthy()
  })
})
