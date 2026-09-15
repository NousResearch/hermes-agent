import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import { getHermesConfigRecord, saveHermesConfigRecord } from '@/api/config'
import type { DesktopAgentRoster, DesktopConnectionsRegistry, HermesApiRequest, HermesConnection } from '@/global'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { _resetFleetRosterForTests } from '@/store/fleet-roster'
import { $activeGatewayProfile, $profiles, refreshProfiles } from '@/store/profile'
import { $activeSessionId, $connection } from '@/store/session'
import { $settingsRequestProfile, $settingsScopeOverride } from '@/store/settings-scope'

import { SettingsProfileScope } from './profile-scope'

const registry = {
  version: 2,
  primary: 'local',
  lastUsed: 'local',
  launchMode: 'primary',
  connections: [
    { id: 'local', kind: 'local', label: 'This device' },
    { id: 'fixture-lab', kind: 'remote', label: 'Lab gateway', url: 'https://lab.example' }
  ]
} as DesktopConnectionsRegistry

const roster = {
  agents: ['local', 'fixture-lab'].flatMap(connectionId =>
    ['default', 'research'].map(profile => ({
      connectionId,
      connectionKind: connectionId === 'local' ? 'local' : 'remote',
      connectionLabel: connectionId,
      profile,
      handle: `${connectionId}-${profile}`
    }))
  ),
  sources: ['local', 'fixture-lab'].map(connectionId => ({ connectionId, reachable: true }))
} as DesktopAgentRoster

const fixtureProfiles = [
  { name: 'default', is_default: true },
  { name: 'research', is_default: false }
] as typeof $profiles.value

// Serve a STABLE list, not an echo of $profiles: since 256edfc1f5 a re-home
// repoints $profiles at the new source's cache (empty until its own read
// lands), so an echoing mock would keep the wiped list forever.
const api = vi.fn(async (request: HermesApiRequest) =>
  request.path === '/api/profiles' ? { profiles: fixtureProfiles } : {}
)

const getConnectionFor = vi.fn()

beforeEach(() => {
  _resetFleetRosterForTests()
  $connection.set({
    connectionId: 'local',
    registryScoped: true,
    mode: 'local',
    baseUrl: 'http://local.example',
    wsUrl: 'ws://local.example',
    token: '',
    isFullscreen: false,
    nativeOverlayWidth: 0,
    platform: 'linux',
    profile: 'research'
  } as unknown as HermesConnection)
  $activeGatewayProfile.set('research')
  $activeSessionId.set('fixture-chat')
  $settingsScopeOverride.set(null)
  $profiles.set([
    { name: 'default', is_default: true },
    { name: 'research', is_default: false }
  ] as typeof $profiles.value)
  $connectionsRegistry.set(registry)
  api.mockClear()
  getConnectionFor.mockClear()
  setApiRequestConnection('local')
  vi.stubGlobal('hermesDesktop', { api, getConnectionFor, getAgentRoster: vi.fn(async () => roster) })
})

afterEach(() => {
  cleanup()
  $connection.set(null)
  $connectionsRegistry.set(null)
  $settingsScopeOverride.set(null)
  setApiRequestConnection(null)
  vi.unstubAllGlobals()
})

it('selects a gateway/profile pair through one grouped radio menu without rehoming chat; reads and writes retain that owner', async () => {
  render(<SettingsProfileScope />)
  const trigger = screen.getByRole('button', { name: /Applies to/ })
  expect(trigger.textContent).toContain('research')
  expect(trigger.textContent).toContain('This device')
  fireEvent.pointerDown(trigger, { button: 0 })
  const remote = await screen.findByRole('menuitemradio', { name: 'research · Lab gateway' })
  expect(
    screen.getByRole('menuitemradio', { name: 'default · Lab gateway' }).querySelector('.codicon-home')
  ).toBeTruthy()
  expect(screen.queryByRole('menuitem', { name: /New profile|Import profile/ })).toBeNull()
  fireEvent.click(remote)
  await waitFor(() =>
    expect($settingsRequestProfile.get()).toEqual({ connectionId: 'fixture-lab', profile: 'research' })
  )
  expect($activeGatewayProfile.get()).toBe('research')
  expect($connection.get()?.connectionId).toBe('local')
  expect($activeSessionId.get()).toBe('fixture-chat')
  expect(getConnectionFor).not.toHaveBeenCalled()
  const owner = $settingsRequestProfile.get()
  await getHermesConfigRecord(owner)
  setApiRequestConnection('fixture-other')
  await saveHermesConfigRecord({ model: { default: 'fixture-model' } }, owner)
  expect(
    api.mock.calls
      .filter(([request]) => request.path === '/api/config')
      .map(([request]) => [request.connectionId, request.profile])
  ).toEqual([
    ['fixture-lab', 'research'],
    ['fixture-lab', 'research']
  ])
  fireEvent.keyDown(screen.getByRole('button', { name: /Applies to/ }), { key: 'ArrowDown' })
  expect(
    (await screen.findByRole('menuitemradio', { name: 'research · Lab gateway' })).getAttribute('aria-checked')
  ).toBe('true')
})

it('groups a managed local primary without a registry scope once, without changing ambient request routing', async () => {
  $connection.set({ ...$connection.get()!, registryScoped: undefined })
  setApiRequestConnection(null)
  render(<SettingsProfileScope />)
  const trigger = screen.getByRole('button', { name: /Applies to/ })
  fireEvent.pointerDown(trigger, { button: 0 })
  await screen.findByRole('menuitemradio', { name: 'research · Lab gateway' })

  expect.soft(trigger.textContent).toContain('research · This device')
  expect.soft(screen.queryByRole('menuitemradio', { name: 'default' })).toBeNull()
  expect.soft(screen.queryByRole('menuitemradio', { name: 'research' })).toBeNull()
  expect(window.document.querySelectorAll('[data-connection-id="local"]')).toHaveLength(1)
  expect(screen.getAllByRole('menuitemradio', { name: / · This device$/ })).toHaveLength($profiles.get().length)
  expect
    .soft(screen.getByRole('menuitemradio', { name: 'research · This device' }).getAttribute('aria-checked'))
    .toBe('true')
  expect($settingsRequestProfile.get()).toBeUndefined()
  await getHermesConfigRecord($settingsRequestProfile.get())
  expect(api.mock.calls.find(([request]) => request.path === '/api/config')?.[0].connectionId).toBeUndefined()
})

it.each(['url', 'ssh'] as const)(
  'keeps an unregistered %s primary legacy even when a local gateway is registered',
  async remoteKind => {
    $connection.set({
      ...$connection.get()!,
      mode: 'remote',
      remoteKind,
      connectionId: undefined,
      registryScoped: undefined
    })
    setApiRequestConnection(null)
    render(<SettingsProfileScope />)
    const trigger = screen.getByRole('button', { name: /Applies to/ })
    expect(trigger.getAttribute('aria-label')).toBe('Applies to: research')
    fireEvent.pointerDown(trigger, { button: 0 })
    await screen.findByRole('menuitemradio', { name: 'research · Lab gateway' })
    expect(screen.getByRole('menuitemradio', { name: 'research' }).getAttribute('aria-checked')).toBe('true')
    expect(screen.getByRole('menuitemradio', { name: 'research · This device' }).getAttribute('aria-checked')).toBe(
      'false'
    )
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'default' }))
    expect($settingsRequestProfile.get()).toBe('default')
    expect(trigger.textContent).toBe('default')
    await getHermesConfigRecord($settingsRequestProfile.get())
    const request = api.mock.calls.find(([candidate]) => candidate.path === '/api/config')?.[0]
    expect(request).toMatchObject({ profile: 'default' })
    expect(request?.connectionId).toBeUndefined()
  }
)

it('picking another legacy profile sets the shared override; re-selecting the active profile clears it', async () => {
  $connection.set({
    ...$connection.get()!,
    mode: 'remote',
    remoteKind: 'url',
    connectionId: undefined,
    registryScoped: undefined
  })
  setApiRequestConnection(null)
  render(<SettingsProfileScope />)
  fireEvent.pointerDown(screen.getByRole('button', { name: /Applies to/ }), { button: 0 })
  fireEvent.click(screen.getByRole('menuitemradio', { name: 'default' }))
  expect($settingsScopeOverride.get()).toBe('default')
  fireEvent.pointerDown(screen.getByRole('button', { name: /Applies to/ }), { button: 0 })
  fireEvent.click(screen.getByRole('menuitemradio', { name: 'research' }))
  expect($settingsScopeOverride.get()).toBeNull()
})

it('keeps registered defaults reachable with an empty/offline roster and preserves the unregistered primary path', async () => {
  vi.mocked(window.hermesDesktop.getAgentRoster!).mockResolvedValue({
    agents: [],
    sources: [{ connectionId: 'fixture-lab', kind: 'remote', label: 'Lab gateway', reachable: false, error: 'offline' }]
  } as DesktopAgentRoster)
  render(<SettingsProfileScope />)
  fireEvent.pointerDown(screen.getByRole('button', { name: /Applies to/ }), { button: 0 })
  expect(await screen.findByRole('menuitemradio', { name: 'default · Lab gateway' })).toBeTruthy()
  fireEvent.click(screen.getByRole('menuitemradio', { name: 'default · Lab gateway' }))
  expect($settingsRequestProfile.get()).toEqual({ connectionId: 'fixture-lab', profile: 'default' })
  act(() => {
    $connection.set({ mode: 'local', profile: 'research' } as HermesConnection)
    $connectionsRegistry.set(null)
  })
  expect($settingsRequestProfile.get()).toBeUndefined()
  // A re-home repoints $profiles at the new source's cache until its own read
  // lands (256edfc1f5) — the menu still renders rows from the roster/default
  // fallbacks, but explicit picks need the refreshed list.
  await act(async () => {
    await refreshProfiles()
  })
  fireEvent.pointerDown(screen.getByRole('button', { name: /Applies to/ }), { button: 0 })
  fireEvent.click(screen.getByRole('menuitemradio', { name: 'default' }))
  expect($settingsRequestProfile.get()).toBe('default')
})

// #89190/#89162 class: after opening a Bot Mode chat, the ACTIVE profile is
// the bot's — so with no override the settings pages silently edit the bot's
// config. The target must be stated (accented) whenever it isn't the default
// profile, override or not.
it('states the edit target loudly when the active profile is a non-default bot (no override)', () => {
  const { container } = render(<SettingsProfileScope />)

  expect($settingsScopeOverride.get()).toBeNull()
  const note = container.querySelector('[role="status"]')
  expect(note).toBeTruthy()
  expect(note?.getAttribute('data-scope-loud')).toBe('true')
  expect(note?.textContent).toContain('research')
})

it('shows no note when following the active DEFAULT profile', () => {
  $activeGatewayProfile.set('default')

  const { container } = render(<SettingsProfileScope />)

  expect(container.querySelector('[role="status"]')).toBeNull()
})

it('keeps the quiet note style for an explicit fleet pick of the default profile', async () => {
  render(<SettingsProfileScope />)
  fireEvent.pointerDown(screen.getByRole('button', { name: /Applies to/ }), { button: 0 })
  fireEvent.click(await screen.findByRole('menuitemradio', { name: 'default · This device' }))
  await waitFor(() => expect($settingsRequestProfile.get()).toEqual({ connectionId: 'local', profile: 'default' }))

  const note = screen.getByRole('status')
  expect(note.getAttribute('data-scope-loud')).toBeNull()
  expect(note.textContent).toContain('default')
})
