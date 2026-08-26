import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopAgentRoster, DesktopConnectionsRegistry } from '@/global'

import { ProfileRail } from './profile-switcher'

// The fleet rail: with several registered gateways, every gateway's agents sit
// on the one strip — the active gateway's squares exactly as before, the rest
// as at-rest groups behind a hairline + kind glyph. Clicking an at-rest square
// performs the same re-home the statusbar switcher does, on that exact
// (gateway, profile). Single-gateway rendering must stay byte-identical.

const navigate = vi.fn()
const selectConnection = vi.fn()
const selectProfile = vi.fn()
const getAgentRoster = vi.fn()

vi.mock('react-router', () => ({
  useNavigate: () => navigate
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel', delete: 'Delete' },
      profiles: {
        actions: 'Actions',
        allProfiles: 'All profiles',
        autoColor: 'Auto',
        color: 'Color…',
        colorFor: 'Color',
        connectGateway: 'Manage gateways…',
        editSoul: 'Edit SOUL.md…',
        exportProfile: 'Export profile…',
        failedLoadSoul: 'Failed to load SOUL.md',
        failedSaveSoul: 'Failed to save SOUL.md',
        fleet: {
          allOnGateway: 'All profiles on this gateway',
          deleteOn: (gateway: string) => ` on ${gateway}`,
          gateway: (gateway: string) => `Profiles on ${gateway}`,
          gatewayUnreachable: (gateway: string) => `${gateway} · unreachable`,
          onGateway: (name: string, gateway: string) => `${name} · ${gateway}`,
          switchTo: (name: string, gateway: string) => `Switch to ${name} on ${gateway}`
        },
        importProfile: 'Import profile…',
        manageProfiles: 'Manage profiles…',
        newProfile: 'New profile',
        remoteOverride: {
          badge: (host: string) => `Runs on ${host}`,
          menuItem: 'Connect to a remote host…'
        },
        renameMenu: 'Rename…',
        saveSoul: 'Save',
        saving: 'Saving…',
        setColor: (color: string) => `Set color ${color}`,
        showAllProfiles: 'Show all profiles',
        soulSaved: 'SOUL.md saved',
        switchConnectionFailed: (name: string) => `Could not connect to ${name}`,
        switchToProfile: (name: string) => `Switch to ${name}`,
        title: 'Profiles'
      },
      settings: { connections: { kindCloud: 'Cloud', kindLocal: 'This device', kindRemote: 'Remote', kindSsh: 'SSH' } }
    }
  })
}))

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: atom('default'),
  $profileColors: atom({}),
  $profileCreateRequest: atom(0),
  $profileOrder: atom([]),
  $profiles: atom([{ is_default: true, name: 'default' }]),
  $profileScope: atom('default'),
  ALL_PROFILES: '*',
  normalizeProfileKey: (name: string) => name,
  profileLabel: (profile: { display_name?: string; name: string }) =>
    (profile.display_name ?? '').trim() || profile.name,
  refreshActiveProfile: vi.fn().mockResolvedValue(undefined),
  selectProfile: (name: string) => selectProfile(name),
  setProfileColor: vi.fn(),
  setProfileOrder: vi.fn(),
  setShowAllProfiles: vi.fn(),
  sortByProfileOrder: (profiles: unknown[]) => profiles
}))

vi.mock('@/store/connections', () => ({
  $activeConnectionId: atom<null | string>(null),
  $connectionsRegistry: atom<DesktopConnectionsRegistry | null>(null),
  $hasMultipleConnections: atom(false),
  selectConnection: (...args: unknown[]) => selectConnection(...args)
}))

vi.mock('@/store/profile-share', () => ({
  runExportProfileFlow: vi.fn(),
  runImportProfileFlow: vi.fn()
}))

vi.mock('./use-profile-prewarm', () => ({
  useProfilePrewarm: () => ({ cancelPrewarm: vi.fn(), startPrewarm: vi.fn() })
}))

vi.mock('./use-profile-rail-refresh-on-active', () => ({
  useProfileRailRefreshOnActive: () => undefined
}))

vi.mock('@/hermes', () => ({
  getProfileSoul: vi.fn().mockResolvedValue({ content: '' }),
  updateProfileSoul: vi.fn()
}))

vi.mock('@/components/chat/code-editor', () => ({ CodeEditor: () => null }))
vi.mock('../../profiles/create-profile-dialog', () => ({ CreateProfileDialog: () => null }))
vi.mock('../../profiles/delete-profile-dialog', () => ({ DeleteProfileDialog: () => null }))
vi.mock('../../profiles/rename-profile-dialog', () => ({ RenameProfileDialog: () => null }))

const connectionsStore = await import('@/store/connections')
const hasMultipleConnections = connectionsStore.$hasMultipleConnections as ReturnType<typeof atom<boolean>>
const activeConnectionId = connectionsStore.$activeConnectionId as ReturnType<typeof atom<null | string>>

const connectionsRegistry = connectionsStore.$connectionsRegistry as ReturnType<
  typeof atom<DesktopConnectionsRegistry | null>
>

const { $profiles, $profileScope } = await import('@/store/profile')
const profiles = $profiles as ReturnType<typeof atom<Array<{ is_default: boolean; name: string }>>>
const profileScope = $profileScope as ReturnType<typeof atom<string>>
const { _resetFleetRosterForTests } = await import('@/store/fleet-roster')

const registry: DesktopConnectionsRegistry = {
  connections: [
    { id: 'local', kind: 'local', label: 'This device' },
    { id: 'pandora', kind: 'remote', label: 'Pandora', url: 'https://pandora.example' },
    { id: 'vps', kind: 'ssh', label: 'VPS', host: 'vps.example' }
  ],
  launchMode: 'primary',
  lastUsed: 'pandora',
  primary: 'pandora',
  version: 2
} as DesktopConnectionsRegistry

const roster: DesktopAgentRoster = {
  agents: [
    { connectionId: 'pandora', connectionKind: 'remote', connectionLabel: 'Pandora', profile: 'default', handle: 'hermes-pandora' },
    { connectionId: 'pandora', connectionKind: 'remote', connectionLabel: 'Pandora', profile: 'scout', handle: 'scout' },
    { connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device', profile: 'default', handle: 'hermes' },
    { connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device', profile: 'omer', handle: 'omer' }
  ],
  sources: [
    { connectionId: 'pandora', kind: 'remote', label: 'Pandora', reachable: true },
    { connectionId: 'local', kind: 'local', label: 'This device', reachable: true },
    { connectionId: 'vps', kind: 'ssh', label: 'VPS', reachable: false, error: 'timed out' }
  ]
}

function armFleet() {
  hasMultipleConnections.set(true)
  connectionsRegistry.set(registry)
  activeConnectionId.set('pandora')
  profiles.set([
    { is_default: true, name: 'default' },
    { is_default: false, name: 'scout' }
  ])
}

async function renderFleet() {
  const view = render(<ProfileRail />)

  // The roster arrives asynchronously via the Electron bridge.
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
  })

  return view.container
}

beforeEach(() => {
  getAgentRoster.mockResolvedValue(roster)
  selectConnection.mockResolvedValue(undefined)
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = { getAgentRoster }
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  _resetFleetRosterForTests()
  hasMultipleConnections.set(false)
  connectionsRegistry.set(null)
  activeConnectionId.set(null)
  profileScope.set('default')
  profiles.set([{ is_default: true, name: 'default' }])
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

describe('ProfileRail fleet mode', () => {
  it('stays on the single-gateway path with one registered gateway', async () => {
    const container = await renderFleet()

    expect(getAgentRoster).not.toHaveBeenCalled()
    expect(screen.queryByRole('group', { name: /^Profiles on/ })).toBeNull()
    expect(container.querySelector('[data-slot="profile-rail-divider"]')).toBeNull()
    expect(screen.getByRole('button', { name: 'Manage gateways…' })).toBeTruthy()
  })

  it('lays every other gateway on the strip as an at-rest group, in switcher order', async () => {
    armFleet()
    const container = await renderFleet()

    expect(getAgentRoster).toHaveBeenCalledTimes(1)

    const groups = Array.from(container.querySelectorAll('[data-slot="profile-rail-gateway"]')).map(node => [
      node.getAttribute('data-connection-id'),
      node.getAttribute('data-active') === 'true'
    ])

    // Registry order for the whole strip — This device first (switcher
    // order), then by label — with the active gateway (Pandora) in ITS slot,
    // never pulled to the front.
    expect(groups).toEqual([
      ['local', false],
      ['pandora', true],
      ['vps', false]
    ])

    // Every group is headed by its kind glyph; hairlines only between groups.
    const dividers = Array.from(container.querySelectorAll('[data-slot="profile-rail-divider"]')).map(node =>
      node.getAttribute('data-connection-id')
    )

    expect(dividers).toEqual(['local', 'pandora', 'vps'])

    const local = screen.getByRole('group', { name: 'Profiles on This device' })
    expect(within(local).getByRole('button', { name: 'default · This device' })).toBeTruthy()
    expect(within(local).getByRole('button', { name: 'omer · This device' })).toBeTruthy()

    // The active gateway's own squares are unchanged and unqualified.
    expect(screen.getByRole('button', { name: 'scout' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'default' })).toBeTruthy()

    // Fleet pill: "all on this gateway" replaces the default↔all toggle.
    expect(screen.getByRole('button', { name: 'All profiles on this gateway' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Manage gateways…' })).toBeNull()
  })

  it('marks an unreachable gateway but never hides it', async () => {
    armFleet()
    const container = await renderFleet()

    const vps = container.querySelector('[data-slot="profile-rail-gateway"][data-connection-id="vps"]')
    expect(vps?.getAttribute('data-reachable')).toBe('false')
    expect(
      container.querySelector(
        '[data-slot="profile-rail-divider"][data-connection-id="vps"] [data-slot="profile-rail-unreachable"]'
      )
    ).toBeTruthy()
    expect(within(vps as HTMLElement).getByRole('button', { name: 'default · VPS' })).toBeTruthy()
  })

  it('re-homes onto the exact (gateway, profile) when an at-rest square is clicked', async () => {
    armFleet()
    await renderFleet()

    let settle: () => void = () => undefined
    selectConnection.mockImplementationOnce(() => new Promise<void>(resolve => (settle = resolve)))

    const omer = screen.getByRole('button', { name: 'omer · This device' })
    fireEvent.click(omer)

    expect(selectConnection).toHaveBeenCalledWith('local', { profile: 'omer' })
    expect(selectProfile).not.toHaveBeenCalled()
    // The dial spinner sits on the clicked square, not in the statusbar.
    expect(omer.getAttribute('aria-busy')).toBe('true')

    await act(async () => {
      settle()
      await Promise.resolve()
    })

    expect(omer.getAttribute('aria-busy')).toBeNull()
  })

  it('re-homes onto another gateway default from its home square', async () => {
    armFleet()
    await renderFleet()

    fireEvent.click(screen.getByRole('button', { name: 'default · This device' }))

    expect(selectConnection).toHaveBeenCalledWith('local', { profile: 'default' })
  })

  it('keeps the active gateway click on the plain profile path', async () => {
    armFleet()
    await renderFleet()

    fireEvent.click(screen.getByRole('button', { name: 'scout' }))

    expect(selectProfile).toHaveBeenCalledWith('scout')
    expect(selectConnection).not.toHaveBeenCalled()
  })

  it('keeps every group in its slot when a different gateway is active', async () => {
    armFleet()
    activeConnectionId.set('local')
    profiles.set([
      { is_default: true, name: 'default' },
      { is_default: false, name: 'omer' }
    ])
    const container = await renderFleet()

    const groups = Array.from(container.querySelectorAll('[data-slot="profile-rail-gateway"]')).map(node => [
      node.getAttribute('data-connection-id'),
      node.getAttribute('data-active') === 'true'
    ])

    expect(groups).toEqual([
      ['local', true],
      ['pandora', false],
      ['vps', false]
    ])
    expect(screen.getByRole('button', { name: 'scout · Pandora' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'omer' })).toBeTruthy()
  })

  it('counts the whole fleet toward the condensed threshold and sections the menu by gateway', async () => {
    armFleet()
    profiles.set([
      { is_default: true, name: 'default' },
      ...Array.from({ length: 11 }, (_, index) => ({ is_default: false, name: `p${index + 1}` }))
    ])
    // 11 named on Pandora + local (default, omer) + vps (default) = 14 > 13.
    const container = await renderFleet()

    expect(screen.getByRole('button', { name: 'Profiles' })).toBeTruthy()
    expect(container.querySelector('[data-slot="profile-rail-rest-square"]')).toBeNull()
  })
})
