// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import type { DesktopAgentRoster, DesktopConnectionsRegistry } from '@/global'
import { $profileRailVisible } from '@/store/profile-rail-prefs'
import type { ProfileInfo } from '@/types/hermes'

import { ProfileSwitcher } from './profile-dropdown-switcher'

// With the colored rail hidden, the statusbar's profile dropdown is the ONLY
// door to switching profiles — it must list every profile of the active
// gateway and select through the same store action the rail uses.

const selectProfile = vi.fn()
const setShowAllProfiles = vi.fn()

const activeGatewayProfiles: ProfileInfo[] = [
  { has_env: false, is_default: true, model: null, name: 'default', path: '/profiles/default', provider: null, skill_count: 0 },
  { has_env: false, is_default: false, model: null, name: 'clippy', path: '/profiles/clippy', provider: null, skill_count: 0 }
]

vi.mock('react-router', () => ({ useNavigate: () => vi.fn() }))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      profiles: {
        allProfiles: 'All profiles',
        fleet: { onGateway: (name: string, gateway: string) => `${name} · ${gateway}` },
        importProfile: 'Import profile…',
        manageProfiles: 'Manage profiles…',
        newProfile: 'New profile',
        switchConnectionFailed: (name: string) => `Could not connect to ${name}`,
        title: 'Profiles'
      }
    }
  })
}))

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: atom('default'),
  $profileColors: atom({}),
  $profileCreateRequest: atom(0),
  $profileOrder: atom([]),
  $profiles: atom([
    { has_env: false, is_default: true, model: null, name: 'default', path: '/profiles/default', provider: null, skill_count: 0 },
    { has_env: false, is_default: false, model: null, name: 'clippy', path: '/profiles/clippy', provider: null, skill_count: 0 }
  ]),
  $showAllProfiles: atom(false),
  ALL_PROFILES: '__all__',
  normalizeProfileKey: (name: string) => name,
  prewarmProfilePick: vi.fn(),
  profileLabel: (profile: { name: string }) => profile.name,
  refreshActiveProfile: vi.fn().mockResolvedValue(undefined),
  selectProfile: (name: string) => selectProfile(name),
  setShowAllProfiles: (value: boolean) => setShowAllProfiles(value),
  sortByProfileOrder: (profiles: Array<{ name: string }>) => profiles
}))

vi.mock('@/store/connections', () => ({
  $activeConnectionId: atom<null | string>('gateway-a'),
  $connectionsRegistry: atom<DesktopConnectionsRegistry | null>(null),
  $hasMultipleConnections: atom(false),
  selectConnection: vi.fn()
}))

vi.mock('@/store/fleet-roster', () => ({ $fleetRoster: atom<DesktopAgentRoster | null>(null) }))
vi.mock('@/store/profile-share', () => ({ runImportProfileFlow: vi.fn() }))
vi.mock('./use-profile-prewarm', () => ({
  useProfilePrewarm: () => ({ cancelPrewarm: vi.fn(), startPrewarm: vi.fn() })
}))
vi.mock('./use-fleet-roster', () => ({ useFleetRoster: () => undefined }))
vi.mock('../../profiles/create-profile-dialog', () => ({ CreateProfileDialog: () => null }))

const connectionsStore = await import('@/store/connections')
const activeConnectionId = connectionsStore.$activeConnectionId as ReturnType<typeof atom<null | string>>
const connectionsRegistry = connectionsStore.$connectionsRegistry as ReturnType<typeof atom<DesktopConnectionsRegistry | null>>
const hasMultipleConnections = connectionsStore.$hasMultipleConnections as ReturnType<typeof atom<boolean>>
const { $fleetRoster: fleetRoster } = await import('@/store/fleet-roster')
const rosterStore = fleetRoster as ReturnType<typeof atom<DesktopAgentRoster | null>>
const { $profiles: profilesStoreAtom } = await import('@/store/profile')
const profilesStore = profilesStoreAtom as ReturnType<typeof atom<ProfileInfo[]>>

const registry: DesktopConnectionsRegistry = {
  connections: [
    { id: 'gateway-a', kind: 'remote', label: 'Gateway A', tokenPreview: null, tokenSet: false, url: 'https://gateway-a.example.com' },
    { id: 'gateway-b', kind: 'remote', label: 'Gateway B', tokenPreview: null, tokenSet: false, url: 'https://gateway-b.example.com' }
  ],
  launchMode: 'primary',
  lastUsed: 'gateway-a',
  primary: 'gateway-a',
  secureTokenStorage: true,
  version: 2
}

const roster: DesktopAgentRoster = {
  agents: [
    {
      connectionId: 'gateway-a',
      connectionKind: 'remote',
      connectionLabel: 'Gateway A',
      handle: 'clippy',
      profile: 'clippy'
    },
    {
      connectionId: 'gateway-b',
      connectionKind: 'remote',
      connectionLabel: 'Gateway B',
      handle: 'other-gateway-profile',
      profile: 'other-gateway-profile'
    }
  ],
  sources: [
    { connectionId: 'gateway-a', kind: 'remote', label: 'Gateway A', reachable: true },
    { connectionId: 'gateway-b', kind: 'remote', label: 'Gateway B', reachable: true }
  ]
}

afterEach(() => {
  cleanup()
  activeConnectionId.set('gateway-a')
  connectionsRegistry.set(null)
  hasMultipleConnections.set(false)
  rosterStore.set(null)
  profilesStore.set(activeGatewayProfiles)
  $profileRailVisible.set(true)
  vi.clearAllMocks()
})

it('switches profiles from the statusbar dropdown while the rail is hidden', async () => {
  act(() => $profileRailVisible.set(false))
  render(<ProfileSwitcher compact />)

  const trigger = screen.getByRole('button', { name: 'Profiles: default' })
  await act(async () => {
    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })
    await Promise.resolve()
  })

  const clippy = await screen.findByRole('menuitemradio', { name: /clippy/ })
  await act(async () => {
    fireEvent.click(clippy)
    await Promise.resolve()
  })

  expect(selectProfile).toHaveBeenCalledWith('clippy')
  expect(setShowAllProfiles).not.toHaveBeenCalled()
})

it('hides other gateways from the ordinary-chat selector', async () => {
  act(() => {
    connectionsRegistry.set(registry)
    hasMultipleConnections.set(true)
    rosterStore.set(roster)
  })
  render(<ProfileSwitcher compact />)

  const trigger = screen.getByRole('button', { name: 'Profiles: default' })
  await act(async () => {
    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })
    await Promise.resolve()
  })

  expect(await screen.findByRole('menuitemradio', { name: /clippy/ })).toBeTruthy()
  expect(screen.queryByRole('menuitem', { name: /other-gateway-profile.*Gateway B/ })).toBeNull()
})

it('keeps every gateway available from the Bot-chat selector', async () => {
  act(() => {
    connectionsRegistry.set(registry)
    hasMultipleConnections.set(true)
    rosterStore.set(roster)
  })
  render(<ProfileSwitcher compact showFleetProfiles />)

  const trigger = screen.getByRole('button', { name: 'Profiles: default' })
  await act(async () => {
    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })
    await Promise.resolve()
  })

  expect(await screen.findByRole('menuitem', { name: /other-gateway-profile.*Gateway B/ })).toBeTruthy()
})

it('keeps the all-profiles fallback without a selected gateway', async () => {
  act(() => {
    activeConnectionId.set(null)
    profilesStore.set([
      ...activeGatewayProfiles,
      {
        has_env: false,
        is_default: false,
        model: null,
        name: 'other-gateway-profile',
        path: '/profiles/other-gateway-profile',
        provider: null,
        skill_count: 0
      }
    ])
  })
  render(<ProfileSwitcher compact />)

  const trigger = screen.getByRole('button', { name: 'Profiles: default' })
  await act(async () => {
    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })
    await Promise.resolve()
  })

  expect(await screen.findByRole('menuitemradio', { name: /other-gateway-profile/ })).toBeTruthy()
})
