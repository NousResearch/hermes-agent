// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { refreshActiveProfile } from '@/store/profile'
import { $profileRailVisible } from '@/store/profile-rail-prefs'

import { ProfileSwitcher } from './profile-dropdown-switcher'

// With the colored rail hidden, the statusbar's profile dropdown is the ONLY
// door to switching profiles — it must list every profile of the active
// gateway and select through the same store action the rail uses.

const selectProfile = vi.fn()
const setShowAllProfiles = vi.fn()

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
    { is_default: true, name: 'default' },
    { is_default: false, name: 'clippy' }
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
  $activeConnectionId: atom<null | string>(null),
  $connectionsRegistry: atom(null),
  $hasMultipleConnections: atom(false),
  selectConnection: vi.fn()
}))

vi.mock('@/store/profile-share', () => ({ runImportProfileFlow: vi.fn() }))
vi.mock('./use-profile-prewarm', () => ({
  useProfilePrewarm: () => ({ cancelPrewarm: vi.fn(), startPrewarm: vi.fn() })
}))
vi.mock('./use-fleet-roster', () => ({ useFleetRoster: () => undefined }))
vi.mock('../../profiles/create-profile-dialog', () => ({ CreateProfileDialog: () => null }))

afterEach(() => {
  cleanup()
  $profileRailVisible.set(true)
  vi.clearAllMocks()
  vi.unstubAllGlobals()
  delete window.document.documentElement.dataset.hermesDesktopHost
})

async function openDropdown() {
  render(<ProfileSwitcher compact />)

  const trigger = screen.getByRole('button', { name: 'Profiles: default' })
  await act(async () => {
    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })
    await Promise.resolve()
  })
  await screen.findByRole('menuitemradio', { name: /clippy/ })
}

const launchMenuTriggers = () => window.document.querySelectorAll('[data-hermes-context-menu-trigger]')

// Native Desktop's statusbar picker is main's: its rows carry no right-click
// launch menu, and it leaves refreshing the profile list to the rail and
// Manage Profiles.
it('adds no right-click launch menu and no focus refresh in native Desktop', async () => {
  vi.stubGlobal('hermesDesktop', {})
  act(() => $profileRailVisible.set(false))
  await openDropdown()

  expect(launchMenuTriggers()).toHaveLength(0)
  expect(refreshActiveProfile).not.toHaveBeenCalled()
})

// The Webapp always shows this dropdown instead of the rail's squares, so it
// carries the rail's right-click launch menu and refreshes the list on focus.
it('carries the rail\'s launch menu and focus refresh in browser-hosted Desktop', async () => {
  window.document.documentElement.dataset.hermesDesktopHost = 'browser'
  act(() => $profileRailVisible.set(false))
  await openDropdown()

  expect(launchMenuTriggers().length).toBeGreaterThan(0)
  expect(refreshActiveProfile).toHaveBeenCalled()
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
