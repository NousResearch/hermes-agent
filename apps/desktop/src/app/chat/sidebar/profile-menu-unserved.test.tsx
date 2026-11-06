import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { DesktopAgentRoster } from '@/global'

import { ProfileSwitcher } from './profile-dropdown-switcher'
import { ProfileRail } from './profile-switcher'

// The Webapp's gateway lists every installed profile and serves only some of them; its roster marks
// the rest `reachable: false`, and picking one is refused. The profile menus read that roster and show
// such a profile as unavailable, the way the Bots pane does, instead of failing after the click. Only
// the bridge's roster is stubbed, so the menus' own read of it is covered.

vi.hoisted(() => {
  // The bridge marks the page before the rail module decides it is always condensed.
  window.document.documentElement.dataset.hermesDesktopHost = 'browser'
})

const selectProfile = vi.fn()
const setShowAllProfiles = vi.fn()

vi.mock('react-router', () => ({ useNavigate: () => vi.fn() }))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    locale: 'en',
    t: {
      common: { cancel: 'Cancel', delete: 'Delete' },
      profiles: {
        actions: 'Actions',
        allProfiles: 'All profiles',
        color: 'Color…',
        colorFor: 'Color',
        connectGateway: 'Manage gateways…',
        editSoul: 'Edit SOUL.md…',
        exportMenu: 'Export profile…',
        fleet: { onGateway: (name: string, gateway: string) => `${name} · ${gateway}` },
        importProfile: 'Import profile…',
        manageProfiles: 'Manage profiles…',
        newProfile: 'New profile',
        notServedHere: 'Not served here',
        openInNewWindow: 'Open in new window',
        renameMenu: 'Rename…',
        remoteOverride: { badge: (host: string) => `Runs on ${host}`, menuItem: 'Connect to a remote host…' },
        setAsDefault: 'Set as default',
        showAllProfiles: 'Show all profiles',
        status: {
          needsInput: (count: number) => `${count} needs input`,
          unread: (count: number) => `${count} unread`,
          working: (count: number) => `${count} working`
        },
        switchConnectionFailed: (name: string) => `Could not connect to ${name}`,
        switchToProfile: (name: string) => `Switch to ${name}`,
        title: 'Profiles'
      }
    }
  })
}))

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: atom('solo'),
  $profileColors: atom({}),
  $profileCreateRequest: atom(0),
  $profileOrder: atom([]),
  $profiles: atom([
    { is_default: true, name: 'default' },
    { is_default: false, name: 'research' },
    { is_default: false, name: 'solo' }
  ]),
  $profileScope: atom('solo'),
  $showAllProfiles: atom(false),
  ALL_PROFILES: '__all__',
  normalizeProfileKey: (name: string) => name,
  prewarmProfilePick: vi.fn(),
  profileLabel: (profile: { name: string }) => profile.name,
  refreshActiveProfile: vi.fn().mockResolvedValue(undefined),
  selectProfile: (name: string) => selectProfile(name),
  setProfileColor: vi.fn(),
  setProfileOrder: vi.fn(),
  setShowAllProfiles: (value: boolean) => setShowAllProfiles(value),
  sortByProfileOrder: (profiles: unknown[]) => profiles
}))

vi.mock('@/store/connections', () => ({
  $activeConnectionId: atom<null | string>('local'),
  $connectionsRegistry: atom(null),
  $hasMultipleConnections: atom(false),
  selectConnection: vi.fn()
}))

vi.mock('@/store/profile-share', () => ({ runExportProfileFlow: vi.fn(), runImportProfileFlow: vi.fn() }))
vi.mock('./use-profile-prewarm', () => ({
  useProfilePrewarm: () => ({ cancelPrewarm: vi.fn(), notePointerMove: vi.fn(), startPrewarm: vi.fn() })
}))
vi.mock('@/hermes', () => ({ getProfileSoul: vi.fn().mockResolvedValue({ content: '' }), updateProfileSoul: vi.fn() }))
vi.mock('@/components/chat/code-editor', () => ({ CodeEditor: () => null }))
vi.mock('../../profiles/create-profile-dialog', () => ({ CreateProfileDialog: () => null }))
vi.mock('../../profiles/delete-profile-dialog', () => ({ DeleteProfileDialog: () => null }))
vi.mock('../../profiles/rename-profile-dialog', () => ({ RenameProfileDialog: () => null }))

/** A gateway launched as `solo` that serves no other profile, as the Webapp bridge reports it. */
function standaloneRoster(): DesktopAgentRoster {
  const agent = (profile: string, served: boolean) => ({
    connectionId: 'local',
    connectionKind: 'local' as const,
    connectionLabel: 'localhost:9119',
    handle: profile === 'default' ? 'hermes' : profile,
    profile,
    targetProfile: profile,
    ...(served ? {} : { reachable: false })
  })

  return {
    agents: [agent('default', false), agent('research', false), agent('solo', true)],
    primaryConnectionId: 'local',
    sources: [{ connectionId: 'local', kind: 'local', label: 'localhost:9119', reachable: true }]
  }
}

const getAgentRoster = vi.fn(async () => standaloneRoster())

beforeEach(() => {
  Object.assign(window, { hermesDesktop: { getAgentRoster } })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  Reflect.deleteProperty(window, 'hermesDesktop')
  window.document.documentElement.dataset.hermesDesktopHost = 'browser'
})

async function openMenu(trigger: HTMLElement) {
  await act(async () => {
    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })
    await Promise.resolve()
  })
}

async function expectUnavailable(name: string) {
  // Greyed and not selectable, with the reason in its name, the way the Bots pane names an unavailable bot.
  const row = await screen.findByRole('menuitemradio', { name: `${name}, Not served here` })

  expect(row.getAttribute('aria-disabled')).toBe('true')

  await act(async () => {
    fireEvent.click(row)
    await Promise.resolve()
  })

  expect(selectProfile).not.toHaveBeenCalled()
}

it('the statusbar profile menu shows the profiles its gateway does not serve as unavailable', async () => {
  render(<ProfileSwitcher compact />)
  await openMenu(screen.getByRole('button', { name: 'Profiles: solo' }))

  await expectUnavailable('research')
  await expectUnavailable('default')

  expect(screen.getByRole('menuitemradio', { name: 'solo' }).getAttribute('aria-disabled')).toBeNull()
  expect(getAgentRoster).toHaveBeenCalled()
})

it("the rail's profile menu shows them the same way, and its home pill offers no switch to an unserved default", async () => {
  render(<ProfileRail />)

  // The home pill would switch to default, which this gateway refuses: it toggles the all-profiles view instead.
  fireEvent.click(await screen.findByRole('button', { name: 'All profiles' }))
  expect(screen.queryByRole('button', { name: 'Switch to default' })).toBeNull()
  expect(setShowAllProfiles).toHaveBeenCalledWith(true)

  await openMenu(screen.getByRole('button', { name: 'Profiles' }))
  await expectUnavailable('research')
  expect(screen.getByRole('menuitemradio', { name: 'solo' }).getAttribute('aria-disabled')).toBeNull()
})

it('native Desktop with one gateway reads no roster for the menu and marks nothing', async () => {
  delete window.document.documentElement.dataset.hermesDesktopHost
  render(<ProfileSwitcher compact />)
  await openMenu(screen.getByRole('button', { name: 'Profiles: solo' }))

  const research = await screen.findByRole('menuitemradio', { name: 'research' })

  expect(research.getAttribute('aria-disabled')).toBeNull()
  expect(getAgentRoster).not.toHaveBeenCalled()
})
