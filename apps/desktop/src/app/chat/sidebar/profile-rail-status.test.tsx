import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ProfileRail } from './profile-switcher'

// #91710: a profile that finishes work while another is selected must carry a
// durable unread/attention indicator on its rail square (and in the condensed
// dropdown), with the counts in the accessible name. The summary derives from
// the REAL session-unread/session-dot-state stores — only the rail's own shell
// dependencies (routing, profile registry, dialogs) are mocked, so this tests
// the marker → summary → square path end to end.

const navigate = vi.fn()

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
        color: 'Color…',
        colorFor: 'Color',
        connectGateway: 'Manage gateways…',
        editSoul: 'Edit SOUL.md…',
        exportMenu: 'Export profile…',
        failedLoadSoul: 'Failed to load SOUL.md',
        failedSaveSoul: 'Failed to save SOUL.md',
        importProfile: 'Import profile…',
        manageProfiles: 'Manage profiles…',
        newProfile: 'New profile',
        renameMenu: 'Rename…',
        remoteOverride: {
          badge: (host: string) => `Runs on ${host}`,
          menuItem: 'Connect to a remote host…'
        },
        saveSoul: 'Save',
        saving: 'Saving…',
        showAllProfiles: 'Show all profiles',
        soulSaved: 'SOUL.md saved',
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
  $activeGatewayProfile: atom('default'),
  $profileColors: atom({}),
  $profileCreateRequest: atom(0),
  $profileOrder: atom([]),
  $profiles: atom([{ is_default: true, name: 'default' }]),
  $profileScope: atom('default'),
  $showAllProfiles: atom(false),
  ALL_PROFILES: '*',
  normalizeProfileKey: (name: string) => name,
  profileLabel: (profile: { display_name?: string; name: string }) =>
    (profile.display_name ?? '').trim() || profile.name,
  refreshActiveProfile: vi.fn().mockResolvedValue(undefined),
  selectProfile: vi.fn(),
  setProfileColor: vi.fn(),
  setProfileOrder: vi.fn(),
  setShowAllProfiles: vi.fn(),
  sortByProfileOrder: (profiles: unknown[]) => profiles
}))

vi.mock('@/store/connections', () => ({
  $activeConnectionId: atom<null | string>(null),
  $connectionsRegistry: atom(null),
  $hasMultipleConnections: atom(false),
  selectConnection: vi.fn()
}))

vi.mock('@/store/profile-share', () => ({
  runExportProfileFlow: vi.fn(),
  runImportProfileFlow: vi.fn()
}))

vi.mock('./use-profile-prewarm', () => ({
  useProfilePrewarm: () => ({ cancelPrewarm: vi.fn(), notePointerMove: vi.fn(), startPrewarm: vi.fn() })
}))

vi.mock('@/hermes', () => ({
  getProfileSoul: vi.fn().mockResolvedValue({ content: '' }),
  updateProfileSoul: vi.fn()
}))

vi.mock('@/components/chat/code-editor', () => ({ CodeEditor: () => null }))
vi.mock('../../profiles/create-profile-dialog', () => ({ CreateProfileDialog: () => null }))
vi.mock('../../profiles/delete-profile-dialog', () => ({ DeleteProfileDialog: () => null }))
vi.mock('../../profiles/rename-profile-dialog', () => ({ RenameProfileDialog: () => null }))

const { $profiles } = await import('@/store/profile')
const profiles = $profiles as ReturnType<typeof atom<Array<{ is_default: boolean; name: string }>>>
const { $unreadFinishedMarkers } = await import('@/store/session-unread')

afterEach(() => {
  cleanup()
  profiles.set([{ is_default: true, name: 'default' }])
  $unreadFinishedMarkers.set({})
})

describe('ProfileRail per-profile status (#91710)', () => {
  it('marks an inactive profile square unread and names the count accessibly', () => {
    profiles.set([
      { is_default: true, name: 'default' },
      { is_default: false, name: 'writer' }
    ])
    // writer's session finished while default was selected; its rows are not
    // even loaded — the persisted marker is the whole signal.
    $unreadFinishedMarkers.set({ writer: ['s1'] })

    render(<ProfileRail />)

    const writerSquare = screen.getByRole('button', { name: 'writer, 1 unread' })

    expect(writerSquare.querySelector('[data-slot="profile-status-dot"]')).not.toBeNull()
  })

  it('keeps the active profile square clean', () => {
    profiles.set([
      { is_default: true, name: 'default' },
      { is_default: false, name: 'writer' }
    ])
    $unreadFinishedMarkers.set({ writer: ['s1'], default: ['s9'] })

    render(<ProfileRail />)

    // default is the active home: no status dot on its square, no status in
    // its accessible name — the sidebar it owns already shows the row dots.
    // (The default square doubles as the "show all" toggle when already home.)
    const homePill = screen.getByRole('button', { name: 'Show all profiles' })

    expect(homePill.querySelector('[data-slot="profile-status-dot"]')).toBeNull()

    const writerSquare = screen.getByRole('button', { name: 'writer, 1 unread' })

    expect(writerSquare.querySelector('[data-slot="profile-status-dot"]')).not.toBeNull()
  })

  it('leaves every square clean when no profile has unread or live work', () => {
    profiles.set([
      { is_default: true, name: 'default' },
      { is_default: false, name: 'writer' }
    ])

    render(<ProfileRail />)

    expect(screen.getByRole('button', { name: 'writer' })).toBeTruthy()
    // eslint-disable-next-line no-restricted-globals -- whole-rail assertion: no square painted a status dot
    expect(document.querySelector('[data-slot="profile-status-dot"]')).toBeNull()
  })

  it('keeps the indicator in the condensed dropdown beyond the square threshold', async () => {
    // Past 13 squares the rail collapses to a dropdown; the unread state must
    // survive the collapse.
    profiles.set([
      { is_default: true, name: 'default' },
      { is_default: false, name: 'writer' },
      ...Array.from({ length: 13 }, (_, index) => ({ is_default: false, name: `bot-${index}` }))
    ])
    $unreadFinishedMarkers.set({ writer: ['s1'] })

    render(<ProfileRail />)

    const trigger = screen.getByRole('button', { name: 'Profiles' })

    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })
    fireEvent.click(trigger)

    const writerItem = await screen.findByRole('menuitemradio', { name: 'writer, 1 unread' })

    expect(writerItem.querySelector('[data-slot="profile-status-dot"]')).not.toBeNull()
  })
})
