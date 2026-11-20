import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ProfileRail } from './profile-switcher'

// A profile a desktop plugin's launcher runs (a sandbox on this computer) is
// not a remote host the user connected: its square says Sandboxed, not
// "Runs on 127.0.0.1:<port>", and it never offers the remote-host dialog,
// which would overwrite the plugin's entry.

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
          menuItem: 'Connect to a remote host…',
          sandboxed: 'Sandboxed'
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
  prewarmProfilePick: vi.fn(),
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
const { $profileRemoteOverrides } = await import('@/store/profile-remote-override')

afterEach(() => {
  cleanup()
  profiles.set([{ is_default: true, name: 'default' }])
  $profileRemoteOverrides.set({})
})

function renderRail() {
  profiles.set([
    { is_default: true, name: 'default' },
    { is_default: false, name: 'research' },
    { is_default: false, name: 'work' }
  ])
  $profileRemoteOverrides.set({
    research: { host: 'box.example:9119', url: 'http://box.example:9119' },
    work: { host: '127.0.0.1:47911', launcher: { plugin: 'openshell' }, url: 'http://127.0.0.1:47911' }
  })

  render(<ProfileRail />)
}

describe('ProfileRail sandboxed profiles', () => {
  it('labels a launched profile Sandboxed and a remote one by its host', () => {
    renderRail()

    const work = screen.getByRole('button', { name: 'work — Sandboxed' })

    expect(work.querySelector('[data-slot="profile-sandboxed-badge"]')).not.toBeNull()
    expect(work.querySelector('[data-slot="profile-remote-badge"]')).toBeNull()
    expect(screen.getByRole('button', { name: 'research — Runs on box.example:9119' })).toBeTruthy()
    expect(screen.queryByText(/127\.0\.0\.1/)).toBeNull()
  })

  it('offers the remote-host dialog on a remote profile but not on a sandboxed one', async () => {
    renderRail()

    fireEvent.contextMenu(screen.getByRole('button', { name: 'research — Runs on box.example:9119' }))
    expect(await screen.findByText('Runs on box.example:9119')).toBeTruthy()
    cleanup()

    renderRail()
    fireEvent.contextMenu(screen.getByRole('button', { name: 'work — Sandboxed' }))
    expect(await screen.findByText('Rename…')).toBeTruthy()
    expect(screen.queryByText('Connect to a remote host…')).toBeNull()
    expect(screen.queryByText('Sandboxed')).toBeNull()
  })
})
