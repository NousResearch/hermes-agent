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
          runHere: 'Run on this computer…',
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
const { $pluginRecords } = await import('@/contrib/plugins-store')

afterEach(() => {
  cleanup()
  profiles.set([{ is_default: true, name: 'default' }])
  $profileRemoteOverrides.set({})
  $pluginRecords.set({})
})

// The plugin that runs `work`. No plugin is loaded unless a case says so.
const PLUGIN = 'acme'

function renderRail() {
  profiles.set([
    { is_default: true, name: 'default' },
    { is_default: false, name: 'research' },
    { is_default: false, name: 'work' }
  ])
  $profileRemoteOverrides.set({
    research: { host: 'box.example:9119', url: 'http://box.example:9119' },
    work: { host: '127.0.0.1:47911', launcher: { plugin: PLUGIN }, url: 'http://127.0.0.1:47911' }
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

  // The route outlives its plugin: connection.json keeps the launcher entry when the
  // plugin is removed, disabled or fails to load, and only that plugin's page calls
  // host.clearProfileBackend. Hiding the remote-host dialog then leaves the person no control at all.
  it('offers a way back to this computer on a sandboxed profile, so a removed plugin does not strand it', async () => {
    const clearProfileBackend = vi.fn(async () => ({ applied: true }))

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { clearProfileBackend }

    try {
      renderRail()
      fireEvent.contextMenu(screen.getByRole('button', { name: 'work — Sandboxed' }))
      expect(await screen.findByText('Rename…')).toBeTruthy()

      const items = screen.getAllByRole('menuitem').map(item => item.textContent ?? '')

      // Whatever it is called: one entry that hands the profile back (main's clear already asks natively).
      expect(items.filter(text => /this computer|switch back|sandbox|run locally/i.test(text))).not.toEqual([])
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('that entry asks the main process to give the profile back, for the plugin its route names', async () => {
    const bridge = {
      clearProfileBackend: vi.fn(async () => ({ applied: true })),
      confirmClearProfileBackend: vi.fn(),
      confirmSetProfileBackend: vi.fn(),
      // What the main process answers for each profile's route until the clear: `work` is the plugin's.
      getConnectionConfig: vi.fn(async (name: string) =>
        name === 'work'
          ? { launcher: { plugin: PLUGIN }, mode: 'remote', remoteUrl: 'http://127.0.0.1:47911' }
          : { launcher: null, mode: 'local', remoteUrl: '' }
      ),
      setProfileBackend: vi.fn()
    }

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = bridge

    try {
      renderRail()
      fireEvent.contextMenu(screen.getByRole('button', { name: 'work — Sandboxed' }))
      fireEvent.click(await screen.findByText('Run on this computer…'))

      // Without a grant: the main process asks the person itself before it changes anything.
      await vi.waitFor(() => expect(bridge.clearProfileBackend).toHaveBeenCalledTimes(1))
      expect(bridge.clearProfileBackend.mock.calls[0]).toEqual([{ plugin: PLUGIN, profile: 'work' }])
      expect(bridge.confirmClearProfileBackend).not.toHaveBeenCalled()
    } finally {
      delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('a profile whose plugin is loaded has no such entry: the plugin page stays the way to switch', async () => {
    $pluginRecords.set({ [PLUGIN]: { id: PLUGIN, kind: 'disk', name: 'Sandbox', status: 'loaded' } })
    renderRail()
    fireEvent.contextMenu(screen.getByRole('button', { name: 'work — Sandboxed' }))
    expect(await screen.findByText('Rename…')).toBeTruthy()
    expect(screen.queryByText('Run on this computer…')).toBeNull()

    // A plugin that failed to load, or is switched off, is not running the profile's page either.
    cleanup()
    $pluginRecords.set({ [PLUGIN]: { id: PLUGIN, kind: 'disk', name: 'Sandbox', status: 'error' } })
    renderRail()
    fireEvent.contextMenu(screen.getByRole('button', { name: 'work — Sandboxed' }))
    expect(await screen.findByText('Run on this computer…')).toBeTruthy()
  })
})
