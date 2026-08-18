import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ProfileRail } from './profile-switcher'

afterEach(cleanup)

// The rail's discoverability pills are navigation, not identity — assert the
// multi-gateway entry point deep-links to Settings → Connections instead of
// relying on someone finding the pane three levels into Settings (the exact
// gap reported against the multi-connection registry launch).

const navigate = vi.fn()

vi.mock('react-router', () => ({
  useNavigate: () => navigate
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel' },
      profiles: {
        allProfiles: 'All profiles',
        connectGateway: 'Connect another Hermes gateway…',
        failedLoadSoul: 'Failed to load SOUL.md',
        failedSaveSoul: 'Failed to save SOUL.md',
        importProfile: 'Import profile…',
        manageProfiles: 'Manage profiles…',
        newProfile: 'New profile',
        saveSoul: 'Save',
        saving: 'Saving…',
        showAllProfiles: 'Show all profiles',
        soulSaved: 'SOUL.md saved',
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

vi.mock('@/store/profile-share', () => ({
  runExportProfileFlow: vi.fn(),
  runImportProfileFlow: vi.fn()
}))

vi.mock('./use-profile-prewarm', () => ({
  useProfilePrewarm: () => ({ cancelPrewarm: vi.fn(), startPrewarm: vi.fn() })
}))

vi.mock('@/hermes', () => ({
  getProfileSoul: vi.fn().mockResolvedValue({ content: '' }),
  updateProfileSoul: vi.fn()
}))

vi.mock('@/components/chat/code-editor', () => ({ CodeEditor: () => null }))
vi.mock('../../profiles/create-profile-dialog', () => ({ CreateProfileDialog: () => null }))
vi.mock('../../profiles/delete-profile-dialog', () => ({ DeleteProfileDialog: () => null }))
vi.mock('../../profiles/rename-profile-dialog', () => ({ RenameProfileDialog: () => null }))

describe('ProfileRail multi-gateway entry point', () => {
  it('deep-links to the unified Settings → Gateways page from the rail', () => {
    render(<ProfileRail />)

    const pill = screen.getByRole('button', { name: 'Connect another Hermes gateway…' })
    fireEvent.click(pill)

    expect(navigate).toHaveBeenCalledWith('/settings?tab=gateway')
  })

  it('keeps the entry point visible for single-profile users', () => {
    render(<ProfileRail />)

    // The whole point is first-run discoverability: the pill must not be
    // gated behind multiProfile the way the default↔all toggle is.
    expect(screen.getByRole('button', { name: 'Connect another Hermes gateway…' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Manage profiles…' })).toBeTruthy()
  })
})
