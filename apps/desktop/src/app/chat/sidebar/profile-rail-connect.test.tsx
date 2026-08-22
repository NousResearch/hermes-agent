import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ProfileRail } from './profile-switcher'

// Gateway identity and management live in the dedicated row above this rail;
// these tests keep the profile controls independent as profile counts change.

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

const { $profiles } = await import('@/store/profile')
const profiles = $profiles as ReturnType<typeof atom<Array<{ is_default: boolean; name: string }>>>

afterEach(() => {
  cleanup()
  profiles.set([{ is_default: true, name: 'default' }])
})

describe('ProfileRail', () => {
  it('keeps gateway management out of the separate profile controls', () => {
    render(<ProfileRail />)

    expect(screen.queryByRole('button', { name: 'Manage gateways…' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Manage profiles…' })).toBeTruthy()
  })

  it('keeps the active profile explicit beside the separate gateway row', () => {
    render(<ProfileRail />)

    expect(screen.getByRole('button', { name: 'default' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Manage gateways…' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Manage profiles…' })).toBeTruthy()
  })

  it('keeps thirteen profiles direct and condenses the fourteenth', () => {
    profiles.set([
      { is_default: true, name: 'default' },
      ...Array.from({ length: 12 }, (_, index) => ({ is_default: false, name: `Profile ${index + 1}` }))
    ])
    const { unmount } = render(<ProfileRail />)

    expect(screen.queryByRole('button', { name: 'Profiles' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Profile 12' })).toBeTruthy()
    unmount()

    profiles.set([
      { is_default: true, name: 'default' },
      ...Array.from({ length: 13 }, (_, index) => ({ is_default: false, name: `Profile ${index + 1}` }))
    ])
    render(<ProfileRail />)

    expect(screen.getByRole('button', { name: 'Profiles' })).toBeTruthy()
  })

  it('stays shrinkable with many profiles', () => {
    profiles.set([
      { is_default: true, name: 'default' },
      ...Array.from({ length: 13 }, (_, index) => ({ is_default: false, name: `Profile ${index + 1}` }))
    ])
    render(<ProfileRail />)

    expect(screen.getByRole('group', { name: 'Profiles' }).className).toContain('min-w-0')
    expect(screen.getByRole('button', { name: 'Profiles' })).toBeTruthy()
  })
})
