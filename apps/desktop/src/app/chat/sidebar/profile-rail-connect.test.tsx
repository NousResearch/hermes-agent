import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ProfileRail } from './profile-switcher'

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
        clearDefaultProfile: 'Clear default',
        connectGateway: 'Manage gateways…',
        editSoul: 'Edit SOUL.md…',
        failedLoadSoul: 'Failed to load SOUL.md',
        failedSaveSoul: 'Failed to save SOUL.md',
        importProfile: 'Import profile…',
        manageProfiles: 'Manage profiles…',
        newProfile: 'New profile',
        saveSoul: 'Save',
        saving: 'Saving…',
        setDefaultProfile: 'Set as default',
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
  $homeProfile: atom(null),
  $profileColors: atom({}),
  $profileCreateRequest: atom(0),
  $profileGlyphs: atom<Record<string, string>>({}),
  $profileOrder: atom([]),
  $profiles: atom([{ is_default: true, name: 'default' }]),
  $profileScope: atom('default'),
  ALL_PROFILES: '*',
  // Same contract as store/profile's resolver, inlined so these tests exercise
  // the rail's wiring against it (the store tests own the resolver itself).
  designatedHomeProfile: (home: null | string, profiles: Array<{ is_default?: boolean; name: string }>) =>
    home ? (profiles.find(profile => profile.name === home) ?? null) : null,
  normalizeProfileKey: (name: string) => name,
  profileLabel: (profile: { display_name?: string; name: string }) =>
    (profile.display_name ?? '').trim() || profile.name,
  refreshActiveProfile: vi.fn().mockResolvedValue(undefined),
  resolveHomeProfile: (home: null | string, profiles: Array<{ is_default?: boolean; name: string }>) =>
    (home ? profiles.find(profile => profile.name === home) : undefined) ??
    profiles.find(profile => profile.is_default) ??
    null,
  selectProfile: vi.fn(),
  setHomeProfile: vi.fn(),
  setProfileColor: vi.fn(),
  setProfileOrder: vi.fn(),
  setShowAllProfiles: vi.fn(),
  sortByProfileOrder: (profiles: unknown[]) => profiles
}))

vi.mock('@/store/connections', () => ({ $hasMultipleConnections: atom(false) }))

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

const { $hasMultipleConnections } = await import('@/store/connections')
const hasMultipleConnections = $hasMultipleConnections as ReturnType<typeof atom<boolean>>

const { getProfileSoul } = await import('@/hermes')

const {
  $activeGatewayProfile,
  $homeProfile,
  $profileGlyphs,
  $profiles,
  selectProfile,
  setHomeProfile,
  setShowAllProfiles
} = await import('@/store/profile')

const profiles = $profiles as ReturnType<typeof atom<Array<{ is_default: boolean; name: string }>>>
const activeGatewayProfile = $activeGatewayProfile as ReturnType<typeof atom<string>>
const homeProfile = $homeProfile as ReturnType<typeof atom<null | string>>
const glyphs = $profileGlyphs as ReturnType<typeof atom<Record<string, string>>>

afterEach(() => {
  cleanup()
  hasMultipleConnections.set(false)
  profiles.set([{ is_default: true, name: 'default' }])
  activeGatewayProfile.set('default')
  homeProfile.set(null)
  glyphs.set({})
  vi.mocked(selectProfile).mockClear()
  vi.mocked(setHomeProfile).mockClear()
  vi.mocked(setShowAllProfiles).mockClear()
})

describe('ProfileRail multi-gateway entry point', () => {
  it('deep-links to the unified Settings → Gateways page from the rail', () => {
    render(<ProfileRail />)

    const pill = screen.getByRole('button', { name: 'Manage gateways…' })
    fireEvent.click(pill)

    expect(navigate).toHaveBeenCalledWith('/settings?tab=gateway')
  })

  it('keeps the entry point visible for single-profile users', () => {
    render(<ProfileRail />)

    // The whole point is first-run discoverability: the pill must not be
    // gated behind multiProfile the way the default↔all toggle is.
    expect(screen.getByRole('button', { name: 'Manage gateways…' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Manage profiles…' })).toBeTruthy()
  })

  it('keeps the active profile explicit when gateway identity moves to the statusbar', () => {
    hasMultipleConnections.set(true)
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

  it('stays shrinkable with many profiles and multiple gateways', () => {
    hasMultipleConnections.set(true)
    profiles.set([
      { is_default: true, name: 'default' },
      ...Array.from({ length: 13 }, (_, index) => ({ is_default: false, name: `Profile ${index + 1}` }))
    ])
    render(<ProfileRail />)

    expect(screen.getByRole('group', { name: 'Profiles' }).className).toContain('min-w-0')
    expect(screen.getByRole('button', { name: 'Profiles' })).toBeTruthy()
  })
})

describe('designated home pill (#89887)', () => {
  const twoProfiles = () =>
    profiles.set([
      { is_default: true, name: 'default' },
      { is_default: false, name: 'work' }
    ])

  it('navigates to the designated home instead of toggling Show all', () => {
    twoProfiles()
    homeProfile.set('work')
    render(<ProfileRail />)

    fireEvent.click(screen.getByRole('button', { name: 'Switch to work' }))

    expect(selectProfile).toHaveBeenCalledWith('work')
    expect(setShowAllProfiles).not.toHaveBeenCalled()
  })

  it('never falls back to the Show-all toggle even when already on home', () => {
    twoProfiles()
    homeProfile.set('work')
    activeGatewayProfile.set('work')
    render(<ProfileRail />)

    // Today's bug shape: sitting on the pill's target flipped it into a
    // Show-all toggle. Designated-home mode is always a go-home navigation.
    fireEvent.click(screen.getByRole('button', { name: 'Switch to work' }))

    expect(selectProfile).toHaveBeenCalledWith('work')
    expect(setShowAllProfiles).not.toHaveBeenCalled()
  })

  it('an unset preference keeps the classic default↔all toggle', () => {
    twoProfiles()
    render(<ProfileRail />)

    // No designation → today's behavior verbatim: on default, the same
    // control offers Show all.
    fireEvent.click(screen.getByRole('button', { name: 'Show all profiles' }))

    expect(setShowAllProfiles).toHaveBeenCalledWith(true)
    expect(selectProfile).not.toHaveBeenCalled()
  })

  it('a stale designation behaves exactly like an unset one', () => {
    twoProfiles()
    homeProfile.set('ghost') // deleted elsewhere; the rail must not dead-end
    render(<ProfileRail />)

    fireEvent.click(screen.getByRole('button', { name: 'Show all profiles' }))

    expect(setShowAllProfiles).toHaveBeenCalledWith(true)
    expect(selectProfile).not.toHaveBeenCalled()
  })

  it('right-click designates a square as home, then offers clearing it', () => {
    twoProfiles()
    render(<ProfileRail />)

    fireEvent.contextMenu(screen.getByRole('button', { name: 'work' }))
    fireEvent.click(screen.getByText('Set as default'))

    expect(setHomeProfile).toHaveBeenCalledWith('work')

    cleanup()
    homeProfile.set('work')
    render(<ProfileRail />)

    fireEvent.contextMenu(screen.getByRole('button', { name: 'work' }))
    fireEvent.click(screen.getByText('Clear default'))

    expect(setHomeProfile).toHaveBeenLastCalledWith(null)
  })

  it('the designated home pill itself clears via its context menu', () => {
    twoProfiles()
    homeProfile.set('work')
    render(<ProfileRail />)

    fireEvent.contextMenu(screen.getByRole('button', { name: 'Switch to work' }))
    fireEvent.click(screen.getByText('Clear default'))

    expect(setHomeProfile).toHaveBeenCalledWith(null)
  })
})

describe('rail glyph overrides and edit entry (#79233)', () => {
  const twoProfiles = () =>
    profiles.set([
      { is_default: true, name: 'default' },
      { is_default: false, name: 'work' }
    ])

  it('the designated home pill renders its own overridden glyph', () => {
    twoProfiles()
    homeProfile.set('work')
    glyphs.set({ work: 'briefcase' })
    render(<ProfileRail />)

    const icon = screen
      .getByRole('button', { name: 'Switch to work' })
      .querySelector('.codicon')

    expect(icon?.className).toContain('codicon-briefcase')
  })

  it('an unset override keeps the home mark on the pill and squares', () => {
    twoProfiles()
    homeProfile.set('work')
    render(<ProfileRail />)

    const pillIcon = screen
      .getByRole('button', { name: 'Switch to work' })
      .querySelector('.codicon')

    expect(pillIcon?.className).toContain('codicon-home')

    // The named square still carries its initial (no codicon inside).
    const square = screen.getByRole('button', { name: 'work' })

    expect(square.querySelector('.codicon')).toBeNull()
    expect(square.textContent).toBe('w')
  })

  it('a named square paints its override instead of the initial', () => {
    twoProfiles()
    glyphs.set({ work: 'rocket' })
    render(<ProfileRail />)

    const icon = screen.getByRole('button', { name: 'work' }).querySelector('.codicon')

    expect(icon?.className).toContain('codicon-rocket')
  })

  it('the designated home pill menu offers editing its SOUL.md', () => {
    twoProfiles()
    homeProfile.set('work')
    render(<ProfileRail />)

    fireEvent.contextMenu(screen.getByRole('button', { name: 'Switch to work' }))

    // Same composition as the named squares' menu — the pill is a full
    // profile surface now, not just a navigation toggle.
    fireEvent.click(screen.getByText('Edit SOUL.md…'))

    return waitFor(() => expect(vi.mocked(getProfileSoul)).toHaveBeenCalledWith('work'))
  })

  it('the classic toggle pill offers editing the built-in default SOUL.md too', () => {
    twoProfiles()
    render(<ProfileRail />)

    fireEvent.contextMenu(screen.getByRole('button', { name: 'Show all profiles' }))
    fireEvent.click(screen.getByText('Edit SOUL.md…'))

    return waitFor(() => expect(vi.mocked(getProfileSoul)).toHaveBeenCalledWith('default'))
  })
})
