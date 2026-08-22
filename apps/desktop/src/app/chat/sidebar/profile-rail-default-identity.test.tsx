import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ProfileRail } from './profile-switcher'

// `hermes profile rename default <Name>` gives the default profile a
// display_name — the rail already puts it in the tooltip, but the pill kept the
// generic home codicon, so the one profile a user lives in was the only one
// without a face (#92033). These pin both halves of the rule: a named default
// gets a mark, an anonymous one keeps home, and ALL stays a view rather than an
// identity.

const { selectProfile, setShowAllProfiles } = vi.hoisted(() => ({
  selectProfile: vi.fn(),
  setShowAllProfiles: vi.fn()
}))

vi.mock('react-router', () => ({
  useNavigate: () => vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel' },
      profiles: {
        allProfiles: 'All profiles',
        connectGateway: 'Manage gateways…',
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
  ALL_PROFILES: '__all__',
  normalizeProfileKey: (name: string) => (name ?? '').trim() || 'default',
  profileLabel: (profile: { display_name?: string; name: string }) =>
    (profile.display_name ?? '').trim() || profile.name,
  profileWearsHomeGlyph: (profile: { display_name?: string; is_default: boolean }) =>
    profile.is_default && (profile.display_name ?? '').trim() === '',
  refreshActiveProfile: vi.fn().mockResolvedValue(undefined),
  selectProfile,
  setProfileColor: vi.fn(),
  setProfileOrder: vi.fn(),
  setShowAllProfiles,
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

interface RailProfile {
  display_name?: string
  is_default: boolean
  name: string
}

const store = await import('@/store/profile')
const profiles = store.$profiles as ReturnType<typeof atom<RailProfile[]>>
const gatewayProfile = store.$activeGatewayProfile as ReturnType<typeof atom<string>>
const scope = store.$profileScope as ReturnType<typeof atom<string>>

const ANONYMOUS_DEFAULT: RailProfile = { is_default: true, name: 'default' }
const NAMED_DEFAULT: RailProfile = { display_name: 'Hermes', is_default: true, name: 'default' }
const WORK: RailProfile = { is_default: false, name: 'work' }

// The default profile is pinned leftmost (profile-switcher.tsx), so position —
// not accessible name — identifies it: the name differs between the
// single-profile and multi-profile branches, and both carry the same rule.
const rail = () => screen.getByRole('group', { name: 'Profiles' })
const defaultSlot = () => within(rail()).getAllByRole('button')[0]

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  profiles.set([ANONYMOUS_DEFAULT])
  gatewayProfile.set('default')
  scope.set('default')
})

describe('ProfileRail default-profile identity (#92033)', () => {
  it('gives a display-renamed default profile a face, not the generic home glyph', () => {
    profiles.set([NAMED_DEFAULT])
    render(<ProfileRail />)

    expect(defaultSlot().querySelector('.codicon-home')).toBeNull()
    expect(defaultSlot().textContent).toBe('H')
  })

  it('keeps the home glyph while the default profile has no name of its own', () => {
    render(<ProfileRail />)

    expect(defaultSlot().querySelector('.codicon-home')).not.toBeNull()
    expect(defaultSlot().textContent).toBe('')
  })

  it('keeps the renamed default pinned left of the named profiles', () => {
    profiles.set([NAMED_DEFAULT, WORK])
    render(<ProfileRail />)

    expect(defaultSlot().querySelector('.codicon-home')).toBeNull()
    expect(defaultSlot().textContent).toBe('H')
    expect(within(rail()).getAllByRole('button')[1]).toBe(screen.getByRole('button', { name: 'work' }))
  })

  it('keeps the layers face while the scope is all profiles', () => {
    profiles.set([NAMED_DEFAULT, WORK])
    scope.set('__all__')
    render(<ProfileRail />)

    // ALL is a view, not a profile — the default's mark must not claim it.
    expect(defaultSlot().querySelector('.codicon-layers')).not.toBeNull()
    expect(defaultSlot().textContent).toBe('')
  })

  it('keeps the scope toggle: on the default profile the pill switches to all profiles', () => {
    profiles.set([NAMED_DEFAULT, WORK])
    render(<ProfileRail />)

    fireEvent.click(defaultSlot())

    expect(setShowAllProfiles).toHaveBeenCalledWith(true)
    expect(selectProfile).not.toHaveBeenCalled()
  })

  it('keeps the scope toggle: from a named profile the pill returns to the default', () => {
    profiles.set([NAMED_DEFAULT, WORK])
    gatewayProfile.set('work')
    render(<ProfileRail />)

    fireEvent.click(defaultSlot())

    // The canonical id routes, never the display name.
    expect(selectProfile).toHaveBeenCalledWith('default')
  })

  it('keeps the mark when the rail condenses to a dropdown', () => {
    // Past the threshold the named profiles collapse into a select, but the
    // default pill sits outside that swap — its face must survive.
    profiles.set([
      NAMED_DEFAULT,
      ...Array.from({ length: 14 }, (_, index) => ({ is_default: false, name: `P${index}` }))
    ])
    render(<ProfileRail />)

    expect(screen.getByRole('button', { name: 'Profiles' })).toBeTruthy()
    expect(defaultSlot().querySelector('.codicon-home')).toBeNull()
    expect(defaultSlot().textContent).toBe('H')
  })

  it('falls back to the placeholder initial for a display name with no letters', () => {
    profiles.set([{ display_name: '🌙', is_default: true, name: 'default' }])
    render(<ProfileRail />)

    expect(defaultSlot().querySelector('.codicon-home')).toBeNull()
    expect(defaultSlot().textContent).toBe('?')
  })

  it('still names the pill by the display name once it carries a mark', () => {
    profiles.set([NAMED_DEFAULT])
    const single = render(<ProfileRail />)

    expect(screen.getByRole('button', { name: 'Hermes' })).toBeTruthy()
    single.unmount()

    profiles.set([NAMED_DEFAULT, WORK])
    gatewayProfile.set('work')
    render(<ProfileRail />)

    expect(screen.getByRole('button', { name: 'Switch to Hermes' })).toBeTruthy()
  })
})
