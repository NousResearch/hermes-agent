import { cleanup, render, screen, within } from '@testing-library/react'
import type * as Nanostores from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ProfileRail } from './profile-switcher'

// Workspace-scoped rail filtering (#64221): when the active sidebar project has
// bound profiles, the strip promotes them above an always-present Shared
// section; with no bindings it must stay byte-for-byte today's flat strip.

vi.mock('react-router', () => ({ useNavigate: vi.fn() }))

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
        sharedProfiles: 'Shared',
        showAllProfiles: 'Show all profiles',
        soulSaved: 'SOUL.md saved',
        switchToProfile: (name: string) => `Switch to ${name}`,
        title: 'Profiles'
      }
    }
  })
}))

// Shared atom instances so each test can drive scope + bindings like real
// surfaces would (mirrors the harness in profile-rail-connect.test.tsx).
// vi.mock factories are hoisted above this file, so the atoms must exist by the
// time the '@/store/projects' factory runs — hence vi.hoisted.
const { atom, $projectScope, $workspaceProfileBindings } = vi.hoisted(() => {
  const { atom } = require('nanostores') as typeof Nanostores

  return {
    $projectScope: atom<string>('__all_projects__'),
    $workspaceProfileBindings: atom<Record<string, string[]>>({}),
    atom
  }
})

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: atom('default'),
  $profileColors: atom({}),
  $profileCreateRequest: atom(0),
  $profileOrder: atom([]),
  $profiles: atom([]),
  $profileScope: atom('default'),
  ALL_PROFILES: '*',
  normalizeProfileKey: (name: string) => (name ?? '').trim() || 'default',
  profileLabel: (profile: { display_name?: string; name: string }) =>
    (profile.display_name ?? '').trim() || profile.name,
  refreshActiveProfile: vi.fn().mockResolvedValue(undefined),
  selectProfile: vi.fn(),
  setProfileColor: vi.fn(),
  setProfileOrder: vi.fn(),
  setShowAllProfiles: vi.fn(),
  sortByProfileOrder: (profiles: unknown[]) => profiles
}))

vi.mock('@/store/projects', () => ({
  $projectScope,
  $workspaceProfileBindings
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

const { $profiles } = await import('@/store/profile')
const profilesAtom = $profiles as unknown as Nanostores.WritableAtom<Array<{ is_default: boolean; name: string }>>

const PROFILES = [
  { is_default: true, name: 'default' },
  { is_default: false, name: 'alpha' },
  { is_default: false, name: 'beta' },
  { is_default: false, name: 'gamma' }
]

const railButtonLabels = (): Array<null | string> =>
  within(screen.getByRole('group', { name: 'Profiles' }))
    .queryAllByRole('button')
    .map(button => button.getAttribute('aria-label'))

afterEach(() => {
  cleanup()
  profilesAtom.set([])
  $projectScope.set('__all_projects__')
  $workspaceProfileBindings.set({})
})

describe('ProfileRail workspace bindings (#64221)', () => {
  it('promotes bound profiles and parks the rest under a visible Shared divider', () => {
    profilesAtom.set(PROFILES)
    $projectScope.set('p_finsight')
    // Atom contents are store-sanitized (canonical keys) by the time the rail
    // reads them; the trim/dedupe tolerance is covered in the store suite.
    $workspaceProfileBindings.set({ p_finsight: ['gamma', 'alpha'] })

    const { container } = render(<ProfileRail />)

    // Bound first (rail order), then the divider label, then the shared
    // remainder in rail order.
    expect(railButtonLabels()).toEqual([
      'Show all profiles',
      'alpha',
      'gamma',
      'beta',
      'New profile',
      'Import profile…',
      'Manage profiles…',
      'Manage gateways…'
    ])
    expect(screen.getByText('Shared')).toBeTruthy()
    expect(container.querySelector('[data-slot="profile-rail-shared-divider"]')).toBeTruthy()
  })

  it('keeps the flat strip byte-identical when the workspace has no bindings', () => {
    profilesAtom.set(PROFILES)
    $projectScope.set('p_unbound')

    const { container } = render(<ProfileRail />)

    expect(railButtonLabels()).toEqual([
      'Show all profiles',
      'alpha',
      'beta',
      'gamma',
      'New profile',
      'Import profile…',
      'Manage profiles…',
      'Manage gateways…'
    ])
    expect(screen.queryByText('Shared')).toBeNull()
    expect(container.querySelector('[data-slot="profile-rail-shared-divider"]')).toBeNull()
  })

  it('does not filter outside a concrete workspace (All projects overview)', () => {
    profilesAtom.set(PROFILES)
    $workspaceProfileBindings.set({ p_somewhere: ['alpha'] })

    render(<ProfileRail />)

    expect(railButtonLabels()).toEqual([
      'Show all profiles',
      'alpha',
      'beta',
      'gamma',
      'New profile',
      'Import profile…',
      'Manage profiles…',
      'Manage gateways…'
    ])
    expect(screen.queryByText('Shared')).toBeNull()
  })

  it('never activates filtering off an emptied binding entry', () => {
    profilesAtom.set(PROFILES)
    $projectScope.set('p_stale')

    render(<ProfileRail />)
    $workspaceProfileBindings.set({ p_stale: [] })

    // The sanitized store drops emptied entries on write; even feeding the raw
    // shape through, the resolver must answer null → no Shared section.
    expect(screen.queryByText('Shared')).toBeNull()
  })
})
