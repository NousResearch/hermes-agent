import type * as DndKitSortable from '@dnd-kit/sortable'
import { cleanup, render } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ProfileRail } from './profile-switcher'

// A profile square's sortable identity must track the canonical profile
// name, not its display label -- SortableContext and handleDragEnd are both
// keyed by name (profile-switcher.tsx), so a square registered under its
// label falls out of that id space the moment display_name differs from
// name, and dragging it silently does nothing (#100480).

const sortableIds: string[] = []

vi.mock('@dnd-kit/sortable', async () => {
  const actual = await vi.importActual<typeof DndKitSortable>('@dnd-kit/sortable')

  return {
    ...actual,
    useSortable: (args: Parameters<typeof actual.useSortable>[0]) => {
      sortableIds.push(String(args.id))

      return actual.useSortable(args)
    }
  }
})

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
        remoteOverride: {
          badge: (host: string) => `Runs on ${host}`,
          menuItem: 'Connect to a remote host…'
        },
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
  $profiles: atom([
    { is_default: true, name: 'default' },
    { display_name: '薄荷', is_default: false, name: 'assistant' }
  ]),
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

vi.mock('@/store/connections', () => ({
  $activeConnectionId: atom(null),
  $connectionsRegistry: atom(null),
  $hasMultipleConnections: atom(false),
  selectConnection: vi.fn()
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

afterEach(() => {
  cleanup()
  sortableIds.length = 0
})

describe('ProfileRail drag-to-reorder identity', () => {
  it('registers a named profile square under its canonical name, not its display label', () => {
    render(<ProfileRail />)

    expect(sortableIds).toContain('assistant')
    expect(sortableIds).not.toContain('薄荷')
  })
})
