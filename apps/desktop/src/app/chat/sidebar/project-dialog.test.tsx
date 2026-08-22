import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type * as Nanostores from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ProjectDialog } from './project-dialog'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel', done: 'Done', save: 'Save' },
      sidebar: {
        projects: {
          addFolder: 'Add folder',
          bindProfilesDesc: (name: string) => `Pick the rail leads while ${name} is active.`,
          bindProfilesNone: 'No named profiles yet.',
          bindProfilesTitle: 'Bind profiles',
          create: 'Create',
          createDesc: 'Create a new project',
          createFailed: 'Failed to create project',
          createTitle: 'New project',
          foldersLabel: 'Folders',
          ideaGenerate: 'Generate',
          ideaGenerating: 'Generating…',
          ideaLabel: 'Idea',
          ideaPlaceholder: 'What are you building?',
          ideaShuffle: 'Shuffle ideas',
          menuBindProfiles: 'Bind profiles…',
          namePlaceholder: 'Project name',
          noFolders: 'No folders yet',
          primaryBadge: 'Primary',
          removeFolder: 'Remove folder'
        }
      }
    }
  })
}))

const { $profiles } = vi.hoisted(() => {
  const { atom } = require('nanostores') as typeof Nanostores

  return {
    $profiles: atom<Array<{ display_name?: null | string; is_default: boolean; name: string }>>([])
  }
})

vi.mock('@/store/profile', () => ({
  $profiles,
  normalizeProfileKey: (name: string) => (name ?? '').trim() || 'default',
  profileLabel: (profile: { display_name?: string; name: string }) =>
    (profile.display_name ?? '').trim() || profile.name
}))

// $projectDialog is a real nanostore atom in the app; recreate it here so
// useStore behaves identically without pulling in the rest of the projects
// store (backend calls, project list, etc.). vi.mock factories are hoisted
// above the rest of the file, so the atoms must be created inside vi.hoisted
// to exist by the time the factory runs.
const { $projectDialog, $workspaceProfileBindings } = vi.hoisted(() => {
  const { atom } = require('nanostores') as typeof Nanostores

  return {
    $projectDialog: atom<{
      mode: 'add-folder' | 'bind-profiles' | 'create' | 'rename'
      name?: string
      projectId?: string
    } | null>({ mode: 'create' }),
    $workspaceProfileBindings: atom<Record<string, string[]>>({})
  }
})

vi.mock('@/store/projects', () => ({
  $projectDialog,
  $workspaceProfileBindings,
  addProjectFolder: vi.fn(),
  bindWorkspaceProfile: vi.fn(),
  closeProjectDialog: vi.fn(),
  createProject: vi.fn(),
  generateProjectIdea: vi.fn(),
  pickProjectFolder: vi.fn(async () => '/Users/test/my-folder'),
  renameProject: vi.fn(),
  unbindWorkspaceProfile: vi.fn()
}))

vi.mock('@/store/notifications', () => ({
  notifyError: vi.fn()
}))

vi.mock('@/lib/project-idea-templates', () => ({
  randomIdeaTemplates: () => [{ emoji: '🚀', idea: 'A rocket tracker', label: 'Rocket tracker' }]
}))

const { bindWorkspaceProfile, unbindWorkspaceProfile } = await import('@/store/projects')

const NAMED_PROFILES = [
  { display_name: null, is_default: true, name: 'default' },
  { display_name: null, is_default: false, name: 'alpha' },
  { display_name: null, is_default: false, name: 'beta' }
]

const tipTrigger = (el: HTMLElement) => el.closest('[data-slot="tooltip-trigger"]')

describe('ProjectDialog', () => {
  beforeEach(() => {
    $profiles.set(NAMED_PROFILES)
    $projectDialog.set({ mode: 'create' })
  })

  afterEach(() => {
    $profiles.set([])
    $projectDialog.set(null)
  })

  it('wraps the "shuffle idea" button in a Tip', () => {
    render(<ProjectDialog />)

    const button = screen.getByRole('button', { name: 'Shuffle ideas' })
    expect(tipTrigger(button)).toBeTruthy()
  })

  it('wraps the "remove folder" button in a Tip once a folder is added', async () => {
    render(<ProjectDialog />)

    fireEvent.click(screen.getByRole('button', { name: 'Add folder' }))

    const button = await screen.findByRole('button', { name: 'Remove folder' })
    expect(tipTrigger(button)).toBeTruthy()
  })
})

describe('ProjectDialog bind-profiles mode (#64221)', () => {
  beforeEach(() => {
    vi.mocked(bindWorkspaceProfile).mockClear()
    vi.mocked(unbindWorkspaceProfile).mockClear()
    $profiles.set(NAMED_PROFILES)
    $workspaceProfileBindings.set({ p1: ['alpha'] })
    $projectDialog.set({ mode: 'bind-profiles', name: 'Finsight', projectId: 'p1' })
  })

  afterEach(() => {
    $workspaceProfileBindings.set({})
    $projectDialog.set(null)
  })

  it('lists named profiles only and marks the bound ones', () => {
    render(<ProjectDialog />)

    // The default profile rides the rail's home pill in every workspace, so
    // binding it would be noise — the picker excludes it.
    expect(screen.queryByRole('button', { name: 'default' })).toBeNull()
    expect(screen.getByRole('button', { name: 'alpha' }).getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByRole('button', { name: 'beta' }).getAttribute('aria-pressed')).toBe('false')
  })

  it('persists each toggle immediately — no save step to forget', () => {
    render(<ProjectDialog />)

    fireEvent.click(screen.getByRole('button', { name: 'beta' }))
    expect(bindWorkspaceProfile).toHaveBeenCalledWith('p1', 'beta')

    fireEvent.click(screen.getByRole('button', { name: 'alpha' }))
    expect(unbindWorkspaceProfile).toHaveBeenCalledWith('p1', 'alpha')

    // The footer only closes; nothing to "save".
    expect(screen.getByRole('button', { name: 'Done' })).toBeTruthy()
  })

  it('says so when there are no named profiles to bind', () => {
    $profiles.set([{ display_name: null, is_default: true, name: 'default' }])
    render(<ProjectDialog />)

    expect(screen.getByText('No named profiles yet.')).toBeTruthy()
  })
})
