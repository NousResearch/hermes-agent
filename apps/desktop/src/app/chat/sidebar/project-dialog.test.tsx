import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type * as Nanostores from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { notify, notifyError } from '@/store/notifications'
import { closeProjectDialog, createProject, moveSessionToProject } from '@/store/projects'

import { ProjectDialog } from './project-dialog'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel', save: 'Save' },
      sidebar: {
        projects: {
          addFolder: 'Add folder',
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
          moveFailed: 'Could not move session',
          movedTo: (name: string) => `Moved to ${name}`,
          namePlaceholder: 'Project name',
          noFolders: 'No folders yet',
          primaryBadge: 'Primary',
          removeFolder: 'Remove folder'
        }
      }
    }
  })
}))

// $projectDialog is a real nanostore atom in the app; recreate it here so
// useStore behaves identically without pulling in the rest of the projects
// store (backend calls, project list, etc.) which is irrelevant to the Tip fix.
// vi.mock factories are hoisted above the rest of the file, so the atom must
// be created inside vi.hoisted to exist by the time the factory runs.
const { $projectDialog } = vi.hoisted(() => {
  const { atom } = require('nanostores') as typeof Nanostores

  return {
    $projectDialog: atom<{
      mode: 'create' | 'rename' | 'add-folder'
      name?: string
      projectId?: string
      moveSessionAfter?: { profile?: null | string; sessionId: string }
    } | null>({
      mode: 'create'
    })
  }
})

vi.mock('@/store/projects', () => ({
  $projectDialog,
  addProjectFolder: vi.fn(),
  closeProjectDialog: vi.fn(),
  createProject: vi.fn(),
  generateProjectIdea: vi.fn(),
  moveSessionToProject: vi.fn(),
  pickProjectFolder: vi.fn(async () => '/Users/test/my-folder'),
  renameProject: vi.fn()
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/lib/project-idea-templates', () => ({
  randomIdeaTemplates: () => [{ emoji: '🚀', idea: 'A rocket tracker', label: 'Rocket tracker' }]
}))

const tipTrigger = (el: HTMLElement) => el.closest('[data-slot="tooltip-trigger"]')

describe('ProjectDialog', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(createProject).mockResolvedValue(null)
    vi.mocked(moveSessionToProject).mockResolvedValue(undefined)
    $projectDialog.set({ mode: 'create' })
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

  // Fill the create form (name + one folder via the mocked picker) and submit.
  async function submitCreate() {
    render(<ProjectDialog />)

    fireEvent.change(screen.getByPlaceholderText('Project name'), { target: { value: 'Demo' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add folder' }))
    // The picker resolves async; wait for the folder to land so Create enables.
    await screen.findByRole('button', { name: 'Remove folder' })
    fireEvent.click(screen.getByRole('button', { name: 'Create' }))
  }

  it('creates with use: true when not opened from the move menu', async () => {
    await submitCreate()

    await waitFor(() =>
      expect(createProject).toHaveBeenCalledWith({
        folders: ['/Users/test/my-folder'],
        idea: undefined,
        name: 'Demo',
        use: true
      })
    )
    expect(moveSessionToProject).not.toHaveBeenCalled()
    expect(closeProjectDialog).toHaveBeenCalled()
  })

  it('creates with use: false and moves the session into the new project', async () => {
    vi.mocked(createProject).mockResolvedValue({ id: 'p_new', name: 'Demo' } as never)
    $projectDialog.set({
      mode: 'create',
      moveSessionAfter: { profile: 'work', sessionId: 'sess-1' }
    })

    await submitCreate()

    await waitFor(() =>
      expect(createProject).toHaveBeenCalledWith({
        folders: ['/Users/test/my-folder'],
        idea: undefined,
        name: 'Demo',
        use: false
      })
    )
    expect(moveSessionToProject).toHaveBeenCalledWith('sess-1', 'p_new', 'work')
    expect(notify).toHaveBeenCalledWith({ durationMs: 2_000, kind: 'success', message: 'Moved to Demo' })
    expect(closeProjectDialog).toHaveBeenCalled()
  })

  it('keeps the created project and closes when the move fails', async () => {
    vi.mocked(createProject).mockResolvedValue({ id: 'p_new', name: 'Demo' } as never)
    vi.mocked(moveSessionToProject).mockRejectedValue(new Error('move boom'))
    $projectDialog.set({ mode: 'create', moveSessionAfter: { sessionId: 'sess-1' } })

    await submitCreate()

    await waitFor(() => expect(moveSessionToProject).toHaveBeenCalled())
    expect(createProject).toHaveBeenCalled()
    expect(notifyError).toHaveBeenCalledWith(expect.any(Error), 'Could not move session')
    expect(closeProjectDialog).toHaveBeenCalled()
  })
})
