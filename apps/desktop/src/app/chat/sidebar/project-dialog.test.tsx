import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { JsonRpcGatewayError } from '@hermes/shared'
import type * as Nanostores from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ProjectDialog } from './project-dialog'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel', close: 'Close', save: 'Save' },
      sidebar: {
        projects: {
          addFolder: 'Add folder',
          create: 'Create project',
          createDesc: 'Create a new project',
          createFailed: 'Failed to create project',
          createTitle: 'New project',
          foldersLabel: 'Folders',
          ideaGenerate: 'Generate',
          ideaGenerating: 'Generating…',
          ideaLabel: 'Idea',
          ideaPlaceholder: 'What are you building?',
          ideaShuffle: 'Shuffle ideas',
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
const { $projectDialog, createProject, pickProjectFolder } = vi.hoisted(() => {
  const { atom } = require('nanostores') as typeof Nanostores

  return {
    $projectDialog: atom<{ mode: 'create' | 'rename' | 'add-folder'; name?: string; projectId?: string } | null>({
      mode: 'create'
    }),
    createProject: vi.fn(),
    pickProjectFolder: vi.fn(async () => '/home/hermes/.hermes/projects/agent-from-scratch')
  }
})

vi.mock('@/store/projects', () => ({
  $projectDialog,
  addProjectFolder: vi.fn(),
  closeProjectDialog: vi.fn(),
  createProject,
  generateProjectIdea: vi.fn(),
  pickProjectFolder,
  renameProject: vi.fn()
}))

vi.mock('@/lib/project-idea-templates', () => ({
  randomIdeaTemplates: () => [{ emoji: '🚀', idea: 'A rocket tracker', label: 'Rocket tracker' }]
}))

const tipTrigger = (el: HTMLElement) => el.closest('[data-slot="tooltip-trigger"]')

describe('ProjectDialog', () => {
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

  it('keeps the sheet open and shows the server RPC message when create fails', async () => {
    createProject.mockRejectedValueOnce(
      new JsonRpcGatewayError(
        "folder already belongs to project 'agent-from-scratch' (p_abc); switch to it instead of creating a duplicate",
        { code: 5063 }
      )
    )

    render(<ProjectDialog />)

    fireEvent.change(screen.getByPlaceholderText('Project name'), {
      target: { value: 'agent-from-scratch' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Add folder' }))
    await screen.findByRole('button', { name: 'Remove folder' })
    fireEvent.click(screen.getByRole('button', { name: 'Create project' }))

    expect(
      await screen.findByText(
        "folder already belongs to project 'agent-from-scratch' (p_abc); switch to it instead of creating a duplicate"
      )
    ).toBeTruthy()
    expect(screen.queryByText(/Hermes RPC request failed \(5063\)/)).toBeNull()
    expect(screen.getByRole('button', { name: 'Create project' })).toBeTruthy()
  })

  it('shows the opaque code fallback only when the gateway omitted a message', async () => {
    createProject.mockRejectedValueOnce(new JsonRpcGatewayError('', { code: 5063 }))

    render(<ProjectDialog />)

    fireEvent.change(screen.getByPlaceholderText('Project name'), {
      target: { value: 'agent-from-scratch' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Add folder' }))
    await screen.findByRole('button', { name: 'Remove folder' })
    fireEvent.click(screen.getByRole('button', { name: 'Create project' }))

    await waitFor(() => {
      expect(screen.getByText('Hermes RPC request failed (5063)')).toBeTruthy()
    })
  })
})
