import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type * as Nanostores from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ProjectRemoveFolderDialog } from './project-remove-folder-dialog'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel' },
      sidebar: {
        projects: {
          primaryBadge: 'primary',
          removeFolder: 'Remove',
          removeFolderDesc: 'Removes the folder from this project.',
          removeFolderFailed: 'Could not remove folder',
          removeFolderLast: 'A project needs at least one folder.',
          removeFolderNewPrimary: (path: string) => `${path} becomes the primary folder.`,
          removeFolderTitle: 'Remove folder'
        }
      }
    }
  })
}))

const { $projectRemoveFolderDialog, $projects, closeProjectRemoveFolder, removeProjectFolder } = vi.hoisted(() => {
  const { atom } = require('nanostores') as typeof Nanostores

  return {
    $projectRemoveFolderDialog: atom<null | { name: string; projectId: string }>(null),
    $projects: atom<unknown[]>([]),
    closeProjectRemoveFolder: vi.fn(),
    removeProjectFolder: vi.fn()
  }
})

vi.mock('@/store/projects', () => ({
  $projectRemoveFolderDialog,
  $projects,
  closeProjectRemoveFolder,
  nextPrimaryFolder: (folders: { added_at: number; is_primary: boolean; path: string }[], removed: string) =>
    folders.some(f => f.path === removed && f.is_primary)
      ? ([...folders.filter(f => f.path !== removed)].sort((a, b) => a.added_at - b.added_at)[0] ?? null)
      : null,
  removeProjectFolder
}))

const notifyError = vi.fn()
vi.mock('@/store/notifications', () => ({ notifyError: (...args: unknown[]) => notifyError(...args) }))

const folder = (path: string, added_at: number, is_primary = false) => ({ added_at, is_primary, label: null, path })

const seed = (folders: ReturnType<typeof folder>[]) => {
  $projects.set([{ folders, id: 'p_1', name: 'Jinli' }])
  $projectRemoveFolderDialog.set({ name: 'Jinli', projectId: 'p_1' })
}

beforeEach(() => {
  vi.clearAllMocks()
  removeProjectFolder.mockResolvedValue(undefined)
})

describe('ProjectRemoveFolderDialog', () => {
  it('removes a non-primary folder without announcing a new primary', async () => {
    seed([folder('/a', 1, true), folder('/b', 2)])
    render(<ProjectRemoveFolderDialog />)

    fireEvent.click(screen.getByRole('button', { name: 'Remove /b' }))

    expect(screen.queryByText(/becomes the primary folder/)).toBeNull()

    fireEvent.click(screen.getAllByRole('button', { name: 'Remove' }).at(-1)!)

    await waitFor(() => expect(removeProjectFolder).toHaveBeenCalledWith('p_1', '/b'))
    expect(closeProjectRemoveFolder).toHaveBeenCalled()
  })

  it('names the folder that takes over when the primary is removed', () => {
    seed([folder('/a', 1, true), folder('/c', 3), folder('/b', 2)])
    render(<ProjectRemoveFolderDialog />)

    fireEvent.click(screen.getByRole('button', { name: 'Remove /a' }))

    expect(screen.getByText('/b becomes the primary folder.')).toBeTruthy()
    expect(removeProjectFolder).not.toHaveBeenCalled()
  })

  it('refuses to remove the only folder', () => {
    seed([folder('/a', 1, true)])
    render(<ProjectRemoveFolderDialog />)

    expect((screen.getByRole('button', { name: 'Remove /a' }) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getAllByText('A project needs at least one folder.').length).toBeGreaterThan(0)
  })

  it('keeps the dialog open and reports the error when the write fails', async () => {
    removeProjectFolder.mockRejectedValue(new Error('nope'))
    seed([folder('/a', 1, true), folder('/b', 2)])
    render(<ProjectRemoveFolderDialog />)

    fireEvent.click(screen.getByRole('button', { name: 'Remove /b' }))
    fireEvent.click(screen.getAllByRole('button', { name: 'Remove' }).at(-1)!)

    await waitFor(() => expect(notifyError).toHaveBeenCalled())
    expect(closeProjectRemoveFolder).not.toHaveBeenCalled()
  })
})
