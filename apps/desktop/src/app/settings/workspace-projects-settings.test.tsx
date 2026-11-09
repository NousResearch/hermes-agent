import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setPrimaryGateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import type * as ProjectsStore from '@/store/projects'
import type { ProjectInfo } from '@/types/hermes'

import { WorkspaceProjectsSettings } from './workspace-projects-settings'

// The panel reads the projects store atoms and drives every mutation through
// the store's action functions — the one mounted sidebar ProjectDialog owns
// create/rename/add-folder via the $projectDialog atom, so the panel must only
// arm that atom, never mount a dialog of its own.
vi.mock('@/store/projects', async () => {
  const actual = await vi.importActual<typeof ProjectsStore>('@/store/projects')

  return {
    ...actual,
    deleteProject: vi.fn(),
    openProjectAddFolder: vi.fn(),
    openProjectCreate: vi.fn(),
    openProjectRename: vi.fn(),
    removeProjectFolder: vi.fn(),
    setDefaultProject: vi.fn(),
    setProjectAppearance: vi.fn(),
    setProjectFolderPrimary: vi.fn()
  }
})

const projectsMock = vi.mocked(await import('@/store/projects'))

// The panel's i18n surface is the shared projects copy plus a few common keys.
vi.mock('@/i18n', () => ({
  translateNow: (key: string) => key,
  useI18n: () => ({
    t: {
      common: {
        cancel: 'Cancel',
        failed: 'Something went wrong',
        set: 'Set'
      },
      settings: {
        subpages: { workspaceProjects: 'Projects' }
      },
      sidebar: {
        projects: {
          addFolder: 'Add folder',
          defaultForNewChats: 'Default for new chats',
          deleteConfirm: 'Files and repos stay untouched.',
          menuAppearance: 'Appearance',
          menuDelete: 'Delete',
          menuRename: 'Rename…',
          newButton: 'New project',
          noColor: 'No color',
          noProjects: 'No projects yet',
          primaryBadge: 'primary',
          removeFolder: 'Remove',
          removeFolderConfirm: 'Remove this folder from the project?',
          removeFolderFailed: 'Could not remove folder',
          setDefaultFailed: 'Could not set the default project',
          setPrimaryFailed: 'Could not set the primary folder',
          staleBackend: 'Update the Hermes backend'
        }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

function project(patch: Partial<ProjectInfo>): ProjectInfo {
  return {
    archived: false,
    board_slug: null,
    color: null,
    created_at: 0,
    description: null,
    folders: [],
    icon: null,
    id: 'p_x',
    name: 'Project',
    primary_path: null,
    slug: 'x',
    ...patch
  }
}

beforeEach(() => {
  projectsMock.$projects.set([
    project({
      color: '#f00',
      folders: [
        { added_at: 0, is_primary: true, label: null, path: '/work/app' },
        { added_at: 0, is_primary: false, label: null, path: '/work/app-docs' }
      ],
      icon: 'rocket',
      id: 'p_one',
      name: 'Apollo',
      primary_path: '/work/app'
    }),
    project({
      folders: [{ added_at: 0, is_primary: true, label: null, path: '/work/seo' }],
      id: 'p_two',
      name: 'Borealis',
      primary_path: '/work/seo'
    })
  ])
  projectsMock.$defaultProjectId.set(null)
  projectsMock.$projectsRpcAvailable.set(true)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('WorkspaceProjectsSettings', () => {
  it('lists saved non-archived projects with icon, color, name and folder paths', () => {
    render(<WorkspaceProjectsSettings />)

    const apollo = screen.getByRole('listitem', { name: 'Apollo' })

    expect(apollo).toBeTruthy()
    expect(apollo.querySelector('[data-testid="project-glyph"]')).toBeTruthy()
    expect(apollo.textContent).toContain('/work/app')
    expect(apollo.textContent).toContain('/work/app-docs')
    expect(screen.getByRole('listitem', { name: 'Borealis' })).toBeTruthy()
  })

  it('filters archived projects out of the list', () => {
    projectsMock.$projects.set([...projectsMock.$projects.get(), project({ archived: true, id: 'p_old', name: 'Old' })])

    render(<WorkspaceProjectsSettings />)

    expect(screen.queryByRole('listitem', { name: 'Old' })).toBeNull()
  })

  it('shows the default marker on the default project only', () => {
    projectsMock.$defaultProjectId.set('p_two')

    render(<WorkspaceProjectsSettings />)

    expect(screen.getByRole('listitem', { name: 'Borealis' }).textContent).toContain('Default for new chats')
    expect(screen.getByRole('listitem', { name: 'Apollo' }).textContent).not.toContain('Default for new chats')
  })

  it('renders an empty state with a create affordance when there are no projects', () => {
    projectsMock.$projects.set([])

    render(<WorkspaceProjectsSettings />)

    expect(screen.getByText('No projects yet')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'New project' }))
    expect(projectsMock.openProjectCreate).toHaveBeenCalledTimes(1)
  })

  it('surfaces a stale-backend note instead of project rows when projects RPC is unavailable', () => {
    projectsMock.$projectsRpcAvailable.set(false)

    render(<WorkspaceProjectsSettings />)

    expect(screen.getByText('Update the Hermes backend')).toBeTruthy()
    expect(screen.queryByRole('list')).toBeNull()
  })

  it('arms the shared dialog atom for create, rename and add-folder instead of mounting a dialog', () => {
    render(<WorkspaceProjectsSettings />)

    fireEvent.click(screen.getAllByRole('button', { name: 'Rename…' })[0])
    expect(projectsMock.openProjectRename).toHaveBeenCalledWith({ id: 'p_one', name: 'Apollo' })

    fireEvent.click(screen.getAllByRole('button', { name: 'Add folder' })[0])
    expect(projectsMock.openProjectAddFolder).toHaveBeenCalledWith({ id: 'p_one', name: 'Apollo' })

    // The panel never mounts its own ProjectDialog — the single sidebar mount
    // owns every flow through the atom.
    expect(projectsMock.$projectDialog.get()).toBeNull()
  })

  it('deletes a project through the store after the destructive confirm', async () => {
    projectsMock.deleteProject.mockResolvedValue(undefined)

    render(<WorkspaceProjectsSettings />)

    fireEvent.click(screen.getAllByRole('button', { name: 'Delete' })[0])

    const dialog = await screen.findByRole('dialog')

    expect(dialog.textContent).toContain('Files and repos stay untouched.')

    fireEvent.click(within(dialog as HTMLElement).getByRole('button', { name: 'Delete' }))

    await waitFor(() => expect(projectsMock.deleteProject).toHaveBeenCalledWith('p_one'))
  })

  it('surfaces a delete failure with notifyError and never touches files', async () => {
    projectsMock.deleteProject.mockRejectedValue(new Error('disk locked'))

    render(<WorkspaceProjectsSettings />)

    fireEvent.click(screen.getAllByRole('button', { name: 'Delete' })[0])

    const dialog = await screen.findByRole('dialog')

    fireEvent.click(within(dialog as HTMLElement).getByRole('button', { name: 'Delete' }))

    await waitFor(() => expect(notifyError).toHaveBeenCalledWith(expect.any(Error), 'Delete'))
  })

  it('removes a folder only after confirmation and routes errors to notifyError', async () => {
    projectsMock.removeProjectFolder.mockResolvedValue(undefined)

    render(<WorkspaceProjectsSettings />)

    fireEvent.click(screen.getAllByRole('button', { name: 'Remove' })[0])

    const dialog = await screen.findByRole('dialog')

    expect(dialog.textContent).toContain('Remove this folder from the project?')

    fireEvent.click(within(dialog as HTMLElement).getByRole('button', { name: 'Remove' }))

    await waitFor(() => expect(projectsMock.removeProjectFolder).toHaveBeenCalledWith('p_one', '/work/app'))

    projectsMock.removeProjectFolder.mockRejectedValue(new Error('boom'))
    fireEvent.click(screen.getAllByRole('button', { name: 'Remove' })[1])

    const second = await screen.findByRole('dialog')

    fireEvent.click(within(second as HTMLElement).getByRole('button', { name: 'Remove' }))

    await waitFor(() => expect(notifyError).toHaveBeenCalledWith(expect.any(Error), 'Could not remove folder'))
  })

  it('sets a folder as primary through the store and surfaces failures', async () => {
    projectsMock.setProjectFolderPrimary.mockResolvedValue(undefined)

    render(<WorkspaceProjectsSettings />)

    fireEvent.click(screen.getAllByRole('button', { name: 'Set primary' })[0])

    await waitFor(() => expect(projectsMock.setProjectFolderPrimary).toHaveBeenCalledWith('p_one', '/work/app-docs'))

    projectsMock.setProjectFolderPrimary.mockRejectedValue(new Error('boom'))
    fireEvent.click(screen.getAllByRole('button', { name: 'Set primary' })[0])

    await waitFor(() =>
      expect(notifyError).toHaveBeenCalledWith(expect.any(Error), 'Could not set the primary folder')
    )
  })

  it('sets the default project for new chats through the store', async () => {
    projectsMock.setDefaultProject.mockResolvedValue(undefined)

    render(<WorkspaceProjectsSettings />)

    fireEvent.click(screen.getByRole('switch', { name: 'Default for new chats Borealis' }))

    await waitFor(() => expect(projectsMock.setDefaultProject).toHaveBeenCalledWith('p_two'))
  })

  it('surfaces a set-default failure with notifyError', async () => {
    projectsMock.setDefaultProject.mockRejectedValue(new Error('boom'))

    render(<WorkspaceProjectsSettings />)

    fireEvent.click(screen.getByRole('switch', { name: 'Default for new chats Apollo' }))

    await waitFor(() =>
      expect(notifyError).toHaveBeenCalledWith(expect.any(Error), 'Could not set the default project')
    )
  })

  it('writes appearance changes through the shared picker', async () => {
    projectsMock.setProjectAppearance.mockResolvedValue(false)

    render(<WorkspaceProjectsSettings />)

    fireEvent.click(screen.getAllByRole('button', { name: 'Appearance' })[0])

    const swatch = await screen.findByRole('button', { name: 'No color' })

    fireEvent.click(swatch)

    await waitFor(() =>
      expect(projectsMock.setProjectAppearance).toHaveBeenCalledWith(
        expect.objectContaining({ id: 'p_one' }),
        expect.objectContaining({ color: null })
      )
    )
  })

  it('disables the row actions while a write is in flight', async () => {
    let release!: () => void
    projectsMock.deleteProject.mockImplementation(
      () =>
        new Promise<void>(resolve => {
          release = resolve
        })
    )

    render(<WorkspaceProjectsSettings />)

    const row = screen.getByRole('listitem', { name: 'Apollo' })

    fireEvent.click(within(row).getByRole('button', { name: 'Delete' }))

    const dialog = await screen.findByRole('dialog')

    fireEvent.click(within(dialog as HTMLElement).getByRole('button', { name: 'Delete' }))

    await waitFor(() => expect(projectsMock.deleteProject).toHaveBeenCalled())

    // While the dialog is open the row sits behind its aria-hidden layer, so
    // assert on the DOM node directly: the row's controls stay disabled until
    // the write settles.
    const rowDelete = [...row.querySelectorAll('button')].find(button => button.textContent === 'Delete')!

    expect(rowDelete.disabled).toBe(true)

    release()

    await waitFor(() => expect(rowDelete.disabled).toBe(false))
  })

  it('locks every run-controlled control on all rows while any write is in flight', async () => {
    let release!: () => void

    projectsMock.deleteProject.mockImplementation(
      () =>
        new Promise<void>(resolve => {
          release = resolve
        })
    )

    render(<WorkspaceProjectsSettings />)

    const apollo = screen.getByRole('listitem', { name: 'Apollo' })
    const borealis = screen.getByRole('listitem', { name: 'Borealis' })

    fireEvent.click(within(apollo).getByRole('button', { name: 'Delete' }))

    const dialog = await screen.findByRole('dialog')

    fireEvent.click(within(dialog as HTMLElement).getByRole('button', { name: 'Delete' }))

    await waitFor(() => expect(projectsMock.deleteProject).toHaveBeenCalled())

    // While the dialog is open the other rows sit behind its aria-hidden
    // layer, so assert on the DOM nodes directly: every run-controlled
    // control on every row must lock until the write settles, otherwise it is
    // an enabled dead click (run() no-ops while busyKey is set).
    const borealisSwitch = borealis.querySelector('button[role="switch"]') as HTMLButtonElement
    const borealisRemove = [...borealis.querySelectorAll('button')].find(b => b.getAttribute('aria-label') === 'Remove')!
    const apolloPrimary = [...apollo.querySelectorAll('button')].find(b => b.getAttribute('aria-label') === 'Set primary')!

    expect(borealisSwitch.disabled).toBe(true)
    expect(borealisRemove.disabled).toBe(true)
    expect(apolloPrimary.disabled).toBe(true)

    release()

    await waitFor(() => expect(borealisSwitch.disabled).toBe(false))
    expect(borealisRemove.disabled).toBe(false)
    expect(apolloPrimary.disabled).toBe(false)
  })
})

describe('store folder helpers', () => {
  const foldered = project({
    folders: [
      { added_at: 0, is_primary: true, label: null, path: '/a' },
      { added_at: 0, is_primary: false, label: null, path: '/b' }
    ],
    id: 'p_f',
    primary_path: '/a'
  })

  beforeEach(() => {
    window.localStorage.clear()
    $activeGatewayProfile.set('default')
    $profiles.set([{ is_default: true, name: 'default' } as never])
  })

  function gatewayReturning(payload: unknown) {
    return async (method: string) => {
      if (method === 'projects.list') {
        return { active_id: null, default_id: null, projects: [payload] }
      }

      if (method === 'projects.tree') {
        return { active_id: null, projects: [], scoped_session_ids: [] }
      }

      return { project: payload }
    }
  }

  it('removeProjectFolder drops the folder optimistically and calls projects.remove_folder', async () => {
    const real = await vi.importActual<typeof ProjectsStore>('@/store/projects')
    const request = vi.fn().mockImplementation(gatewayReturning(foldered))

    setPrimaryGateway({ connectionState: 'open', request } as never, 'default')
    real.$projects.set([foldered])

    await real.removeProjectFolder('p_f', '/a')

    expect(request).toHaveBeenCalledWith('projects.remove_folder', { id: 'p_f', path: '/a', profile: 'default' })
    expect(real.$projects.get().find(p => p.id === 'p_f')?.folders.map(f => f.path)).toEqual(['/b'])
    expect(real.$projects.get().find(p => p.id === 'p_f')?.primary_path).toBe('/b')
  })

  it('removeProjectFolder rolls the cache back when the write fails', async () => {
    const real = await vi.importActual<typeof ProjectsStore>('@/store/projects')

    const request = vi.fn().mockImplementation(async (method: string) => {
      if (method === 'projects.remove_folder') {
        throw new Error('locked')
      }

      return gatewayReturning(foldered)(method)
    })

    setPrimaryGateway({ connectionState: 'open', request } as never, 'default')
    real.$projects.set([foldered])

    await expect(real.removeProjectFolder('p_f', '/b')).rejects.toThrow('locked')
    expect(real.$projects.get().find(p => p.id === 'p_f')?.folders.map(f => f.path)).toEqual(['/a', '/b'])
  })

  it('setProjectFolderPrimary marks the folder primary and calls projects.set_primary', async () => {
    const real = await vi.importActual<typeof ProjectsStore>('@/store/projects')
    const request = vi.fn().mockImplementation(gatewayReturning(foldered))

    setPrimaryGateway({ connectionState: 'open', request } as never, 'default')
    real.$projects.set([foldered])

    await real.setProjectFolderPrimary('p_f', '/b')

    expect(request).toHaveBeenCalledWith('projects.set_primary', { id: 'p_f', path: '/b', profile: 'default' })

    const folders = real.$projects.get().find(p => p.id === 'p_f')?.folders

    expect(folders?.find(f => f.path === '/b')?.is_primary).toBe(true)
    expect(folders?.find(f => f.path === '/a')?.is_primary).toBe(false)
    expect(real.$projects.get().find(p => p.id === 'p_f')?.primary_path).toBe('/b')
  })
})
