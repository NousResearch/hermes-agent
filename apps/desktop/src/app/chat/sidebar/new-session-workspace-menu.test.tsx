import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SidebarSectionAddButton } from './chrome'
import { useNewSessionWorkspaceMenuItems } from './new-session-workspace-menu'
import type { SidebarProjectTree } from './projects'

const startNewProjectDrag = vi.hoisted(() => vi.fn())
const startNewSessionDrag = vi.hoisted(() => vi.fn())

vi.mock('@/app/chat/new-session-drag', () => ({ startNewProjectDrag, startNewSessionDrag }))

vi.mock('@/store/projects', () => ({
  $projectTree: atom<SidebarProjectTree[]>([]),
  pickProjectFolder: vi.fn()
}))

vi.mock('./projects', () => ({
  projectTreeCwd: (project: SidebarProjectTree): null | string =>
    project.path || project.repos.find(repo => repo.path)?.path || null
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        newSessionIn: (label: string) => `New session in ${label}`,
        projects: { addFolder: 'Add folder', createFailed: 'Could not create project' }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))

vi.mock('@/lib/keybinds/use-keybind-hint', () => ({ useKeybindHint: () => null }))

import { $projectTree, pickProjectFolder } from '@/store/projects'

const mockedPick = vi.mocked(pickProjectFolder)

function project(overrides: Partial<SidebarProjectTree> & { id: string; label: string }): SidebarProjectTree {
  return {
    groups: [],
    isAuto: false,
    isNoProject: false,
    lastActive: 0,
    path: null,
    previewSessions: [],
    repos: [],
    sessionCount: 0,
    ...overrides
  } as unknown as SidebarProjectTree
}

// The ungrouped sidebar "+": plain click stays a detached draft, right-click
// offers the project folders.
function UngroupedPlus({
  onPick,
  onPlainClick
}: {
  onPick: (path: string) => void
  onPlainClick: () => void
}) {
  const items = useNewSessionWorkspaceMenuItems(onPick)

  return (
    <SidebarSectionAddButton
      ariaLabel="New session"
      onPlainClick={onPlainClick}
      workspaceMenu={{ ariaLabel: 'New session', items }}
    />
  )
}

afterEach(() => {
  cleanup()
  mockedPick.mockReset()
})

describe('ungrouped new-session plus', () => {
  it('lists projects with a folder and starts the session in the picked one', async () => {
    $projectTree.set([
      project({ id: 'home', isNoProject: true, label: 'Home', path: null }),
      project({ id: 'p_alpha', label: 'Alpha', path: '/work/alpha' }),
      project({ id: 'p_beta', label: 'Beta', path: null })
    ])

    const onPick = vi.fn()
    const onPlainClick = vi.fn()

    render(<UngroupedPlus onPick={onPick} onPlainClick={onPlainClick} />)

    // Plain click is untouched: still a detached draft.
    fireEvent.click(screen.getByRole('button', { name: 'New session' }))
    expect(onPlainClick).toHaveBeenCalledOnce()
    expect(onPick).not.toHaveBeenCalled()

    fireEvent.contextMenu(screen.getByRole('button', { name: 'New session' }))

    // Only the project with a real folder is offered — Home stays detached.
    expect(await screen.findByRole('menuitem', { name: 'New session in Alpha' })).toBeTruthy()
    expect(screen.queryByRole('menuitem', { name: /Home/ })).toBeNull()
    expect(screen.queryByRole('menuitem', { name: /Beta/ })).toBeNull()

    fireEvent.click(screen.getByRole('menuitem', { name: 'New session in Alpha' }))
    expect(onPick).toHaveBeenCalledExactlyOnceWith('/work/alpha')
  })

  it('routes the folder picker into onPick and ignores a cancelled pick', async () => {
    $projectTree.set([project({ id: 'p_alpha', label: 'Alpha', path: '/work/alpha' })])

    const onPick = vi.fn()
    mockedPick.mockResolvedValueOnce('/picked/dir').mockResolvedValueOnce(null)

    render(<UngroupedPlus onPick={onPick} onPlainClick={vi.fn()} />)

    fireEvent.contextMenu(screen.getByRole('button', { name: 'New session' }))
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Add folder' }))
    await vi.waitFor(() => expect(onPick).toHaveBeenCalledExactlyOnceWith('/picked/dir'))

    fireEvent.contextMenu(screen.getByRole('button', { name: 'New session' }))
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Add folder' }))
    await vi.waitFor(() => expect(mockedPick).toHaveBeenCalledTimes(2))
    expect(onPick).toHaveBeenCalledTimes(1)
  })
})
