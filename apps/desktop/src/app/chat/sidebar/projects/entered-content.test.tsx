import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import * as notifications from '@/store/notifications'
import * as projectStore from '@/store/projects'

import { EnteredMainSessionButton, EnteredProjectContent } from './entered-content'
import type { SidebarProjectTree, SidebarSessionGroup } from './workspace-groups'

const switchBranchInRepoMock = vi.spyOn(projectStore, 'switchBranchInRepo')
const notifyErrorMock = vi.spyOn(notifications, 'notifyError')

afterEach(() => {
  cleanup()
  notifyErrorMock.mockReset()
  switchBranchInRepoMock.mockReset()
})

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel' },
      sidebar: {
        newSessionIn: (label: string) => `New session in ${label}`,
        showMoreIn: (count: number, label: string) => `Show ${count} more in ${label}`,
        projects: {
          forceRemove: 'Force remove',
          removeFromSidebar: 'Remove from sidebar',
          removeWorktree: 'Remove worktree',
          removeWorktreeConfirm: 'Remove this worktree?',
          removeWorktreeDirty: 'This worktree has changes.',
          removeWorktreeFailed: 'Could not remove worktree'
        }
      },
      statusStack: { coding: { switchFailed: (label: string) => `Could not switch to ${label}` } }
    }
  })
}))

vi.mock('./model', () => ({
  SIDEBAR_GROUP_PAGE: 5,
  useWorkspaceNodeOpen: () => [true, vi.fn()]
}))

vi.mock('./workspace-group', () => ({
  SidebarWorkspaceGroup: ({ group }: { group: SidebarSessionGroup }) => (
    <div data-testid={`workspace-group-${group.id}`}>{group.label}</div>
  )
}))

const session = (id: string, lastActive: number): SessionInfo =>
  ({ id, last_active: lastActive, started_at: lastActive }) as SessionInfo

const projectWithMain = (): SidebarProjectTree => ({
  id: 'project',
  label: 'Project',
  path: '/repo',
  repos: [
    {
      id: '/repo',
      label: 'repo',
      path: '/repo',
      sessionCount: 0,
      groups: [
        {
          id: 'main',
          isMain: true,
          label: 'main',
          path: '/repo',
          sessions: []
        }
      ]
    }
  ],
  sessionCount: 0
})

describe('EnteredProjectContent', () => {
  it('renders deduplicated main-checkout sessions directly and keeps linked worktrees grouped', () => {
    const staleDuplicate = session('duplicate', 2)
    const freshDuplicate = session('duplicate', 5)
    const recent = session('recent', 4)
    const oldest = session('oldest', 1)
    const linked = session('linked-session', 3)

    const project: SidebarProjectTree = {
      id: 'project',
      label: 'Project',
      path: '/repo',
      repos: [
        {
          id: '/repo',
          label: 'repo',
          path: '/repo',
          sessionCount: 5,
          groups: [
            {
              id: 'main',
              isMain: true,
              label: 'main',
              path: '/repo',
              sessions: [oldest, staleDuplicate]
            },
            {
              id: 'old-main',
              isMain: true,
              label: 'old-main',
              path: '/repo',
              sessions: [recent, freshDuplicate]
            },
            {
              id: 'linked',
              label: 'feature',
              path: '/repo-feature',
              sessions: [linked]
            }
          ]
        }
      ],
      sessionCount: 5
    }

    const renderRows = vi.fn((_sessions: SessionInfo[]) => null)

    render(<EnteredProjectContent project={project} renderRows={renderRows} />)

    expect(renderRows).toHaveBeenCalledOnce()
    expect(renderRows.mock.calls[0][0]).toEqual([freshDuplicate, recent, oldest])
    expect(screen.queryByTestId('workspace-group-main')).toBeNull()
    expect(screen.queryByTestId('workspace-group-old-main')).toBeNull()
    expect(screen.getByTestId('workspace-group-linked').textContent).toBe('feature')
  })

  it('pages flattened main-checkout sessions without restoring the branch header', () => {
    const sessions = Array.from({ length: 7 }, (_, index) => session(`session-${index}`, 7 - index))
    const project = projectWithMain()
    const repo = project.repos[0]
    const mainGroup = repo?.groups[0]

    if (!repo || !mainGroup) {
      throw new Error('expected main checkout fixture')
    }

    mainGroup.sessions = sessions
    repo.sessionCount = sessions.length
    project.sessionCount = sessions.length

    const renderRows = vi.fn((_sessions: SessionInfo[]) => null)

    render(<EnteredProjectContent project={project} renderRows={renderRows} />)

    expect(renderRows).toHaveBeenCalledOnce()
    expect(renderRows).toHaveBeenLastCalledWith(sessions.slice(0, 5))
    expect(screen.queryByTestId('workspace-group-main')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Show 2 more in main' }))

    expect(renderRows).toHaveBeenCalledTimes(2)
    expect(renderRows).toHaveBeenLastCalledWith(sessions)
    expect(screen.queryByRole('button', { name: 'Show 2 more in main' })).toBeNull()
  })

  it('switches to the visible main lane before creating a session from the entered-project header', async () => {
    switchBranchInRepoMock.mockResolvedValue(undefined)
    const onNewSession = vi.fn()

    render(<EnteredMainSessionButton onNewSession={onNewSession} project={projectWithMain()} />)
    fireEvent.click(screen.getByRole('button', { name: 'New session in main' }))

    await waitFor(() => expect(switchBranchInRepoMock).toHaveBeenCalledWith('/repo', 'main'))
    expect(onNewSession).toHaveBeenCalledWith('/repo')
    expect(switchBranchInRepoMock.mock.invocationCallOrder[0]).toBeLessThan(onNewSession.mock.invocationCallOrder[0])
  })

  it('does not create a session when switching the main lane fails', async () => {
    const error = new Error('switch failed')
    switchBranchInRepoMock.mockRejectedValue(error)
    notifyErrorMock.mockReturnValue('notification')
    const onNewSession = vi.fn()

    render(<EnteredMainSessionButton onNewSession={onNewSession} project={projectWithMain()} />)
    fireEvent.click(screen.getByRole('button', { name: 'New session in main' }))

    await waitFor(() => expect(notifyErrorMock).toHaveBeenCalledWith(error, 'Could not switch to main'))
    expect(onNewSession).not.toHaveBeenCalled()
  })
})
