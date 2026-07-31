import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { switchBranchInRepo } from '@/store/projects'

import {
  EnteredProjectContent,
  type SidebarProjectTree,
  type SidebarSessionGroup,
  SidebarWorkspaceGroup
} from './projects'
import { SidebarSessionsSection } from './sessions-section'

const startNewSessionDrag = vi.hoisted(() => vi.fn())

vi.mock('../new-session-drag', () => ({ startNewSessionDrag }))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel' },
      desktop: {},
      sidebar: {
        dateDivider: { earlier: 'Earlier', lastWeek: 'Last week', today: 'Today', yesterday: 'Yesterday' },
        newSessionIn: (label: string) => `New session in ${label}`,
        noSessions: 'No sessions yet',
        projects: {
          copyPath: 'Copy path',
          enter: (label: string) => `Enter ${label}`,
          forceRemove: 'Force remove',
          menu: 'Actions',
          removeFromSidebar: 'Remove from sidebar',
          removeWorktree: 'Remove worktree',
          removeWorktreeConfirm: 'Remove this worktree?',
          removeWorktreeDirty: 'This worktree has changes.',
          removeWorktreeFailed: 'Could not remove worktree',
          reorder: (label: string) => `Reorder ${label}`,
          reveal: 'Reveal in file manager',
          toggle: (label: string, open: boolean) => `${open ? 'Show' : 'Hide'} ${label} sessions`
        },
        showMoreIn: (count: number, label: string) => `Show ${count} more in ${label}`
      },
      statusStack: { coding: { switchFailed: (label: string) => `Could not switch to ${label}` } }
    }
  })
}))

vi.mock('./projects/model', () => ({
  PROJECT_PREVIEW_COUNT: 3,
  SIDEBAR_GROUP_PAGE: 20,
  latestProjectSessions: () => [],
  useWorkspaceNodeOpen: () => [false, vi.fn()]
}))

vi.mock('./projects/project-menu', () => ({
  ProjectContextMenu: ({ children }: { children: ReactNode }) => children,
  ProjectMenu: () => null
}))

vi.mock('@/store/projects', async () => ({
  ...(await vi.importActual('@/store/projects')),
  removeWorktreePath: vi.fn(),
  switchBranchInRepo: vi.fn()
}))

const noop = vi.fn()

const baseProps = () => ({
  activeSessionId: null,
  emptyState: null,
  label: 'Projects',
  onArchiveSession: noop,
  onDeleteSession: noop,
  onNewSessionInWorkspace: noop,
  onResumeSession: noop,
  onToggle: noop,
  onTogglePin: noop,
  open: true,
  pinned: false,
  sessions: [] as SessionInfo[],
  workingSessionIdSet: new Set<string>()
})

const group = (overrides: Partial<SidebarSessionGroup> = {}): SidebarSessionGroup => ({
  id: '/repo/.worktrees/feature',
  isMain: false,
  label: 'feature',
  path: '/repo/.worktrees/feature',
  sessions: [],
  ...overrides
})

const project = (overrides: Partial<SidebarProjectTree> = {}): SidebarProjectTree => ({
  id: 'project-1',
  isNoProject: false,
  label: 'Project One',
  path: '/repo/project-one',
  repos: [],
  sessionCount: 0,
  ...overrides
})

function commitLatestDrag() {
  expect(startNewSessionDrag).toHaveBeenCalledOnce()

  const commit = startNewSessionDrag.mock.calls[0]?.[0] as (placement: {
    anchor: string
    before?: null | string
    dir: 'center' | 'right'
  }) => void

  commit({ anchor: 'workspace', before: 'session-tile:next', dir: 'right' })
}

afterEach(cleanup)

beforeEach(() => {
  noop.mockClear()
  startNewSessionDrag.mockReset()
  vi.mocked(switchBranchInRepo).mockReset()
})

describe('project-associated new-session drag sources', () => {
  it('drags from the project overview + with the project cwd', () => {
    const onNewSessionSplit = vi.fn()

    render(
      <SidebarSessionsSection {...baseProps()} onNewSessionSplit={onNewSessionSplit} projectOverview={[project()]} />
    )

    fireEvent.pointerDown(screen.getByRole('button', { name: 'New session in Project One' }), { button: 0 })
    commitLatestDrag()

    expect(onNewSessionSplit).toHaveBeenCalledWith('right', {
      anchor: 'workspace',
      before: 'session-tile:next',
      cwd: '/repo/project-one'
    })
  })

  it('drags from a workspace/worktree + with that lane cwd', async () => {
    const onNewSessionSplit = vi.fn()

    render(
      <SidebarSessionsSection
        {...baseProps()}
        groups={[group({ sessions: [{ id: 'workspace-session' } as SessionInfo] })]}
        onNewSessionSplit={onNewSessionSplit}
      />
    )

    fireEvent.pointerDown(screen.getByRole('button', { name: 'New session in feature' }), { button: 0 })
    commitLatestDrag()

    await waitFor(() => {
      expect(onNewSessionSplit).toHaveBeenCalledWith('right', {
        anchor: 'workspace',
        before: 'session-tile:next',
        cwd: '/repo/.worktrees/feature'
      })
    })
  })

  it('switches a main-checkout lane to its labeled branch before creating the dragged session', async () => {
    const onNewSessionSplit = vi.fn()
    vi.mocked(switchBranchInRepo).mockResolvedValue(undefined)

    render(
      <SidebarSessionsSection
        {...baseProps()}
        groups={[
          group({
            id: '/repo::main',
            isMain: true,
            label: 'main',
            path: '/repo',
            sessions: [{ id: 'main-session' } as SessionInfo]
          })
        ]}
        onNewSessionSplit={onNewSessionSplit}
      />
    )

    fireEvent.pointerDown(screen.getByRole('button', { name: 'New session in main' }), { button: 0 })
    commitLatestDrag()

    await waitFor(() => expect(onNewSessionSplit).toHaveBeenCalledOnce())
    expect(switchBranchInRepo).toHaveBeenCalledWith('/repo', 'main')
    expect(vi.mocked(switchBranchInRepo).mock.invocationCallOrder[0]).toBeLessThan(
      onNewSessionSplit.mock.invocationCallOrder[0]
    )
  })

  it('drags from an entered-project repo + with that repo cwd', () => {
    const onNewSessionSplit = vi.fn()

    const repoA = {
      groups: [group({ id: '/repo/a::main', isMain: true, label: 'main', path: '/repo/a' })],
      id: '/repo/a',
      label: 'Repo A',
      path: '/repo/a',
      sessionCount: 1
    }

    const repoB = {
      groups: [group({ id: '/repo/b::main', isMain: true, label: 'main', path: '/repo/b' })],
      id: '/repo/b',
      label: 'Repo B',
      path: '/repo/b',
      sessionCount: 1
    }

    render(
      <SidebarSessionsSection
        {...baseProps()}
        onNewSessionSplit={onNewSessionSplit}
        projectContent={project({ repos: [repoA, repoB], sessionCount: 1 })}
      />
    )

    fireEvent.pointerDown(screen.getByRole('button', { name: 'New session in Repo A' }), { button: 0 })
    commitLatestDrag()

    expect(onNewSessionSplit).toHaveBeenCalledWith('right', {
      anchor: 'workspace',
      before: 'session-tile:next',
      cwd: '/repo/a'
    })
  })

  it('keeps profile-group add buttons click-only', () => {
    const onNewSessionSplit = vi.fn()

    render(
      <SidebarWorkspaceGroup
        group={group({ id: 'profile:reviewer', label: 'Reviewer', mode: 'profile', path: null })}
        onNewSession={noop}
        onNewSessionSplit={onNewSessionSplit}
        renderRows={() => null}
      />
    )

    fireEvent.pointerDown(screen.getByRole('button', { name: 'New session in Reviewer' }), { button: 0 })

    expect(startNewSessionDrag).not.toHaveBeenCalled()
    expect(onNewSessionSplit).not.toHaveBeenCalled()
  })

  it('does not expose a drag or click source for kanban aggregate lanes', () => {
    const onNewSessionSplit = vi.fn()
    const kanbanGroup = group({ id: 'kanban:review', isKanban: true, label: 'Review board' })

    const kanbanProject = project({
      repos: [
        {
          groups: [kanbanGroup],
          id: '/repo',
          label: 'Repo',
          path: '/repo',
          sessionCount: 0
        }
      ]
    })

    render(
      <EnteredProjectContent
        onNewSession={noop}
        onNewSessionSplit={onNewSessionSplit}
        project={kanbanProject}
        renderRows={() => null}
      />
    )

    expect(screen.queryByRole('button', { name: 'New session in Review board' })).toBeNull()
    expect(startNewSessionDrag).not.toHaveBeenCalled()
    expect(onNewSessionSplit).not.toHaveBeenCalled()
  })
})
