import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import type * as ProjectStoreModule from '@/store/projects'

import { SidebarSessionsSection } from '../sessions-section'

import { EnteredProjectContent } from './entered-content'
import type { SidebarProjectTree, SidebarSessionGroup, SidebarWorkspaceTree } from './workspace-groups'

const projectStoreMocks = vi.hoisted(() => ({ switchBranchInRepo: vi.fn() }))

vi.mock('@/store/projects', async importOriginal => ({
  ...(await importOriginal<typeof ProjectStoreModule>()),
  switchBranchInRepo: projectStoreMocks.switchBranchInRepo
}))

let nextSession = 0

function session(title: string, startedAt: number, cwd: string): SessionInfo {
  return {
    archived: false,
    cwd,
    ended_at: null,
    id: `session-${nextSession++}`,
    input_tokens: 0,
    is_active: false,
    last_active: startedAt,
    message_count: 1,
    model: 'gpt-5',
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: startedAt,
    title,
    tool_call_count: 0
  }
}

function lane(
  id: string,
  label: string,
  sessions: SessionInfo[],
  options: Partial<Pick<SidebarSessionGroup, 'isHome' | 'isMain' | 'path'>> = {}
): SidebarSessionGroup {
  return { id, label, path: id, sessions, ...options }
}

function repo(id: string, label: string, groups: SidebarSessionGroup[]): SidebarWorkspaceTree {
  return {
    groups,
    id,
    label,
    path: id,
    sessionCount: groups.reduce((total, group) => total + group.sessions.length, 0)
  }
}

function project(repos: SidebarWorkspaceTree[]): SidebarProjectTree {
  return {
    id: 'clawdified-brand',
    label: 'Clawdified Brand',
    path: '/workspace/clawdified',
    repos,
    sessionCount: repos.reduce((total, item) => total + item.sessionCount, 0)
  }
}

function chatRows(sessions: SessionInfo[]) {
  return sessions.map(item => <div key={item.id}>{item.title}</div>)
}

function fixture() {
  const cards = repo('/workspace/clawdified/cards', 'Business cards', [
    lane(
      '/workspace/clawdified/cards::main',
      'main',
      [
        session('Cards chat 1', 10, '/workspace/clawdified/cards'),
        session('Cards chat 2', 20, '/workspace/clawdified/cards'),
        session('Cards chat 3', 30, '/workspace/clawdified/cards'),
        session('Cards chat 4', 40, '/workspace/clawdified/cards'),
        session('Cards chat 7', 70, '/workspace/clawdified/cards'),
        session('Cards chat 8', 80, '/workspace/clawdified/cards'),
        session('Cards chat 9', 90, '/workspace/clawdified/cards')
      ],
      { isHome: true, isMain: true, path: '/workspace/clawdified/cards' }
    ),
    lane(
      '/workspace/clawdified/cards::concept',
      'concept',
      [
        session('Cards chat 5', 50, '/workspace/clawdified/cards'),
        session('Cards chat 6', 60, '/workspace/clawdified/cards')
      ],
      { path: '/workspace/clawdified/cards/.worktrees/concept' }
    )
  ])

  const website = repo('/workspace/clawdified/website', 'Website', [
    lane('/workspace/clawdified/website::main', 'main', [session('Website chat', 70, '/workspace/clawdified/website')])
  ])

  return { cards, project: project([cards, website]), website }
}

afterEach(() => {
  cleanup()
  window.localStorage.clear()
})

beforeEach(() => {
  projectStoreMocks.switchBranchInRepo.mockReset()
  projectStoreMocks.switchBranchInRepo.mockResolvedValue(undefined)
})

describe('EnteredProjectContent folder drill-down', () => {
  it('shows only folder rows until the user chooses one in a multi-folder project', () => {
    const { project } = fixture()
    const onFocusRepo = vi.fn()

    render(
      <EnteredProjectContent focusedRepoId={null} onFocusRepo={onFocusRepo} project={project} renderRows={chatRows} />
    )

    expect(screen.getByRole('button', { name: 'Business cards' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Website' })).toBeTruthy()
    expect(screen.queryByText('Cards chat 1')).toBeNull()
    expect(screen.queryByText('Website chat')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Business cards' }))
    expect(onFocusRepo).toHaveBeenCalledWith('/workspace/clawdified/cards')
  })

  it('focuses one folder, hides its siblings, and renders every chat without an ellipsis', () => {
    const { project } = fixture()
    const onExitRepo = vi.fn()

    render(
      <EnteredProjectContent
        focusedRepoId="/workspace/clawdified/cards"
        onExitRepo={onExitRepo}
        project={project}
        renderRows={chatRows}
      />
    )

    for (let index = 1; index <= 9; index += 1) {
      expect(screen.getByText(`Cards chat ${index}`)).toBeTruthy()
    }

    expect(screen.queryByText('Website chat')).toBeNull()
    expect(screen.queryByLabelText(/Show .* more/)).toBeNull()

    const backButton = screen.getByText('Clawdified Brand').closest('button')

    expect(backButton?.className).toContain('opacity-70')
    fireEvent.click(backButton!)
    expect(onExitRepo).toHaveBeenCalledOnce()
  })

  it('opens a single-folder project directly as one complete chat list', () => {
    const { cards } = fixture()

    render(<EnteredProjectContent project={project([cards])} renderRows={chatRows} />)

    for (let index = 1; index <= 9; index += 1) {
      expect(screen.getByText(`Cards chat ${index}`)).toBeTruthy()
    }

    expect(screen.queryByRole('button', { name: 'Business cards' })).toBeNull()
    expect(screen.queryByLabelText(/Show .* more/)).toBeNull()
  })

  it('deduplicates chats across lanes and globally sorts them by recency', () => {
    const repoPath = '/workspace/clawdified/sorted'
    const older = session('Older chat', 10, repoPath)
    const duplicate = session('Duplicate chat', 20, repoPath)
    const newest = session('Newest chat', 30, `${repoPath}/.worktrees/concept`)

    const sortedRepo = repo(repoPath, 'Sorted chats', [
      lane(`${repoPath}::main`, 'main', [older, duplicate], { isHome: true, isMain: true, path: repoPath }),
      lane(`${repoPath}::concept`, 'concept', [duplicate, newest], { path: `${repoPath}/.worktrees/concept` })
    ])

    render(<EnteredProjectContent project={project([sortedRepo])} renderRows={chatRows} />)

    expect(screen.getAllByText(/^(Newest|Duplicate|Older) chat$/).map(row => row.textContent)).toEqual([
      'Newest chat',
      'Duplicate chat',
      'Older chat'
    ])
    expect(screen.getAllByText('Duplicate chat')).toHaveLength(1)
  })

  it('preserves linked-worktree targeting and controls inside a focused folder', async () => {
    const { project } = fixture()
    const onNewSession = vi.fn()

    render(
      <EnteredProjectContent
        focusedRepoId="/workspace/clawdified/cards"
        onNewSession={onNewSession}
        project={project}
        renderRows={chatRows}
      />
    )

    fireEvent.pointerDown(screen.getByRole('button', { name: 'New session in Business cards' }), {
      button: 0,
      ctrlKey: false,
      pointerType: 'mouse'
    })

    const worktreeTarget = await screen.findByRole('menuitem', { name: 'New session in concept' })

    const worktreeActions = screen.getByRole('menuitem', { name: 'concept' })

    fireEvent.pointerMove(worktreeActions, { pointerType: 'mouse' })
    expect(await screen.findByRole('menuitem', { name: 'Remove worktree…' })).toBeTruthy()
    fireEvent.click(worktreeTarget)

    await waitFor(() => expect(onNewSession).toHaveBeenCalledWith('/workspace/clawdified/cards/.worktrees/concept'))
    expect(screen.queryByLabelText(/Show .* more/)).toBeNull()
  })

  it('switches the primary lane branch before creating a session there', async () => {
    const { project } = fixture()
    const onNewSession = vi.fn()

    render(
      <EnteredProjectContent
        focusedRepoId="/workspace/clawdified/cards"
        onNewSession={onNewSession}
        project={project}
        renderRows={chatRows}
      />
    )

    fireEvent.pointerDown(screen.getByRole('button', { name: 'New session in Business cards' }), {
      button: 0,
      ctrlKey: false,
      pointerType: 'mouse'
    })
    fireEvent.click(await screen.findByRole('menuitem', { name: 'New session in main' }))

    await waitFor(() => {
      expect(projectStoreMocks.switchBranchInRepo).toHaveBeenCalledWith('/workspace/clawdified/cards', 'main')
      expect(onNewSession).toHaveBeenCalledWith('/workspace/clawdified/cards')
    })
  })
})

describe('SidebarSessionsSection folder focus lifecycle', () => {
  it('returns to the folder list after exiting and re-entering the same project', () => {
    const { project: projectContent } = fixture()

    const section = (content?: SidebarProjectTree) => (
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={<div>No sessions</div>}
        label="Sessions"
        onArchiveSession={vi.fn()}
        onDeleteSession={vi.fn()}
        onResumeSession={vi.fn()}
        onToggle={vi.fn()}
        onTogglePin={vi.fn()}
        open
        pinned={false}
        projectContent={content}
        sessions={[]}
        workingSessionIdSet={new Set()}
      />
    )

    const view = render(section(projectContent))

    fireEvent.click(screen.getByRole('button', { name: 'Business cards' }))
    expect(screen.getByText('Cards chat 1')).toBeTruthy()

    view.rerender(section())
    view.rerender(section(projectContent))

    expect(screen.getByRole('button', { name: 'Business cards' })).toBeTruthy()
    expect(screen.queryByText('Cards chat 1')).toBeNull()
  })

  it('restores the project back row if the focused folder disappears from the live project tree', () => {
    const { project: projectContent, website } = fixture()
    const projectBackRow = <div>Back to projects</div>

    const section = (content: SidebarProjectTree) => (
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={<div>No sessions</div>}
        label="Sessions"
        onArchiveSession={vi.fn()}
        onDeleteSession={vi.fn()}
        onResumeSession={vi.fn()}
        onToggle={vi.fn()}
        onTogglePin={vi.fn()}
        open
        pinned={false}
        projectBackRow={projectBackRow}
        projectContent={content}
        sessions={[]}
        workingSessionIdSet={new Set()}
      />
    )

    const view = render(section(projectContent))

    fireEvent.click(screen.getByRole('button', { name: 'Business cards' }))
    expect(screen.queryByText('Back to projects')).toBeNull()

    view.rerender(section(project([website])))

    expect(screen.getByText('Back to projects')).toBeTruthy()
    expect(screen.getByText('Website chat')).toBeTruthy()
  })
})
