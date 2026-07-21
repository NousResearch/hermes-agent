import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { SidebarSessionsSection } from '../sessions-section'

import { EnteredProjectContent } from './entered-content'
import type { SidebarProjectTree, SidebarSessionGroup, SidebarWorkspaceTree } from './workspace-groups'

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

function lane(id: string, label: string, sessions: SessionInfo[]): SidebarSessionGroup {
  return { id, label, path: id, sessions }
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
    lane('/workspace/clawdified/cards::main', 'main', [
      session('Cards chat 1', 10, '/workspace/clawdified/cards'),
      session('Cards chat 2', 20, '/workspace/clawdified/cards'),
      session('Cards chat 3', 30, '/workspace/clawdified/cards'),
      session('Cards chat 4', 40, '/workspace/clawdified/cards'),
      session('Cards chat 7', 70, '/workspace/clawdified/cards'),
      session('Cards chat 8', 80, '/workspace/clawdified/cards'),
      session('Cards chat 9', 90, '/workspace/clawdified/cards')
    ]),
    lane('/workspace/clawdified/cards::concept', 'concept', [
      session('Cards chat 5', 50, '/workspace/clawdified/cards'),
      session('Cards chat 6', 60, '/workspace/clawdified/cards')
    ])
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

describe('EnteredProjectContent folder drill-down', () => {
  it('shows only folder rows until the user chooses one in a multi-folder project', () => {
    const { project } = fixture()
    const onFocusRepo = vi.fn()

    render(
      <EnteredProjectContent focusedRepoId={null} onFocusRepo={onFocusRepo} project={project} renderRows={chatRows} />
    )

    expect(screen.getByRole('button', { name: 'Open Business cards' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open Website' })).toBeTruthy()
    expect(screen.queryByText('Cards chat 1')).toBeNull()
    expect(screen.queryByText('Website chat')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Open Business cards' }))
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

    expect(screen.queryByRole('button', { name: 'Open Business cards' })).toBeNull()
    expect(screen.queryByLabelText(/Show .* more/)).toBeNull()
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

    fireEvent.click(screen.getByRole('button', { name: 'Open Business cards' }))
    expect(screen.getByText('Cards chat 1')).toBeTruthy()

    view.rerender(section())
    view.rerender(section(projectContent))

    expect(screen.getByRole('button', { name: 'Open Business cards' })).toBeTruthy()
    expect(screen.queryByText('Cards chat 1')).toBeNull()
  })
})
