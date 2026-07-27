import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGitWorktree } from '@/global'
import type { SessionInfo } from '@/hermes'

import type { SidebarProjectTree } from './projects'
import { SidebarSessionsSection } from './sessions-section'

vi.mock('@/i18n', () => ({
  useI18n: () => ({ t: { sidebar: { dateDivider: {} } } })
}))

vi.mock('./projects', async importOriginal => {
  const actual = (await importOriginal()) as Record<string, unknown>

  return {
    ...actual,
    EnteredProjectContent: ({ project }: { project: SidebarProjectTree }) => (
      <pre data-testid="entered-project">
        {JSON.stringify(
          project.repos.map(repo => ({
            id: repo.id,
            rows: repo.groups.flatMap(group => group.sessions.map(session => session.id))
          }))
        )}
      </pre>
    ),
    ProjectOverviewRow: () => null,
    SidebarWorkspaceGroup: () => null
  }
})

afterEach(cleanup)

const noop = vi.fn()

function session(id: string, cwd: string): SessionInfo {
  return { cwd, id, last_active: 2, started_at: 1 } as unknown as SessionInfo
}

function project(repos: SidebarProjectTree['repos']): SidebarProjectTree {
  return { id: 'project', label: 'Project', path: null, repos, sessionCount: 0 }
}

function repo(id: string, path: string): SidebarProjectTree['repos'][number] {
  return {
    groups: [{ id: `${id}::branch::main`, isMain: true, label: 'main', path, sessions: [] }],
    id,
    label: id,
    path,
    sessionCount: 0
  }
}

function renderProject(
  projectContent: SidebarProjectTree,
  liveSessions: SessionInfo[],
  projectRepoWorktrees: Record<string, HermesGitWorktree[]>
) {
  render(
    <SidebarSessionsSection
      activeSessionId={null}
      emptyState={<div>empty project</div>}
      label="Sessions"
      liveSessions={liveSessions}
      onArchiveSession={noop}
      onDeleteSession={noop}
      onResumeSession={noop}
      onToggle={noop}
      onTogglePin={noop}
      open
      pinned={false}
      projectContent={projectContent}
      projectRepoWorktrees={projectRepoWorktrees}
      sessions={[]}
      workingSessionIdSet={new Set()}
    />
  )
}

describe('SidebarSessionsSection entered-project composition', () => {
  it('checks the empty state only after placing a live row in an out-of-tree visual worktree', () => {
    const tree = project([repo('/repo', '/repo')])

    const worktrees = {
      '/repo': [
        { branch: 'main', detached: false, isMain: true, locked: false, path: '/repo' },
        { branch: 'feature', detached: false, isMain: false, locked: false, path: '/outside/repo-feature' }
      ]
    }

    renderProject(tree, [session('fresh', '/outside/repo-feature/src')], worktrees)

    expect(screen.getByTestId('entered-project').textContent).toContain('fresh')
    expect(screen.queryByText('empty project')).toBeNull()
  })

  it('routes a fresh row to a more specific visual-worktree repo before assigning snapshot ownership', () => {
    const tree = project([repo('broad', '/workspace'), repo('specific', '/repo')])

    const worktrees = {
      '/workspace': [{ branch: 'main', detached: false, isMain: true, locked: false, path: '/workspace' }],
      '/repo': [
        { branch: 'main', detached: false, isMain: true, locked: false, path: '/repo' },
        { branch: 'feature', detached: false, isMain: false, locked: false, path: '/workspace/specific-wt' }
      ]
    }

    renderProject(tree, [session('fresh', '/workspace/specific-wt/src')], worktrees)

    expect(JSON.parse(screen.getByTestId('entered-project').textContent ?? '[]')).toEqual([
      { id: 'broad', rows: [] },
      { id: 'specific', rows: ['fresh'] }
    ])
  })
})
