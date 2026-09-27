import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { ProjectInfo, SessionInfo } from '@/hermes'
import { $sidebarShowAllSessions } from '@/store/layout'
import { $projects, $projectTree } from '@/store/projects'

import { EnteredProjectContent } from './entered-content'
import type * as ProjectModel from './model'
import {
  liveSessionsForProject,
  NO_PROJECT_ID,
  overlayLiveLanes,
  projectOwnerBySessionId,
  type SidebarProjectTree
} from './workspace-groups'

vi.mock('./model', async importOriginal => ({
  ...(await importOriginal<typeof ProjectModel>()),
  useWorkspaceNodeOpen: () => [true, vi.fn()]
}))

const row = (id: string, cwd: null | string, git_repo_root: null | string = '/workspace') =>
  ({ id, cwd, git_repo_root, git_branch: 'main', title: id, last_active: 1 }) as SessionInfo

const definitions = [
  ['career', 'Career', '/workspace/career', '/workspace'],
  ['hermes', 'Hermes Setup & Operations', '/workspace/hermes', '/workspace'],
  ['apply', 'Apply', '/workspace/apply', '/workspace'],
  ['ide', 'Agent OS IDE', '/workspace/ide', '/workspace/ide'],
  ['brand', 'Personal Brand Site', '/workspace/brand', '/workspace/brand']
]

const projects = definitions.map(([id, name, path]) => ({
  id,
  name,
  folders: [{ path }],
  archived: false
})) as ProjectInfo[]

const snapshots = (): SidebarProjectTree[] => [
  ...definitions.map(([id, label, path, root]) => {
    const sessions = id === 'apply' ? [] : [row(`${id}-owned`, path, root)]

    return {
      id,
      label,
      path,
      sessionCount: sessions.length,
      sessionIds: sessions.map(s => s.id),
      repos: [
        {
          id: root,
          label,
          path: root,
          sessionCount: sessions.length,
          groups: [{ id: `${root}::branch::main`, label: 'main', path: root, isMain: true, sessions }]
        }
      ]
    }
  }),
  {
    id: NO_PROJECT_ID,
    label: 'Home',
    path: null,
    isNoProject: true,
    sessionCount: 1,
    repos: [
      {
        id: NO_PROJECT_ID,
        label: 'Home',
        path: null,
        sessionCount: 1,
        groups: [{ id: NO_PROJECT_ID, label: 'Home', path: null, sessions: [row('home-owned', null, null)] }]
      }
    ]
  }
]

const renderRows = (sessions: SessionInfo[]) =>
  sessions.map(session => (
    <div data-testid="membership" key={session.id}>
      {session.id}:{session.title}
    </div>
  ))

const membership = () =>
  screen
    .queryAllByTestId('membership')
    .map(el => el.textContent)
    .sort()

afterEach(() => {
  cleanup()
  $projects.set([])
  $projectTree.set([])
  $sidebarShowAllSessions.set(false)
})

it('keeps backend ownership at the final rendered boundary despite shared roots and stale cwd', () => {
  const tree = snapshots()
  $projects.set(projects)
  $projectTree.set(tree)
  $sidebarShowAllSessions.set(true)

  const live = [
    {
      ...row('career-tip', '/workspace/hermes', null),
      _lineage_root_id: 'career-owned',
      _lineage_ids: ['career-owned', 'career-tip']
    },
    row('hermes-owned', '/workspace/hermes'),
    row('ide-owned', '/workspace/ide', '/workspace/ide'),
    row('brand-owned', '/workspace/brand', '/workspace/brand'),
    { ...row('home-tip', '/workspace/hermes', null), _lineage_ids: ['home-owned', 'home-tip'] }
  ]

  const owners = projectOwnerBySessionId(tree)

  for (const project of tree) {
    const overlaid = overlayLiveLanes(project, live, new Set(), owners)
    const view = render(<EnteredProjectContent liveSessions={live} project={overlaid} renderRows={renderRows} />)

    const expected = project.repos
      .flatMap(repo => repo.groups.flatMap(group => group.sessions.map(s => `${s.id}:${s.title}`)))
      .map(value => value.replaceAll('career-owned', 'career-tip'))
      .sort()

    expect(membership(), project.label).toEqual(expected)
    view.unmount()
  }
})

it('resolves new rows with every project folder while preserving updates, removals, Home and discovered worktrees', () => {
  const tree = snapshots()
  $projects.set(projects)
  $projectTree.set(tree)
  $sidebarShowAllSessions.set(true)

  const live = [
    ...definitions.map(([id, , path]) => row(`${id}-new`, `${path}/src`, null)),
    { ...row('career-owned', '/workspace/career'), title: 'updated' },
    row('home-new', null, null),
    row('hermes-sibling', '/workspace/hermes-feature/src', '/workspace/hermes'),
    row('hermes-outside', '/other/hermes-feature/src', '/workspace/hermes'),
    row('unclaimed-workspace', '/workspace/unclaimed', null)
  ]

  const removed = new Set(['hermes-owned'])
  const owners = projectOwnerBySessionId(tree)

  const expectedById: Record<string, string[]> = {
    career: ['career-new:career-new', 'career-owned:updated'],
    hermes: ['hermes-new:hermes-new', 'hermes-outside:hermes-outside', 'hermes-sibling:hermes-sibling'],
    apply: ['apply-new:apply-new'],
    ide: ['ide-new:ide-new', 'ide-owned:ide-owned'],
    brand: ['brand-new:brand-new', 'brand-owned:brand-owned'],
    [NO_PROJECT_ID]: ['home-new:home-new', 'home-owned:home-owned']
  }

  const worktrees = ['/workspace/hermes-feature', '/other/hermes-feature', '/other/empty'].map(path => ({
    path,
    branch: path,
    isMain: false,
    detached: false,
    locked: false
  }))

  for (const project of tree) {
    const admitted = liveSessionsForProject(project, live, projects, owners)
    const overlaid = overlayLiveLanes(project, admitted, removed, owners)

    const view = render(
      <EnteredProjectContent
        liveSessions={live}
        project={overlaid}
        removedSessionIds={removed}
        renderRows={renderRows}
        repoWorktrees={{ '/workspace': worktrees }}
      />
    )

    expect(membership(), project.label).toEqual(expectedById[project.id])

    if (project.id === 'hermes') {
      expect(screen.getByTitle(/\/other\/empty/)).toBeTruthy()
    }

    view.unmount()
  }
})
