import { describe, expect, it } from 'vitest'

import type { HermesWorktreeInfo } from '@/global'
import type { SessionInfo } from '@/types/hermes'

import { uniqueCwds, workspaceGroupsFor, workspaceTreeFor, type WorktreeResolver } from './workspace-groups'

let nextId = 0

function makeSession(cwd: null | string, overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: false,
    cwd,
    ended_at: null,
    id: `s${nextId++}`,
    input_tokens: 0,
    is_active: false,
    last_active: 1_000,
    message_count: 1,
    model: 'claude',
    output_tokens: 0,
    preview: null,
    source: 'cli',
    started_at: 1_000,
    title: null,
    tool_call_count: 0,
    ...overrides
  }
}

const labels = (sessions: SessionInfo[]) => workspaceGroupsFor(sessions, 'No workspace').map(g => g.label)

describe('workspaceGroupsFor', () => {
  it('sorts rows within a group by last activity, newest first (#42707)', () => {
    const stale = makeSession('/work/proj', { id: 'stale', started_at: 100, last_active: 100 })
    const fresh = makeSession('/work/proj', { id: 'fresh', started_at: 50, last_active: 900 })

    const [group] = workspaceGroupsFor([stale, fresh], 'No workspace')

    // fresh has the later last_active despite an earlier started_at.
    expect(group.sessions.map(s => s.id)).toEqual(['fresh', 'stale'])
  })

  it('groups by full cwd, not by basename — same-named folders are separate groups', () => {
    const groups = workspaceGroupsFor(
      [makeSession('/a/hermes-agent/apps/desktop'), makeSession('/a/hermes-agent-wt-rtl/apps/desktop')],
      'No workspace'
    )

    expect(groups).toHaveLength(2)
  })

  it('disambiguates colliding basenames by walking up the path', () => {
    expect(
      labels([makeSession('/a/hermes-agent/apps/desktop'), makeSession('/a/hermes-agent-wt-rtl/apps/desktop')])
    ).toEqual(['hermes-agent/apps/desktop', 'hermes-agent-wt-rtl/apps/desktop'])
  })

  it('leaves a unique basename as its short label', () => {
    expect(labels([makeSession('/a/hermes-agent/apps/desktop'), makeSession('/b/heval-py')])).toEqual([
      'desktop',
      'heval-py'
    ])
  })

  it('grows the prefix past one segment when the parent also collides', () => {
    expect(labels([makeSession('/x/proj/apps/desktop'), makeSession('/y/proj/apps/desktop')])).toEqual([
      'x/proj/apps/desktop',
      'y/proj/apps/desktop'
    ])
  })

  it('keeps the synthetic no-workspace group untouched even if a real group shares its label', () => {
    const groups = workspaceGroupsFor([makeSession(null), makeSession('/a/No workspace')], 'No workspace')
    const noWorkspace = groups.find(g => g.path === null)

    expect(noWorkspace?.label).toBe('No workspace')
  })
})

const info = (over: Partial<HermesWorktreeInfo> & Pick<HermesWorktreeInfo, 'repoRoot' | 'worktreeRoot'>): HermesWorktreeInfo => ({
  branch: null,
  isMainWorktree: false,
  ...over
})

describe('workspaceTreeFor', () => {
  it('heuristic nests `<repo>-wt-<branch>` under its sibling repo', () => {
    const tree = workspaceTreeFor(
      [makeSession('/www/hermes-agent'), makeSession('/www/hermes-agent-wt-rtl')],
      'No workspace'
    )

    expect(tree).toHaveLength(1)
    expect(tree[0].label).toBe('hermes-agent')
    expect(tree[0].groups.map(g => g.label).sort()).toEqual(['hermes-agent', 'rtl'])
  })

  it('git metadata is authoritative — worktrees group by repoRoot regardless of directory naming', () => {
    const resolver: WorktreeResolver = cwd => {
      if (cwd === '/www/hermes-agent') {
        return info({ repoRoot: '/www/hermes-agent', worktreeRoot: '/www/hermes-agent', isMainWorktree: true, branch: 'main' })
      }

      if (cwd === '/elsewhere/ha-rtl') {
        return info({ repoRoot: '/www/hermes-agent', worktreeRoot: '/elsewhere/ha-rtl', branch: 'rtl' })
      }

      return null
    }

    const tree = workspaceTreeFor(
      [makeSession('/www/hermes-agent'), makeSession('/elsewhere/ha-rtl')],
      'No workspace',
      resolver
    )

    expect(tree).toHaveLength(1)
    expect(tree[0].label).toBe('hermes-agent')
    // The main checkout labels by directory (its branch is transient — using it
    // would misattribute old sessions to the currently checked-out branch);
    // linked worktrees label by branch.
    expect(tree[0].groups.map(g => g.label)).toEqual(['hermes-agent', 'rtl'])
  })

  it('a standalone directory is its own parent (always parent → worktree → sessions)', () => {
    const tree = workspaceTreeFor([makeSession('/www/heval-node')], 'No workspace')

    expect(tree).toHaveLength(1)
    expect(tree[0].label).toBe('heval-node')
    expect(tree[0].groups).toHaveLength(1)
    expect(tree[0].groups[0].label).toBe('heval-node')
  })

  it('aggregates session counts across a repo’s worktrees', () => {
    const tree = workspaceTreeFor(
      [makeSession('/www/ha'), makeSession('/www/ha-wt-x'), makeSession('/www/ha-wt-x')],
      'No workspace'
    )

    const parent = tree.find(p => p.label === 'ha')

    expect(parent?.sessionCount).toBe(3)
  })

  it('no-workspace sessions form their own parent', () => {
    const tree = workspaceTreeFor([makeSession(null)], 'No workspace')

    expect(tree).toHaveLength(1)
    expect(tree[0].label).toBe('No workspace')
    expect(tree[0].path).toBeNull()
  })
})

describe('uniqueCwds', () => {
  it('dedupes and drops empty/whitespace cwds', () => {
    expect(uniqueCwds([makeSession('/a'), makeSession('/a'), makeSession(null), makeSession('   ')])).toEqual(['/a'])
  })
})
