import { describe, expect, it } from 'vitest'

import type { HermesGitWorktree } from '@/global'
import type { ProjectInfo, SessionInfo } from '@/types/hermes'

import {
  baseName,
  kanbanWorktreeDir,
  liveSessionProjectId,
  mergeRepoWorktreeGroups,
  overlayLiveLanes,
  overlayLivePreviews,
  sessionProjectColor,
  type SidebarProjectTree,
  type SidebarSessionGroup,
  sortWorktreeGroups
} from './workspace-groups'

// The grouping itself now lives on the backend (tui_gateway/project_tree.py,
// covered by tests/tui_gateway/test_project_tree.py). This file only covers the
// thin render helpers the desktop still owns + the VISUAL worktree enhancer.

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

const lane = (over: Partial<SidebarSessionGroup> & Pick<SidebarSessionGroup, 'id' | 'label'>): SidebarSessionGroup => ({
  path: null,
  sessions: [],
  ...over
})

describe('baseName', () => {
  it('returns the final path segment, ignoring trailing slashes and separators', () => {
    expect(baseName('/www/hermes-agent/')).toBe('hermes-agent')
    expect(baseName('C:\\repos\\app')).toBe('app')
    expect(baseName('')).toBeUndefined()
  })
})

describe('kanbanWorktreeDir', () => {
  it('matches a kanban task worktree (t_<hex>) and returns its .worktrees dir', () => {
    expect(kanbanWorktreeDir('/repo/.worktrees/t_aaaaaaaa')).toBe('/repo/.worktrees')
  })

  it('does NOT match a user-named "New worktree" under .worktrees/ (its own lane)', () => {
    expect(kanbanWorktreeDir('/repo/.worktrees/test-gui-stuff')).toBeNull()
  })

  it('returns null for non-kanban paths', () => {
    expect(kanbanWorktreeDir('/repo/src')).toBeNull()
    expect(kanbanWorktreeDir('/repo')).toBeNull()
  })
})

describe('sortWorktreeGroups', () => {
  it('pins trunk to the top, sinks kanban to the bottom, and orders the rest by recency', () => {
    const at = (t: number) => [makeSession('/x', { last_active: t })]

    const groups = [
      lane({ id: 'k', label: 'kanban', isKanban: true, sessions: at(999) }),
      lane({ id: 'stale', label: 'stale-branch', isMain: true, sessions: at(10) }),
      lane({ id: 'wt', label: 'busy-worktree', isMain: false, sessions: at(500) }),
      lane({ id: 'main', label: 'main', isMain: true, sessions: at(1) })
    ]

    // main (trunk) first despite being least recent; kanban last despite being
    // most recent; busy-worktree ahead of stale-branch by activity.
    expect(sortWorktreeGroups(groups).map(g => g.label)).toEqual(['main', 'busy-worktree', 'stale-branch', 'kanban'])
  })

  it('pins the live home checkout above trunk, even when it has no sessions yet', () => {
    const groups = [
      lane({ id: 'main', label: 'main', isMain: true, sessions: [makeSession('/x', { last_active: 999 })] }),
      lane({ id: 'home', label: 'bb/projects-paradigm', isMain: true, isHome: true })
    ]

    expect(sortWorktreeGroups(groups).map(g => g.label)).toEqual(['bb/projects-paradigm', 'main'])
  })

  it('falls back to label order for equally-idle lanes', () => {
    const groups = [lane({ id: 'b', label: 'beta', isMain: false }), lane({ id: 'a', label: 'alpha', isMain: false })]

    expect(sortWorktreeGroups(groups).map(g => g.label)).toEqual(['alpha', 'beta'])
  })
})

describe('mergeRepoWorktreeGroups (visual enhancer)', () => {
  it('injects a linked worktree lane discovered by git that has no sessions yet', () => {
    const repo = {
      id: '/repo',
      path: '/repo',
      groups: [lane({ id: '/repo::branch::main', label: 'main', isMain: true, path: '/repo' })]
    }

    const discovered: HermesGitWorktree[] = [
      { branch: 'feature', detached: false, isMain: false, locked: false, path: '/repo-wt-feature' }
    ]

    const merged = mergeRepoWorktreeGroups(repo, discovered)

    expect(merged.map(g => g.label)).toEqual(['main', 'feature'])
    // The injected lane is empty (visual only — never carries sessions).
    expect(merged.find(g => g.label === 'feature')?.sessions).toEqual([])
  })

  it('never spawns a lane per kanban task worktree', () => {
    const repo = {
      id: '/repo',
      path: '/repo',
      groups: [lane({ id: '/repo::branch::main', label: 'main', isMain: true, path: '/repo' })]
    }

    const discovered: HermesGitWorktree[] = [
      { branch: 'wt/t_aaaaaaaa', detached: false, isMain: false, locked: false, path: '/repo/.worktrees/t_aaaaaaaa' },
      { branch: 'wt/t_bbbbbbbb', detached: false, isMain: false, locked: false, path: '/repo/.worktrees/t_bbbbbbbb' }
    ]

    expect(mergeRepoWorktreeGroups(repo, discovered).map(g => g.label)).toEqual(['main'])
  })

  it('does not duplicate a lane already present from the backend (by id/path)', () => {
    const repo = {
      id: '/repo',
      path: '/repo',
      groups: [
        lane({
          id: '/repo::branch::main',
          label: 'main',
          isMain: true,
          path: '/repo',
          sessions: [makeSession('/repo')]
        })
      ]
    }

    const discovered: HermesGitWorktree[] = [
      { branch: 'main', detached: false, isMain: true, locked: false, path: '/repo' }
    ]

    const merged = mergeRepoWorktreeGroups(repo, discovered)

    expect(merged).toHaveLength(1)
    // The backend lane keeps its session rows; the enhancer left it untouched.
    expect(merged[0].sessions).toHaveLength(1)
  })

  it('is a no-op when git worktree list is unavailable (remote backend)', () => {
    const groups = [lane({ id: '/repo::branch::main', label: 'main', isMain: true, path: '/repo' })]

    expect(mergeRepoWorktreeGroups({ id: '/repo', path: '/repo', groups }, undefined).map(g => g.label)).toEqual([
      'main'
    ])
  })

  it('does not add a second "main" for a linked worktree checked out on main', () => {
    const groups = [
      lane({ id: '/repo::branch::main', label: 'main', isMain: true, path: '/repo', sessions: [makeSession('/repo')] })
    ]

    const discovered: HermesGitWorktree[] = [
      { branch: 'main', detached: false, isMain: false, locked: false, path: '/repo/.worktrees/main-mirror' }
    ]

    expect(
      mergeRepoWorktreeGroups({ id: '/repo', path: '/repo', groups }, discovered).filter(g => g.label === 'main')
    ).toHaveLength(1)
  })

  it('surfaces a user-named "New worktree" under .worktrees/ as its own lane', () => {
    const discovered: HermesGitWorktree[] = [
      {
        branch: 'hermes/test-gui-stuff',
        detached: false,
        isMain: false,
        locked: false,
        path: '/repo/.worktrees/test-gui-stuff'
      }
    ]

    const merged = mergeRepoWorktreeGroups({ id: '/repo', path: '/repo', groups: [] }, discovered)

    expect(merged.map(g => g.label)).toContain('hermes/test-gui-stuff')
  })

  it('relabels a dir-named linked worktree lane to its live checked-out branch', () => {
    // Backend labels the lane by the worktree dir (`hermes-agent-ci`); the live
    // `git worktree list` says HEAD there is `bb/ci-affected-only` → branch wins.
    const repo = {
      id: '/repo',
      path: '/repo',
      groups: [
        lane({
          id: '/repo::branch::main',
          label: 'main',
          isMain: true,
          path: '/repo',
          sessions: [makeSession('/repo')]
        }),
        lane({
          id: '/repo-ci',
          label: 'hermes-agent-ci',
          isMain: false,
          path: '/repo-ci',
          sessions: [makeSession('/repo-ci')]
        })
      ]
    }

    const discovered: HermesGitWorktree[] = [
      { branch: 'main', detached: false, isMain: true, locked: false, path: '/repo' },
      { branch: 'bb/ci-affected-only', detached: false, isMain: false, locked: false, path: '/repo-ci' }
    ]

    const merged = mergeRepoWorktreeGroups(repo, discovered)
    const ci = merged.find(g => g.id === '/repo-ci')

    expect(ci?.label).toBe('bb/ci-affected-only')
    // The relabel is label-only — the lane keeps its id, path, and sessions.
    expect(ci?.path).toBe('/repo-ci')
    expect(ci?.sessions).toHaveLength(1)
  })

  it('re-anchors a lane whose path drifted from git truth back to its branch path', () => {
    // The reported bug: a lane is correctly labeled by its branch (`bb/attempts`)
    // but its stored PATH points at a stale/old worktree dir. git pins a branch
    // to exactly one worktree, so the lane must follow the branch's real path —
    // otherwise "reveal in Finder" opens a completely different worktree.
    const repo = {
      id: '/repo',
      path: '/repo',
      groups: [
        lane({
          id: '/repo/.worktrees/attempts',
          label: 'bb/attempts',
          isMain: false,
          path: '/repo/.worktrees/attempts',
          sessions: [makeSession('/repo/.worktrees/attempts')]
        })
      ]
    }

    // git now has `bb/attempts` at a sibling dir, not the stale `.worktrees` one.
    const discovered: HermesGitWorktree[] = [
      { branch: 'bb/attempts', detached: false, isMain: false, locked: false, path: '/repo-pr-attempts' }
    ]

    const merged = mergeRepoWorktreeGroups(repo, discovered)
    const attempts = merged.filter(g => g.label === 'bb/attempts')

    // Exactly one lane, re-pointed at git's real path (label preserved, sessions
    // preserved), and NO leftover lane on the stale path.
    expect(attempts).toHaveLength(1)
    expect(attempts[0].path).toBe('/repo-pr-attempts')
    expect(attempts[0].sessions).toHaveLength(1)
    expect(merged.some(g => g.path === '/repo/.worktrees/attempts')).toBe(false)
  })

  it('collapses a re-anchored lane onto the real lane that already holds that path', () => {
    // A stale lane (branch label, wrong path) AND the real worktree lane both
    // exist. Re-anchoring the stale one onto git's path must not leave a twin —
    // keep the richer (more sessions) lane.
    const repo = {
      id: '/repo',
      path: '/repo',
      groups: [
        lane({ id: 'stale', label: 'bb/feature', isMain: false, path: '/repo/.worktrees/old', sessions: [] }),
        lane({
          id: '/repo-feature',
          label: 'bb/feature',
          isMain: false,
          path: '/repo-feature',
          sessions: [makeSession('/repo-feature'), makeSession('/repo-feature')]
        })
      ]
    }

    const discovered: HermesGitWorktree[] = [
      { branch: 'bb/feature', detached: false, isMain: false, locked: false, path: '/repo-feature' }
    ]

    const merged = mergeRepoWorktreeGroups(repo, discovered)
    const feature = merged.filter(g => g.path === '/repo-feature')

    expect(feature).toHaveLength(1)
    expect(feature[0].sessions).toHaveLength(2)
    expect(merged.some(g => g.path === '/repo/.worktrees/old')).toBe(false)
  })

  it('keeps the dir label for a detached-HEAD worktree (no branch to show)', () => {
    const repo = {
      id: '/repo',
      path: '/repo',
      groups: [
        lane({ id: '/repo-ci', label: 'repo-ci', isMain: false, path: '/repo-ci', sessions: [makeSession('/repo-ci')] })
      ]
    }

    const discovered: HermesGitWorktree[] = [
      { branch: null, detached: true, isMain: false, locked: false, path: '/repo-ci' }
    ]

    expect(mergeRepoWorktreeGroups(repo, discovered).find(g => g.id === '/repo-ci')?.label).toBe('repo-ci')
  })

  it('collapses the main checkout into one home lane labeled by the live branch (off-trunk)', () => {
    const repo = {
      id: '/repo',
      path: '/repo',
      groups: [
        lane({
          id: '/repo::branch::main',
          label: 'main',
          isMain: true,
          path: '/repo',
          sessions: [makeSession('/repo')]
        })
      ]
    }

    // The repo root is switched to a feature branch. The historical "main"
    // sessions fold into ONE home lane labeled by the live branch — no stale
    // "main" lane lingering beside it.
    const discovered: HermesGitWorktree[] = [
      { branch: 'some-feature', detached: false, isMain: true, locked: false, path: '/repo' }
    ]

    const merged = mergeRepoWorktreeGroups(repo, discovered)
    const home = merged.find(g => g.isHome)

    expect(merged.filter(g => g.isMain)).toHaveLength(1)
    expect(home?.label).toBe('some-feature')
    expect(home?.path).toBe('/repo')
    expect(home?.sessions).toHaveLength(1)
    expect(merged.some(g => g.label === 'main')).toBe(false)
  })

  it('labels the home lane "main" (still home-flagged) when the root is on trunk', () => {
    const repo = {
      id: '/repo',
      path: '/repo',
      groups: [
        lane({
          id: '/repo::branch::main',
          label: 'main',
          isMain: true,
          path: '/repo',
          sessions: [makeSession('/repo')]
        })
      ]
    }

    const discovered: HermesGitWorktree[] = [
      { branch: 'main', detached: false, isMain: true, locked: false, path: '/repo' }
    ]

    const home = mergeRepoWorktreeGroups(repo, discovered).find(g => g.isHome)

    expect(home?.label).toBe('main')
    expect(home?.isHome).toBe(true)
  })

  it('folds multiple historical main-checkout branch lanes into the single live home lane', () => {
    const repo = {
      id: '/repo',
      path: '/repo',
      groups: [
        lane({
          id: '/repo::branch::main',
          label: 'main',
          isMain: true,
          path: '/repo',
          sessions: [makeSession('/repo', { id: 'a' })]
        }),
        lane({
          id: '/repo::branch::old',
          label: 'old-feature',
          isMain: true,
          path: '/repo',
          sessions: [makeSession('/repo', { id: 'b' })]
        })
      ]
    }

    const discovered: HermesGitWorktree[] = [
      { branch: 'bb/live', detached: false, isMain: true, locked: false, path: '/repo' }
    ]

    const merged = mergeRepoWorktreeGroups(repo, discovered)
    const home = merged.find(g => g.isHome)

    expect(merged.filter(g => g.isMain)).toHaveLength(1)
    expect(home?.label).toBe('bb/live')
    expect(home?.sessions.map(s => s.id).sort()).toEqual(['a', 'b'])
  })

  it('leaves main lanes untouched on a remote backend (no git probe)', () => {
    const repo = {
      id: '/repo',
      path: '/repo',
      groups: [
        lane({
          id: '/repo::branch::main',
          label: 'main',
          isMain: true,
          path: '/repo',
          sessions: [makeSession('/repo')]
        })
      ]
    }

    // No discovered worktrees → no live branch truth → backend label stands.
    const merged = mergeRepoWorktreeGroups(repo, undefined)

    expect(merged.map(g => g.label)).toEqual(['main'])
    expect(merged[0].isHome).toBeFalsy()
  })
})

const makeProject = (id: string, folders: string[]): ProjectInfo => ({
  archived: false,
  board_slug: null,
  color: null,
  created_at: 0,
  description: null,
  folders: folders.map((path, i) => ({ added_at: 0, is_primary: i === 0, label: null, path })),
  icon: null,
  id,
  name: id,
  primary_path: folders[0] ?? null,
  slug: id
})

const projectNode = (over: Partial<SidebarProjectTree> & Pick<SidebarProjectTree, 'id'>): SidebarProjectTree => ({
  label: over.id,
  path: over.id,
  repos: [],
  sessionCount: 0,
  ...over
})

describe('liveSessionProjectId', () => {
  it('maps a brand-new (unpersisted) session to its auto project (the repo root)', () => {
    expect(liveSessionProjectId(makeSession('/www/app'), [])).toBe('/www/app')
  })

  it('routes a session under an explicit project folder to that project', () => {
    const id = liveSessionProjectId(makeSession('/www/app/src', { git_repo_root: '/www/app', git_branch: 'feat' }), [
      makeProject('p_app', ['/www/app'])
    ])

    expect(id).toBe('p_app')
  })

  it('anchors a cwd-less session on its git_repo_root (backend groups it there too)', () => {
    // Older/imported rows carry only a repo root; the sidebar files them under
    // the repo's project, so membership (and color) must resolve from the root.
    expect(liveSessionProjectId(makeSession(null, { git_repo_root: '/www/app' }), [])).toBe('/www/app')
    expect(
      liveSessionProjectId(makeSession(null, { git_repo_root: '/www/app' }), [makeProject('p_app', ['/www/app'])])
    ).toBe('p_app')
  })

  it('skips cwd-less, kanban-task, and out-of-tree (sibling) worktree sessions', () => {
    expect(liveSessionProjectId(makeSession(null), [])).toBeNull()
    // Kanban task worktree → folds into the kanban bucket, not a project preview.
    expect(liveSessionProjectId(makeSession('/repo/.worktrees/t_aaaaaaaa'), [])).toBeNull()
    // Sibling worktree OUTSIDE the repo root → project can't be derived from the row.
    expect(liveSessionProjectId(makeSession('/elsewhere/wt', { git_repo_root: '/repo' }), [])).toBeNull()
  })

  it('places an in-tree worktree session under its repo project (the root is in the path)', () => {
    // "Convert a branch" / "new worktree" land at `<repoRoot>/.worktrees/<slug>`,
    // so they belong to the same auto project as the repo root and must show in
    // the overview at once, not wait for the next backend refresh.
    expect(liveSessionProjectId(makeSession('/www/app/.worktrees/test1', { git_repo_root: '/www/app' }), [])).toBe(
      '/www/app'
    )
  })

  it('routes an in-tree worktree session to the owning explicit project', () => {
    const id = liveSessionProjectId(makeSession('/www/app/.worktrees/test1', { git_repo_root: '/www/app' }), [
      makeProject('p_app', ['/www/app'])
    ])

    expect(id).toBe('p_app')
  })

  it('places a cwd-outside-root session under an explicit project matching either path', () => {
    // A mid-session relocation (or a sibling worktree) leaves cwd outside the
    // recorded repo root. An explicit folder match is still authoritative —
    // only the auto-project (repo root) fallback needs cwd-under-root
    // confidence. Match via the repo root...
    expect(
      liveSessionProjectId(makeSession('/www/elsewhere', { git_repo_root: '/home/u/proj' }), [
        makeProject('p_proj', ['/home/u/proj'])
      ])
    ).toBe('p_proj')
    // ...and via the cwd.
    expect(
      liveSessionProjectId(makeSession('/www/elsewhere/sub', { git_repo_root: '/home/u/proj' }), [
        makeProject('p_www', ['/www/elsewhere'])
      ])
    ).toBe('p_www')
  })

  it('matches a mixed-case/separator Windows cwd to its explicit project in the live overlay', () => {
    // The bug: a fresh Windows session drops into the overlay before the next
    // backend refresh; case-sensitive matching missed its project until then.
    const id = liveSessionProjectId(makeSession('c:/work/notes/SUB'), [makeProject('p_notes', ['C:\\Work\\Notes'])])

    expect(id).toBe('p_notes')
  })

  it('matches a root-relative WSL cwd (single backslash) case-insensitively', () => {
    const id = liveSessionProjectId(makeSession('//wsl.localhost/Ubuntu/home/alice/PROJ'), [
      makeProject('p_proj', ['\\wsl.localhost\\Ubuntu\\home\\alice\\proj'])
    ])

    expect(id).toBe('p_proj')
  })

  it('keeps POSIX cwd matching case-sensitive (no false project match)', () => {
    // Distinct case on POSIX is a distinct path → falls back to its own auto id.
    expect(liveSessionProjectId(makeSession('/work/notes'), [makeProject('p_notes', ['/Work/Notes'])])).toBe(
      '/work/notes'
    )
  })
})

describe('sessionProjectColor', () => {
  const colored = (id: string, folders: string[], color: string): ProjectInfo => ({
    ...makeProject(id, folders),
    color
  })

  it('inherits the color of the explicit project the session belongs to', () => {
    const session = makeSession('/www/app/src', { git_repo_root: '/www/app' })

    expect(sessionProjectColor(session, [colored('p_app', ['/www/app'], '#4a9eff')])).toBe('#4a9eff')
  })

  it('returns null when the owning project has no color set', () => {
    const session = makeSession('/www/app/src', { git_repo_root: '/www/app' })

    expect(sessionProjectColor(session, [makeProject('p_app', ['/www/app'])])).toBeNull()
  })

  it('colors a cwd-less session by its git_repo_root project (the grouped-but-grey fix)', () => {
    const session = makeSession(null, { git_repo_root: '/www/app' })

    expect(sessionProjectColor(session, [colored('p_app', ['/www/app'], '#4a9eff')])).toBe('#4a9eff')
  })

  it('colors a cwd-outside-root session when an explicit project folder matches', () => {
    // The backend tree groups such a row under the project; the client color
    // derivation must agree instead of leaving the row (and its tab) grey.
    const session = makeSession('/www/elsewhere', { git_repo_root: '/home/u/proj' })

    expect(sessionProjectColor(session, [colored('p_proj', ['/home/u/proj'], '#4a9eff')])).toBe('#4a9eff')
  })

  it('returns null for a session that only maps to an auto repo root (no explicit project)', () => {
    // liveSessionProjectId falls back to the repo root id, which is not a
    // project row and therefore carries no color.
    expect(sessionProjectColor(makeSession('/www/app'), [])).toBeNull()
  })

  it('returns null for an unplaceable (cwd-less) session', () => {
    expect(sessionProjectColor(makeSession(null), [colored('p_app', ['/www/app'], '#4a9eff')])).toBeNull()
  })

  it('uses the longest-prefix project when nested projects both match', () => {
    const session = makeSession('/www/app/packages/api/src', { git_repo_root: '/www/app' })

    const projects = [
      colored('p_root', ['/www/app'], '#111111'),
      colored('p_api', ['/www/app/packages/api'], '#222222')
    ]

    expect(sessionProjectColor(session, projects)).toBe('#222222')
  })
})

describe('overlayLiveLanes', () => {
  it('injects a live session into the matching main lane instantly', () => {
    const project = projectNode({
      id: '/www/app',
      isAuto: true,
      repos: [{ id: '/www/app', label: 'app', path: '/www/app', sessionCount: 0, groups: [] }]
    })

    const live = [makeSession('/www/app', { id: 'fresh', git_branch: 'main' })]

    const overlaid = overlayLiveLanes(project, live)
    const lane = overlaid.repos[0].groups.find(g => g.label === 'main')

    expect(lane?.sessions.map(session => session.id)).toContain('fresh')
    expect(overlaid.sessionCount).toBe(1)
  })

  it('injects a session created in a fresh worktree into that worktree lane (no git_repo_root yet)', () => {
    // The brand-new session row has only a cwd — no git_repo_root. The entered
    // project knows its repo root, so the worktree session still lands in its
    // own lane (not kanban, not skipped) optimistically.
    const project = projectNode({
      id: '/www/app',
      isAuto: true,
      repos: [{ id: '/www/app', label: 'app', path: '/www/app', sessionCount: 0, groups: [] }]
    })

    const live = [makeSession('/www/app/.worktrees/baby', { id: 'fresh' })]

    const overlaid = overlayLiveLanes(project, live)
    const lane = overlaid.repos[0].groups.find(g => g.id === '/www/app/.worktrees/baby')

    expect(lane?.label).toBe('baby')
    expect(lane?.sessions.map(s => s.id)).toEqual(['fresh'])
  })

  it('folds a kanban-task worktree session into the kanban lane', () => {
    const project = projectNode({
      id: '/www/app',
      isAuto: true,
      repos: [{ id: '/www/app', label: 'app', path: '/www/app', sessionCount: 0, groups: [] }]
    })

    const live = [makeSession('/www/app/.worktrees/t_abc12345', { id: 'k' })]

    const overlaid = overlayLiveLanes(project, live)
    const lane = overlaid.repos[0].groups.find(g => g.isKanban)

    expect(lane?.id).toBe('/www/app::kanban')
    expect(lane?.sessions.map(s => s.id)).toEqual(['k'])
  })

  it('does not duplicate a session already present in a backend lane', () => {
    const existing = makeSession('/www/app', { id: 'dup', git_branch: 'main' })

    const project = projectNode({
      id: '/www/app',
      repos: [
        {
          id: '/www/app',
          label: 'app',
          path: '/www/app',
          sessionCount: 1,
          groups: [
            lane({ id: '/www/app::branch::main', label: 'main', isMain: true, path: '/www/app', sessions: [existing] })
          ]
        }
      ]
    })

    const overlaid = overlayLiveLanes(project, [existing])

    expect(overlaid.repos[0].groups.flatMap(g => g.sessions.map(s => s.id))).toEqual(['dup'])
  })

  it('adds a new session to an existing worktree lane keyed by a divergent id (matches by path)', () => {
    // Backend keyed the worktree lane off a branch-style id (no live git probe),
    // but the lane PATH is the worktree dir. A new session under that worktree
    // must join the existing lane, not spawn a twin.
    const existing = makeSession('/www/app/.worktrees/baby', { id: 'old' })

    const project = projectNode({
      id: '/www/app',
      repos: [
        {
          id: '/www/app',
          label: 'app',
          path: '/www/app',
          sessionCount: 1,
          groups: [
            lane({
              id: '/www/app::branch::baby',
              label: 'baby',
              path: '/www/app/.worktrees/baby',
              sessions: [existing]
            })
          ]
        }
      ]
    })

    const fresh = makeSession('/www/app/.worktrees/baby', { id: 'fresh' })

    const overlaid = overlayLiveLanes(project, [existing, fresh])
    const lanes = overlaid.repos[0].groups.filter(g => g.path === '/www/app/.worktrees/baby')

    expect(lanes).toHaveLength(1)
    expect(lanes[0].sessions.map(s => s.id).sort()).toEqual(['fresh', 'old'])
  })

  it('places a session into an out-of-tree (sibling) worktree lane by its path', () => {
    // `hermes-agent-ci` is a linked worktree living BESIDE the repo, not under
    // it — repo-root nesting fails, but the existing lane carries its real path.
    const existing = makeSession('/www/app-ci', { id: 'old' })

    const project = projectNode({
      id: '/www/app',
      repos: [
        {
          id: '/www/app',
          label: 'app',
          path: '/www/app',
          sessionCount: 1,
          groups: [
            lane({ id: '/www/app::branch::main', label: 'main', isMain: true, path: '/www/app', sessions: [] }),
            lane({ id: '/www/app-ci', label: 'app-ci', path: '/www/app-ci', sessions: [existing] })
          ]
        }
      ]
    })

    const fresh = makeSession('/www/app-ci', { id: 'fresh' })

    const overlaid = overlayLiveLanes(project, [existing, fresh])
    const ci = overlaid.repos[0].groups.find(g => g.path === '/www/app-ci')
    const main = overlaid.repos[0].groups.find(g => g.label === 'main')

    expect(ci?.sessions.map(s => s.id).sort()).toEqual(['fresh', 'old'])
    expect(main?.sessions ?? []).toHaveLength(0)
  })

  it('places into a visual-only discovered worktree lane after merge', () => {
    const discovered = [
      { path: '/www/app-retry', branch: 'bb/ci-install-retry', isMain: false, detached: false, locked: false }
    ]

    const groups = mergeRepoWorktreeGroups({ id: '/www/app', path: '/www/app', groups: [] }, discovered)

    const project = projectNode({
      id: '/www/app',
      repos: [{ id: '/www/app', label: 'app', path: '/www/app', sessionCount: 0, groups }]
    })

    const fresh = makeSession('/www/app-retry', { id: 'fresh' })

    const overlaid = overlayLiveLanes(project, [fresh])
    const lane = overlaid.repos[0].groups.find(g => g.path === '/www/app-retry')

    expect(lane?.sessions.map(s => s.id)).toEqual(['fresh'])
  })

  it('evicts a deleted/archived snapshot row (and drops the lane once empty)', () => {
    const a = makeSession('/www/app', { id: 'keep', git_branch: 'main' })
    const b = makeSession('/www/app/.worktrees/baby', { id: 'gone' })

    const project = projectNode({
      id: '/www/app',
      repos: [
        {
          id: '/www/app',
          label: 'app',
          path: '/www/app',
          sessionCount: 2,
          groups: [
            lane({ id: '/www/app::branch::main', label: 'main', isMain: true, path: '/www/app', sessions: [a] }),
            lane({ id: '/www/app/.worktrees/baby', label: 'baby', path: '/www/app/.worktrees/baby', sessions: [b] })
          ]
        }
      ]
    })

    // No live rows (both deleted from $sessions); only 'gone' is tombstoned.
    const overlaid = overlayLiveLanes(project, [a], new Set(['gone']))

    expect(overlaid.repos[0].groups.map(g => g.id)).toEqual(['/www/app::branch::main'])
    expect(overlaid.repos[0].groups[0].sessions.map(s => s.id)).toEqual(['keep'])
    expect(overlaid.sessionCount).toBe(1)
  })
})

describe('overlayLivePreviews', () => {
  it('merges live sessions into a project preview, live first, capped to the limit', () => {
    const project = projectNode({
      id: '/www/app',
      previewSessions: [makeSession('/www/app', { id: 'old', started_at: 1, last_active: 1 })]
    })

    const live = [makeSession('/www/app', { id: 'fresh', started_at: 99, last_active: 99 })]

    const previews = overlayLivePreviews([project], live, [], 3)

    expect(previews['/www/app'].map(s => s.id)).toEqual(['fresh', 'old'])
  })

  it('evicts a deleted session from a project preview (snapshot + live)', () => {
    const project = projectNode({
      id: '/www/app',
      previewSessions: [
        makeSession('/www/app', { id: 'gone', started_at: 5, last_active: 5 }),
        makeSession('/www/app', { id: 'old', started_at: 1, last_active: 1 })
      ]
    })

    const previews = overlayLivePreviews([project], [], [], 3, new Set(['gone']))

    expect(previews['/www/app'].map(s => s.id)).toEqual(['old'])
  })
})
