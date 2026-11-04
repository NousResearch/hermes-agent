import { describe, expect, it } from 'vitest'

import type { HermesGitWorktree } from '@/global'
import type { SessionInfo } from '@/hermes'

import { mergeRepoWorktreeGroups, overlayRepoLanes } from './workspace-groups'

const REPO = '/repo'

function session(id: string, lastActive: number, cwd: null | string = REPO): SessionInfo {
  return {
    id,
    cwd,
    git_repo_root: REPO,
    git_branch: null,
    last_active: lastActive,
    started_at: lastActive,
    source: 'desktop',
    title: id
  } as unknown as SessionInfo
}

const worktrees: HermesGitWorktree[] = [{ path: REPO, branch: 'deployed/x', isMain: true } as HermesGitWorktree]

describe('entered-project home lane keeps recency after the live overlay re-merge', () => {
  it('newest rows stay on the first page (not appended after the backend tail)', () => {
    // Backend main lane, newest first (as projects.project_sessions emits).
    const backend = [session('new-1', 500), session('new-2', 400), session('old-1', 100), session('old-2', 50)]
    const repo = {
      id: REPO,
      path: REPO,
      label: 'repo',
      sessionCount: backend.length,
      groups: [{ id: `${REPO}::branch::main`, label: 'main', isMain: true, path: REPO, sessions: backend }]
    }

    // 1st merge: fold into home lane (entered-content.tsx mergedGroups)
    const merged = mergeRepoWorktreeGroups(repo, worktrees)
    // overlay with the live page (only the recent ones are in $sessions)
    const live = [session('new-1', 500), session('new-2', 400)]
    const { groups } = overlayRepoLanes({ ...repo, groups: merged }, live)
    // 2nd merge (entered-content.tsx overlaidGroups)
    const final = mergeRepoWorktreeGroups({ id: repo.id, path: repo.path, groups }, worktrees)
    const home = final.find(g => g.isHome)!

    expect(home.sessions.map(s => s.id)).toEqual(['new-1', 'new-2', 'old-1', 'old-2'])
  })
})
