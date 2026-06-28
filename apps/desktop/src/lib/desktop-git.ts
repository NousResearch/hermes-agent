import type {
  HermesGitBranch,
  HermesGitWorktree,
  HermesRepoStatus,
  HermesReviewList,
  HermesReviewShipInfo
} from '@/global'

import { isDesktopFsRemoteMode } from './desktop-fs'

// Remote-aware git facade. Locally the desktop runs git through Electron
// (window.hermesDesktop.git); on a remote gateway that's the wrong filesystem,
// so we mirror the same surface over the dashboard REST API (/api/git/*) — the
// coding rail, worktree lanes, review pane, and branch ops then act on the
// BACKEND repo where sessions actually run. Mirrors desktop-fs.ts.

type GitBridge = NonNullable<NonNullable<Window['hermesDesktop']>['git']>

const q = (value: string) => encodeURIComponent(value)

function desktopApi<T>(path: string, body?: Record<string, unknown>): Promise<T> {
  const desktop = window.hermesDesktop

  if (!desktop) {
    throw new Error('Hermes Desktop bridge is unavailable')
  }

  return desktop.api<T>(body ? { body, method: 'POST', path } : { path })
}

const remoteGit: GitBridge = {
  worktreeList: async repoPath =>
    (await desktopApi<{ worktrees: HermesGitWorktree[] }>(`/api/git/worktrees?path=${q(repoPath)}`)).worktrees,

  worktreeAdd: (repoPath, options) => desktopApi(`/api/git/worktree/add`, { path: repoPath, ...options }),

  worktreeRemove: (repoPath, worktreePath, options) =>
    desktopApi(`/api/git/worktree/remove`, { force: options?.force ?? false, path: repoPath, worktreePath }),

  branchSwitch: (repoPath, branch) => desktopApi(`/api/git/branch/switch`, { branch, path: repoPath }),

  branchList: async repoPath =>
    (await desktopApi<{ branches: HermesGitBranch[] }>(`/api/git/branches?path=${q(repoPath)}`)).branches,

  repoStatus: repoPath => desktopApi<HermesRepoStatus | null>(`/api/git/status?path=${q(repoPath)}`),

  fileDiff: async (repoPath, filePath) =>
    (await desktopApi<{ diff: string }>(`/api/git/file-diff?path=${q(repoPath)}&file=${q(filePath)}`)).diff,

  review: {
    list: (repoPath, scope, baseRef) =>
      desktopApi<HermesReviewList>(
        `/api/git/review/list?path=${q(repoPath)}&scope=${q(scope)}${baseRef ? `&base=${q(baseRef)}` : ''}`
      ),

    diff: async (repoPath, filePath, scope, baseRef, staged) =>
      (
        await desktopApi<{ diff: string }>(
          `/api/git/review/diff?path=${q(repoPath)}&file=${q(filePath)}&scope=${q(scope)}&staged=${staged ? 'true' : 'false'}${baseRef ? `&base=${q(baseRef)}` : ''}`
        )
      ).diff,

    stage: (repoPath, filePath) => desktopApi(`/api/git/review/stage`, { file: filePath ?? null, path: repoPath }),

    unstage: (repoPath, filePath) => desktopApi(`/api/git/review/unstage`, { file: filePath ?? null, path: repoPath }),

    revert: (repoPath, filePath) => desktopApi(`/api/git/review/revert`, { file: filePath ?? null, path: repoPath }),

    revParse: async (repoPath, ref) =>
      (
        await desktopApi<{ sha: null | string }>(
          `/api/git/review/rev-parse?path=${q(repoPath)}${ref ? `&ref=${q(ref)}` : ''}`
        )
      ).sha,

    commit: (repoPath, message, push) => desktopApi(`/api/git/review/commit`, { message, path: repoPath, push }),

    commitContext: repoPath => desktopApi(`/api/git/review/commit-context?path=${q(repoPath)}`),

    push: repoPath => desktopApi(`/api/git/review/push`, { path: repoPath }),

    shipInfo: repoPath => desktopApi<HermesReviewShipInfo>(`/api/git/review/ship-info?path=${q(repoPath)}`),

    createPr: repoPath => desktopApi(`/api/git/review/create-pr`, { path: repoPath })
  },

  // Repo discovery is a local-disk crawl; on a remote gateway the backend
  // already merges session-derived repos, so this is a no-op.
  scanRepos: async () => []
}

export function desktopGit(): GitBridge | undefined {
  return isDesktopFsRemoteMode() ? remoteGit : window.hermesDesktop?.git
}
