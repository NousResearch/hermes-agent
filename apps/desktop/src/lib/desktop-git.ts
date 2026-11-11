import { hermesApiAs } from '@/api/client'
import type {
  HermesGitBaseBranch,
  HermesGitBranch,
  HermesGitWorktree,
  HermesRepoPullRequests,
  HermesRepoStatus,
  HermesReviewList,
  HermesReviewShipInfo
} from '@/global'
import { hermesApi } from '@/hermes'

import { desktopFsProfile, isDesktopFsRemoteMode } from './desktop-fs'

// Remote-aware git facade. Locally the desktop runs git through Electron
// (window.hermesDesktop.git); on a remote gateway that's the wrong filesystem,
// so we mirror the same surface over the dashboard REST API (/api/git/*) — the
// coding rail, worktree lanes, review pane, and branch ops then act on the
// BACKEND repo where sessions actually run. Mirrors desktop-fs.ts.

type GitBridge = NonNullable<NonNullable<Window['hermesDesktop']>['git']>

type GitApi = <T>(path: string, body?: Record<string, unknown>) => Promise<T>

/** The gateway (connection, profile) a repo lives on when it is not the window's. */
export interface GitOwnerRoute {
  connectionId: string
  profile: string
}

const windowApi: GitApi = <T>(path: string, body?: Record<string, unknown>) => {
  if (!window.hermesDesktop) {
    throw new Error('Unitalk bridge is unavailable')
  }

  return hermesApi<T>(
    body ? { body, method: 'POST', path, profile: desktopFsProfile() } : { path, profile: desktopFsProfile() }
  )
}

const ownerApi =
  (owner: GitOwnerRoute): GitApi =>
  <T>(path: string, body?: Record<string, unknown>) =>
    hermesApiAs<T>(owner, body ? { body, method: 'POST', path } : { path })

function remoteGitVia(api: GitApi): GitBridge {
  const gitGet = <T>(route: string, params: Record<string, boolean | null | string | undefined>): Promise<T> => {
    const query = new URLSearchParams()

    for (const [key, value] of Object.entries(params)) {
      if (value !== null && value !== undefined) {
        query.set(key, String(value))
      }
    }

    return api<T>(`/api/git/${route}?${query.toString()}`)
  }

  const gitPost = <T>(route: string, body: Record<string, unknown>): Promise<T> => api<T>(`/api/git/${route}`, body)

  return {
    worktreeList: async repoPath =>
      (await gitGet<{ worktrees: HermesGitWorktree[] }>('worktrees', { path: repoPath })).worktrees,

    worktreeAdd: (repoPath, options) => gitPost('worktree/add', { path: repoPath, ...options }),

    worktreeRemove: (repoPath, worktreePath, options) =>
      gitPost('worktree/remove', { force: options?.force ?? false, path: repoPath, worktreePath }),

    branchSwitch: (repoPath, branch) => gitPost('branch/switch', { branch, path: repoPath }),

    branchList: async repoPath =>
      (await gitGet<{ branches: HermesGitBranch[] }>('branches', { path: repoPath })).branches,

    baseBranchList: async repoPath =>
      (await gitGet<{ branches: HermesGitBaseBranch[] }>('base-branches', { path: repoPath })).branches,

    repoStatus: repoPath => gitGet<HermesRepoStatus | null>('status', { path: repoPath }),

    fileDiff: async (repoPath, filePath) =>
      (await gitGet<{ diff: string }>('file-diff', { file: filePath, path: repoPath })).diff,

    review: {
      list: (repoPath, scope, baseRef) =>
        gitGet<HermesReviewList>('review/list', { base: baseRef, path: repoPath, scope }),

      diff: async (repoPath, filePath, scope, baseRef, staged) =>
        (
          await gitGet<{ diff: string }>('review/diff', {
            base: baseRef,
            file: filePath,
            path: repoPath,
            scope,
            staged
          })
        ).diff,

      stage: (repoPath, filePath) => gitPost('review/stage', { file: filePath || null, path: repoPath }),

      unstage: (repoPath, filePath) => gitPost('review/unstage', { file: filePath || null, path: repoPath }),

      revert: (repoPath, filePath) => gitPost('review/revert', { file: filePath || null, path: repoPath }),

      revParse: async (repoPath, ref) =>
        (await gitGet<{ sha: null | string }>('review/rev-parse', { path: repoPath, ref })).sha,

      commit: (repoPath, message, push) => gitPost('review/commit', { message, path: repoPath, push }),

      commitContext: repoPath => gitGet('review/commit-context', { path: repoPath }),

      push: repoPath => gitPost('review/push', { path: repoPath }),

      shipInfo: repoPath => gitGet<HermesReviewShipInfo>('review/ship-info', { path: repoPath }),

      prList: (repoPath, branches, numbers) =>
        gitPost<HermesRepoPullRequests>('review/pr-list', { branches, numbers: numbers ?? [], path: repoPath }),

      createPr: repoPath => gitPost('review/create-pr', { path: repoPath })
    },

    // Repo discovery is a local-disk crawl; on a remote gateway the backend
    // already merges session-derived repos, so this is a no-op.
    scanRepos: async () => []
  }
}

const remoteGit = remoteGitVia(windowApi)

export function desktopGit(): GitBridge | undefined {
  if (typeof window === 'undefined') {
    return undefined
  }

  return isDesktopFsRemoteMode() ? remoteGit : window.hermesDesktop?.git
}

/** Git for a repo another gateway owns: that gateway's /api/git, so worktrees and branch switches
 *  happen on the machine the repo is on. */
export function desktopGitFor(owner: GitOwnerRoute): GitBridge | undefined {
  return typeof window === 'undefined' ? undefined : remoteGitVia(ownerApi(owner))
}
