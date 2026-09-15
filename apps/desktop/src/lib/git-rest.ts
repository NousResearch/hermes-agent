import type {
  HermesGitBaseBranch,
  HermesGitBranch,
  HermesGitWorktree,
  HermesRepoPullRequests,
  HermesRepoStatus,
  HermesReviewList,
  HermesReviewShipInfo
} from '@/global'

export type GitBridge = NonNullable<NonNullable<Window['hermesDesktop']>['git']>

interface GitRestTransport {
  get<T>(route: string, params: Record<string, boolean | null | string | undefined>): Promise<T>
  post<T>(route: string, body: Record<string, unknown>): Promise<T>
}

/** Share the REST mapping while each host owns connection and profile routing. */
export function createGitRestBridge({ get, post }: GitRestTransport): GitBridge {
  return {
    worktreeList: async repoPath =>
      (await get<{ worktrees: HermesGitWorktree[] }>('worktrees', { path: repoPath })).worktrees,

    worktreeAdd: (repoPath, options) => post('worktree/add', { path: repoPath, ...options }),

    worktreeRemove: (repoPath, worktreePath, options) =>
      post('worktree/remove', { force: Boolean(options?.force), path: repoPath, worktreePath }),

    branchSwitch: (repoPath, branch) => post('branch/switch', { branch, path: repoPath }),

    branchList: async repoPath =>
      (await get<{ branches: HermesGitBranch[] }>('branches', { path: repoPath })).branches,

    baseBranchList: async repoPath =>
      (await get<{ branches: HermesGitBaseBranch[] }>('base-branches', { path: repoPath })).branches,

    repoStatus: repoPath => get<HermesRepoStatus | null>('status', { path: repoPath }),

    fileDiff: async (repoPath, filePath) =>
      (await get<{ diff: string }>('file-diff', { file: filePath, path: repoPath })).diff,

    review: {
      list: (repoPath, scope, baseRef) =>
        get<HermesReviewList>('review/list', { base: baseRef, path: repoPath, scope }),

      diff: async (repoPath, filePath, scope, baseRef, staged) =>
        (await get<{ diff: string }>('review/diff', { base: baseRef, file: filePath, path: repoPath, scope, staged }))
          .diff,

      stage: (repoPath, filePath) => post('review/stage', { file: filePath || null, path: repoPath }),

      unstage: (repoPath, filePath) => post('review/unstage', { file: filePath || null, path: repoPath }),

      revert: (repoPath, filePath) => post('review/revert', { file: filePath || null, path: repoPath }),

      revParse: async (repoPath, ref) =>
        (await get<{ sha: null | string }>('review/rev-parse', { path: repoPath, ref })).sha,

      commit: (repoPath, message, push) => post('review/commit', { message, path: repoPath, push }),

      commitContext: repoPath => get('review/commit-context', { path: repoPath }),

      push: repoPath => post('review/push', { path: repoPath }),

      shipInfo: repoPath => get<HermesReviewShipInfo>('review/ship-info', { path: repoPath }),

      prList: (repoPath, branches, numbers) =>
        post<HermesRepoPullRequests>('review/pr-list', { branches, numbers: numbers ?? [], path: repoPath }),

      // Remote gateways have no PR-comment route yet; resolve to null so the
      // paste degrades to a plain URL instead of throwing mid-paste.
      fetchPrComment: async () => null,

      createPr: repoPath => post('review/create-pr', { path: repoPath })
    },

    // Repo discovery is a local-disk crawl; on a remote gateway the backend
    // already merges session-derived repos, so this is a no-op.
    scanRepos: async () => []
  }
}
