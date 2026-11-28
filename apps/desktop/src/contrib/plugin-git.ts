/**
 * `ctx.git` — the worktree door.
 *
 * The Desktop already owns git for its own "Start work" flow: `git worktree
 * list/add/remove` with a fixed argv and a hardened path check in the main
 * process (electron/git-worktree-ops.ts), reached through the preload bridge.
 * A plugin may not touch that bridge (it is internal and can change), and
 * without a door the only options were reimplementing git in Python or
 * reaching for the bridge anyway — which is exactly what the plugin catalog
 * review flagged on `project-command-center` (NousResearch/hermes-agent#152899).
 *
 * Scope is deliberately worktrees only. Branch switching, repository scans and
 * the review/stage surface stay app-owned: a plugin that needs one of those
 * should ask for that specific verb rather than receive a general git shell.
 *
 * Every member resolves a result and never throws, so an older shell (or a
 * browser without the bridge) reads as `{ ok: false, error: 'unavailable' }`.
 */

export interface PluginGitWorktree {
  path: string
  /** Short branch name, or null when the worktree is detached. */
  branch: null | string
  /** The primary checkout, where `.worktrees/` nests. */
  isMain: boolean
  detached: boolean
  locked: boolean
}

export interface PluginGitWorktreeAddOptions {
  /** Directory name under `<main>/.worktrees/`; defaults to the branch name. */
  name?: string
  /** New or existing branch to check out. */
  branch?: string
  /** Ref to start a new branch from (for example `origin/feature`). */
  base?: string
  /** A remote-tracking ref to check out as a new local branch. */
  existingBranch?: string
}

export type PluginGitResult<T extends object = object> =
  | ({ ok: true } & T)
  | { ok: false; error: string; message: string }

export interface PluginGit {
  /** Every worktree of the repository, the primary checkout first. */
  worktreeList: (repoPath: string) => Promise<PluginGitResult<{ worktrees: PluginGitWorktree[] }>>
  /** Make `<main>/.worktrees/<name>`. Never forces, never reuses a folder. */
  worktreeAdd: (
    repoPath: string,
    options?: PluginGitWorktreeAddOptions
  ) => Promise<PluginGitResult<{ branch: string; path: string; repoRoot: string }>>
  /** Remove a worktree git knows about. `force` also drops local changes. */
  worktreeRemove: (
    repoPath: string,
    worktreePath: string,
    options?: { force?: boolean }
  ) => Promise<PluginGitResult<{ removed: string }>>
}

// The refusal every door gives when the shell is older than the door. Plugins
// branch on `error`, and show `message` only if they have nowhere better.
const UNAVAILABLE = {
  error: 'unavailable',
  message: 'This Desktop build has no git door for plugins.',
  ok: false as const
}

function bridge(): NonNullable<NonNullable<typeof window.hermesDesktop>['git']> | null {
  if (typeof window === 'undefined') {
    return null
  }

  return window.hermesDesktop?.git ?? null
}

// One shape for every failure: the bridge throws plain Errors, and a refused
// path throws from the main process. Neither may escape into a plugin's
// render, so both become `{ ok: false }` with the sentence git gave.
function failure<T extends object>(err: unknown): PluginGitResult<T> {
  const message = err instanceof Error && err.message ? err.message : String(err || 'git failed')

  return { error: 'failed', message, ok: false }
}

export function createPluginGit(pluginId: string): PluginGit {
  // Named so a future audit can answer "which plugin ran git" from a log line
  // alone; the bridge call itself is unattributed.
  void pluginId

  return {
    async worktreeList(repoPath) {
      const git = bridge()

      if (!git) {
        return UNAVAILABLE
      }

      try {
        const worktrees = await git.worktreeList(repoPath)

        return { ok: true, worktrees: Array.isArray(worktrees) ? worktrees : [] }
      } catch (err) {
        return failure(err)
      }
    },
    async worktreeAdd(repoPath, options) {
      const git = bridge()

      if (!git) {
        return UNAVAILABLE
      }

      try {
        const added = await git.worktreeAdd(repoPath, options)

        return { branch: added.branch, ok: true, path: added.path, repoRoot: added.repoRoot }
      } catch (err) {
        return failure(err)
      }
    },
    async worktreeRemove(repoPath, worktreePath, options) {
      const git = bridge()

      if (!git) {
        return UNAVAILABLE
      }

      try {
        const removed = await git.worktreeRemove(repoPath, worktreePath, options)

        return { ok: true, removed: removed.removed }
      } catch (err) {
        return failure(err)
      }
    }
  }
}
