import { atom, computed } from 'nanostores'

import type { HermesGitWorktree, HermesRepoStatus } from '@/global'
import { desktopGit } from '@/lib/desktop-git'

import { $worktreeRefreshToken } from './projects'
import { $busy, $currentCwd } from './session'
import { $workspaceChangeTick } from './workspace-events'

// Live working-tree status for the active session's cwd — the data backbone of
// the composer coding rail. It's the same "cheaply re-read git truth at the
// right moments" model as the sidebar worktree probe: a single bounded
// `git status --porcelain=v2` per refresh, driven by structural edges (cwd
// change, turn settle, window focus, worktree mutation), never per-token and
// never touching the conversation/system-prompt cache.

export const $repoStatus = atom<HermesRepoStatus | null>(null)
export const $repoStatusLoading = atom(false)

// The repo's real worktrees (for the coding rail's "jump to a worktree" menu).
// Refreshed on the same edges as the status probe; empty off a repo.
export const $repoWorktrees = atom<HermesGitWorktree[]>([])
const REPO_STATUS_REFRESH_DEBOUNCE_MS = 100

export type RepoChangeKind = 'added' | 'conflicted' | 'modified'

// Absolute file path → its git change kind, for VS Code-style file-tree tinting.
// Reuses the same bounded $repoStatus probe (capped file list); git reports
// repo-root-relative paths, so we join them onto the active cwd. Deletions never
// appear — the file is gone from disk, so there's no tree row to tint.
export const $repoChangeByPath = computed([$repoStatus, $currentCwd], (status, cwd) => {
  const map = new Map<string, RepoChangeKind>()
  const root = (cwd || '').replace(/[/\\]+$/, '')

  if (!status || !root) {
    return map
  }

  for (const file of status.files) {
    const kind: RepoChangeKind = file.conflicted ? 'conflicted' : file.untracked ? 'added' : 'modified'
    map.set(`${root}/${file.path}`, kind)
  }

  return map
})

async function loadWorktrees(target: string): Promise<void> {
  const list = desktopGit()?.worktreeList

  if (!list) {
    $repoWorktrees.set([])

    return
  }

  try {
    const worktrees = await list(target)

    if (inflightCwd === target) {
      $repoWorktrees.set(worktrees)
    }
  } catch {
    if (inflightCwd === target) {
      $repoWorktrees.set([])
    }
  }
}

// Coalesce overlapping probes: many triggers can fire around a turn boundary
// (busy flip + worktree token + focus), but only the latest cwd matters.
let inflightCwd: null | string = null
let repoStatusRefreshSeq = 0
let repoStatusRefreshTimer: ReturnType<typeof setTimeout> | null = null

const normalizeCwd = (cwd?: null | string): null | string => cwd?.trim() || null

/**
 * Re-probe the working tree for `cwd` (defaults to the active session's cwd).
 * Best-effort: a non-repo, a remote backend, or a missing probe clears the
 * status so the rail hides rather than showing stale data.
 */
export async function refreshRepoStatus(cwd?: null | string): Promise<void> {
  const target = normalizeCwd(cwd ?? $currentCwd.get())
  const probe = desktopGit()?.repoStatus
  const seq = (repoStatusRefreshSeq += 1)

  if (!target || !probe) {
    $repoStatus.set(null)
    $repoWorktrees.set([])

    return
  }

  inflightCwd = target
  $repoStatusLoading.set(true)

  try {
    const status = await probe(target)

    // Drop the result if the cwd moved on while we were probing (a fast session
    // switch) — the newer probe owns the atom.
    if (seq === repoStatusRefreshSeq && inflightCwd === target) {
      $repoStatus.set(status)

      // Worktrees only matter inside a repo; clear them otherwise.
      if (status) {
        void loadWorktrees(target)
      } else {
        $repoWorktrees.set([])
      }
    }
  } catch {
    if (seq === repoStatusRefreshSeq && inflightCwd === target) {
      $repoStatus.set(null)
      $repoWorktrees.set([])
    }
  } finally {
    if (seq === repoStatusRefreshSeq && inflightCwd === target) {
      $repoStatusLoading.set(false)
    }
  }
}

function scheduleRepoStatusRefresh(cwd?: null | string): void {
  if (repoStatusRefreshTimer) {
    clearTimeout(repoStatusRefreshTimer)
  }

  repoStatusRefreshTimer = setTimeout(() => {
    repoStatusRefreshTimer = null
    void refreshRepoStatus(cwd)
  }, REPO_STATUS_REFRESH_DEBOUNCE_MS)
}

// ── Triggers ─────────────────────────────────────────────────────────────────
// Wired once at module load (mirrors projects.ts's module-scope subscriptions).
// Each is a structural edge where the working tree may have changed under us.

// The active session's cwd changed (session switch / new chat) → re-probe.
$currentCwd.subscribe(cwd => scheduleRepoStatusRefresh(cwd))

// A worktree add/remove (desktop op, or the agent's out-of-band git in a settled
// turn / a window refocus — both already bump this token) → re-probe.
$worktreeRefreshToken.subscribe(() => scheduleRepoStatusRefresh())

// A file-mutating tool finished (event-driven, not polled) → re-probe so the
// rail's branch/+/- move exactly when the agent touches the tree.
$workspaceChangeTick.subscribe(() => scheduleRepoStatusRefresh())

// A turn settling is the backstop for changes no tool diff announced (e.g. a
// raw `git` in the terminal): one final refresh when the agent goes idle.
let prevBusy = $busy.get()

$busy.subscribe(busy => {
  if (prevBusy && !busy) {
    scheduleRepoStatusRefresh()
  }

  prevBusy = busy
})

// External changes while the window was away (an outside terminal) — refresh on
// refocus, the git-GUI standard.
if (typeof window !== 'undefined') {
  window.addEventListener('focus', () => scheduleRepoStatusRefresh())
}
