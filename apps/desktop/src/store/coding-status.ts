import { atom, computed } from 'nanostores'

import type { HermesGitWorktree, HermesRepoStatus } from '@/global'
import { desktopFsCacheKey } from '@/lib/desktop-fs'
import { desktopGit } from '@/lib/desktop-git'

import { $worktreeRefreshToken } from './projects'
import { $busy, $connection, $currentCwd, $selectedStoredSessionId } from './session'
import { $workspaceChangeTick } from './workspace-events'

// Live working-tree status for the active session's cwd — the data backbone of
// the composer coding rail. It's the same "cheaply re-read git truth at the
// right moments" model as the sidebar worktree probe: a single bounded
// `git status --porcelain=v2` per refresh, driven by structural edges (cwd
// change, turn settle, window focus, worktree mutation), never per-token and
// never touching the conversation/system-prompt cache.

interface RepoStatusContext {
  backendKey: string
  cwd: string
  storedSessionId: null | string
}

let activeRepoStatusContext: null | RepoStatusContext = null
let repoStatusContextEpoch = 0

function normalizeCwd(cwd?: null | string): null | string {
  const trimmed = cwd?.trim()

  if (!trimmed) {
    return null
  }

  return trimmed.replace(/[/\\]+$/, '') || trimmed
}

function getRepoStatusContext(cwd?: null | string): null | RepoStatusContext {
  const target = normalizeCwd(cwd)

  if (!target) {
    return null
  }

  return {
    backendKey: desktopFsCacheKey(),
    cwd: target,
    storedSessionId: $selectedStoredSessionId.get()
  }
}

function contextsMatch(left: null | RepoStatusContext, right: null | RepoStatusContext): boolean {
  return (
    left?.backendKey === right?.backendKey &&
    left?.cwd === right?.cwd &&
    left?.storedSessionId === right?.storedSessionId
  )
}

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
  const root = normalizeCwd(cwd)

  if (!status || !root || activeRepoStatusContext?.cwd !== root) {
    return map
  }

  for (const file of status.files) {
    const kind: RepoChangeKind = file.conflicted ? 'conflicted' : file.untracked ? 'added' : 'modified'
    map.set(`${root}/${file.path}`, kind)
  }

  return map
})

async function loadWorktrees(request: RepoStatusRefreshRequest): Promise<void> {
  const list = desktopGit()?.worktreeList

  if (!list) {
    if (isCurrentRepoStatusRequest(request)) {
      $repoWorktrees.set([])
    }

    return
  }

  try {
    const worktrees = await list(request.context.cwd)

    if (isCurrentRepoStatusRequest(request)) {
      $repoWorktrees.set(worktrees)
    }
  } catch {
    if (isCurrentRepoStatusRequest(request)) {
      $repoWorktrees.set([])
    }
  }
}

interface RepoStatusRefreshRequest {
  context: RepoStatusContext
  epoch: number
  probe: (cwd: string) => Promise<HermesRepoStatus | null>
  seq: number
}

// Coalesce overlapping probes: many triggers can fire around a turn boundary
// (busy flip + worktree token + focus), but only the latest active context
// matters. Keep one probe in flight and retain at most one trailing request so
// a slow Git status cannot multiply into an unbounded subprocess pile-up.
let pendingRepoStatusRefresh: RepoStatusRefreshRequest | null = null
let repoStatusRefreshInFlight: Promise<void> | null = null
let repoStatusRefreshSeq = 0
let repoStatusRefreshTimer: ReturnType<typeof setTimeout> | undefined

function isActiveRepoStatusContext(context: RepoStatusContext, epoch: number): boolean {
  return epoch === repoStatusContextEpoch && contextsMatch(activeRepoStatusContext, context)
}

function isCurrentRepoStatusRequest(request: RepoStatusRefreshRequest): boolean {
  return request.seq === repoStatusRefreshSeq && isActiveRepoStatusContext(request.context, request.epoch)
}

function activateRepoStatusContext(context: null | RepoStatusContext): number {
  if (contextsMatch(activeRepoStatusContext, context)) {
    return repoStatusContextEpoch
  }

  activeRepoStatusContext = context
  repoStatusContextEpoch += 1
  pendingRepoStatusRefresh = null
  $repoStatus.set(null)
  $repoWorktrees.set([])

  return repoStatusContextEpoch
}

async function runRepoStatusRefresh(request: RepoStatusRefreshRequest): Promise<void> {
  try {
    const status = await request.probe(request.context.cwd)

    // A stale response may finish after the session, cwd, or filesystem backend
    // changed. Only the latest request for the currently active context owns
    // the shared coding rail atoms.
    if (isCurrentRepoStatusRequest(request)) {
      $repoStatus.set(status)

      // Worktrees only matter inside a repo; clear them otherwise.
      if (status) {
        void loadWorktrees(request)
      } else {
        $repoWorktrees.set([])
      }
    }
  } catch {
    if (isCurrentRepoStatusRequest(request)) {
      $repoStatus.set(null)
      $repoWorktrees.set([])
    }
  }
}

async function drainRepoStatusRefreshes(): Promise<void> {
  while (pendingRepoStatusRefresh) {
    const request = pendingRepoStatusRefresh

    pendingRepoStatusRefresh = null
    await runRepoStatusRefresh(request)
  }

  // This reset is synchronous with the final empty-queue check. A refresh
  // arriving before this continuation runs is drained above; one arriving
  // afterward sees no in-flight promise and starts a new drain.
  repoStatusRefreshInFlight = null
  $repoStatusLoading.set(false)
}

function enqueueRepoStatusRefresh(context: RepoStatusContext, epoch: number): Promise<void> {
  if (!isActiveRepoStatusContext(context, epoch)) {
    return Promise.resolve()
  }

  const probe = desktopGit()?.repoStatus
  const seq = (repoStatusRefreshSeq += 1)

  if (!probe) {
    pendingRepoStatusRefresh = null
    $repoStatus.set(null)
    $repoWorktrees.set([])

    if (!repoStatusRefreshInFlight) {
      $repoStatusLoading.set(false)
    }

    return repoStatusRefreshInFlight || Promise.resolve()
  }

  pendingRepoStatusRefresh = { context, epoch, probe, seq }
  $repoStatusLoading.set(true)

  if (!repoStatusRefreshInFlight) {
    repoStatusRefreshInFlight = drainRepoStatusRefreshes()
  }

  return repoStatusRefreshInFlight
}

/**
 * Re-probe the working tree for `cwd` (defaults to the active session's cwd).
 * Best-effort: a non-repo, a remote backend, or a missing probe clears the
 * status so the rail hides rather than showing stale data.
 */
export function refreshRepoStatus(cwd?: null | string): Promise<void> {
  const context = getRepoStatusContext(cwd ?? $currentCwd.get())
  const epoch = activateRepoStatusContext(context)

  if (!context) {
    pendingRepoStatusRefresh = null
    $repoStatus.set(null)
    $repoWorktrees.set([])

    if (!repoStatusRefreshInFlight) {
      $repoStatusLoading.set(false)
    }

    return repoStatusRefreshInFlight || Promise.resolve()
  }

  return enqueueRepoStatusRefresh(context, epoch)
}

function scheduleRepoStatusRefresh(cwd?: null | string): void {
  const context = getRepoStatusContext(cwd ?? $currentCwd.get())

  // Context moves are a foreground ownership change, not a cosmetic refresh:
  // clear synchronously so the Composer never paints the old backend/session's
  // status during the debounce before Git can re-probe.
  const epoch = activateRepoStatusContext(context)

  clearTimeout(repoStatusRefreshTimer)

  if (!context) {
    repoStatusRefreshTimer = undefined

    return
  }

  repoStatusRefreshTimer = setTimeout(() => {
    repoStatusRefreshTimer = undefined

    // The timer represents the context that scheduled it. Do not turn an old
    // cwd back into a current request after a newer context already took over.
    void enqueueRepoStatusRefresh(context, epoch)
  }, REPO_STATUS_REFRESH_DEBOUNCE_MS)
}

// ── Triggers ─────────────────────────────────────────────────────────────────
// Wired once at module load (mirrors projects.ts's module-scope subscriptions).
// Each is a structural edge where the working tree may have changed under us.

// The active session's cwd changed (session switch / new chat) → re-probe.
$currentCwd.subscribe(cwd => scheduleRepoStatusRefresh(cwd))

// The same path can be served by another profile/backend. Match the filesystem
// cache identity so a remote profile's status cannot leak through the shared rail.
$connection.subscribe(() => scheduleRepoStatusRefresh())

// Switching sessions can land on the same cwd but a different checked-out
// branch (the agent ran `git checkout` in another session's terminal). The cwd
// subscription above won't fire when the path is identical, so the branch label
// would stay stale until a window focus or turn-settle triggers a refresh.
// Treat the stored-session id as a structural edge in its own right.
$selectedStoredSessionId.subscribe(() => scheduleRepoStatusRefresh())

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
