import { atom } from 'nanostores'

import { $currentCwd } from './session'

// Source-control state for the desktop Source Control panel. The main process
// owns all git execution (electron/git-scm.cjs); this store holds the latest
// status snapshot, a busy/error flag, and a small action layer that refreshes
// after every mutation so the panel stays in sync with the working tree.

export type GitFileStatus =
  | 'added'
  | 'conflicted'
  | 'copied'
  | 'deleted'
  | 'modified'
  | 'renamed'
  | 'typechange'
  | 'untracked'

export interface GitFile {
  path: string
  origPath?: string
  status: GitFileStatus
  staged: boolean
}

export interface GitStatusSnapshot {
  root: string | null
  branch: string | null
  upstream: string | null
  ahead: number
  behind: number
  staged: GitFile[]
  unstaged: GitFile[]
  untracked: GitFile[]
  conflicted: GitFile[]
}

const EMPTY: GitStatusSnapshot = {
  root: null,
  branch: null,
  upstream: null,
  ahead: 0,
  behind: 0,
  staged: [],
  unstaged: [],
  untracked: [],
  conflicted: []
}

export const $gitStatus = atom<GitStatusSnapshot>(EMPTY)
export const $gitAvailable = atom(false)
export const $gitBusy = atom(false)
export const $gitError = atom<string | null>(null)
// True once a status fetch has completed at least once (so the panel can tell
// "loading" from "clean repo").
export const $gitLoaded = atom(false)

function gitApi() {
  return typeof window !== 'undefined' ? window.hermesDesktop?.git : undefined
}

function currentCwd(): string {
  return $currentCwd.get().trim()
}

/** Fetch the current git status for the active cwd and update the store. */
export async function refreshGitStatus(): Promise<void> {
  const api = gitApi()
  const cwd = currentCwd()

  if (!api || !cwd) {
    $gitAvailable.set(false)
    $gitStatus.set(EMPTY)
    $gitLoaded.set(true)

    return
  }

  $gitBusy.set(true)

  try {
    const result = await api.status(cwd)

    if (!result.ok) {
      // not-a-repo is the common, non-error case (cwd isn't a checkout).
      $gitAvailable.set(false)
      $gitStatus.set(EMPTY)
      $gitError.set(result.error === 'not-a-repo' ? null : (result.error ?? 'git status failed'))

      return
    }

    $gitAvailable.set(true)
    $gitError.set(null)
    $gitStatus.set({
      root: result.root ?? null,
      branch: result.branch ?? null,
      upstream: result.upstream ?? null,
      ahead: result.ahead ?? 0,
      behind: result.behind ?? 0,
      staged: (result.staged ?? []) as GitFile[],
      unstaged: (result.unstaged ?? []) as GitFile[],
      untracked: (result.untracked ?? []) as GitFile[],
      conflicted: (result.conflicted ?? []) as GitFile[]
    })
  } catch (error) {
    $gitError.set(error instanceof Error ? error.message : String(error))
  } finally {
    $gitBusy.set(false)
    $gitLoaded.set(true)
  }
}

// Run a mutation, then refresh. Surfaces the op's error without throwing so the
// panel can show it inline. The post-mutation refresh must NOT clobber a
// mutation error (a failed push still has a clean working tree), so we re-assert
// the error after refreshing.
async function mutate(run: (cwd: string) => Promise<{ ok: boolean; error?: string }>): Promise<boolean> {
  const cwd = currentCwd()

  if (!cwd) {
    return false
  }

  $gitBusy.set(true)
  let opError: string | null = null

  try {
    const result = await run(cwd)

    if (!result.ok) {
      opError = result.error ?? 'git operation failed'

      return false
    }

    return true
  } catch (error) {
    opError = error instanceof Error ? error.message : String(error)

    return false
  } finally {
    $gitBusy.set(false)
    await refreshGitStatus()
    // refreshGitStatus clears $gitError on success; restore the mutation error
    // so the panel can still explain why the operation failed.
    $gitError.set(opError)
  }
}

export function stageFiles(paths: string[]) {
  return mutate(cwd => gitApi()!.stage(cwd, paths))
}

export function unstageFiles(paths: string[]) {
  return mutate(cwd => gitApi()!.unstage(cwd, paths))
}

export function discardFiles(paths: string[]) {
  return mutate(cwd => gitApi()!.discard(cwd, paths))
}

export function stageAll() {
  const { unstaged, untracked, conflicted } = $gitStatus.get()
  const paths = [...unstaged, ...untracked, ...conflicted].map(f => f.path)

  return paths.length ? stageFiles(paths) : Promise.resolve(false)
}

export function unstageAll() {
  const paths = $gitStatus.get().staged.map(f => f.path)

  return paths.length ? unstageFiles(paths) : Promise.resolve(false)
}

export function commit(message: string, options?: { amend?: boolean; all?: boolean }) {
  return mutate(cwd => gitApi()!.commit(cwd, message, options))
}

export function push(options?: { setUpstream?: boolean }) {
  return mutate(cwd => gitApi()!.push(cwd, options))
}

export function pull() {
  return mutate(cwd => gitApi()!.pull(cwd))
}

export async function fetchDiff(filePath: string, staged: boolean): Promise<string> {
  const api = gitApi()
  const cwd = currentCwd()

  if (!api || !cwd) {
    return ''
  }

  try {
    const result = await api.diff(cwd, filePath, staged)

    return result.ok ? (result.diff ?? '') : ''
  } catch {
    return ''
  }
}

// ─── Branches ────────────────────────────────────────────────────────────────

export interface GitBranch {
  name: string
  current: boolean
  upstream: string | null
  ahead: number
  behind: number
  sha: string
  subject: string
}

export const $gitBranches = atom<{ local: GitBranch[]; remote: GitBranch[] }>({ local: [], remote: [] })

export async function refreshBranches(): Promise<void> {
  const api = gitApi()
  const cwd = currentCwd()

  if (!api?.branches || !cwd) {
    $gitBranches.set({ local: [], remote: [] })

    return
  }

  try {
    const result = await api.branches(cwd)

    if (result.ok) {
      $gitBranches.set({ local: (result.local ?? []) as GitBranch[], remote: (result.remote ?? []) as GitBranch[] })
    }
  } catch {
    /* leave previous */
  }
}

export function checkoutBranch(branch: string) {
  return mutate(cwd => gitApi()!.checkout(cwd, branch)).then(async ok => {
    await refreshBranches()

    return ok
  })
}

export function createBranch(name: string, startPoint?: string) {
  return mutate(cwd => gitApi()!.createBranch(cwd, name, startPoint ? { startPoint } : undefined)).then(async ok => {
    await refreshBranches()

    return ok
  })
}

export function deleteBranch(name: string, force = false) {
  return mutate(cwd => gitApi()!.deleteBranch(cwd, name, { force })).then(async ok => {
    await refreshBranches()

    return ok
  })
}

export function fetchRemote() {
  return mutate(cwd => gitApi()!.fetch(cwd)).then(async ok => {
    await refreshBranches()

    return ok
  })
}

// ─── Log / history ───────────────────────────────────────────────────────────

export interface GitCommit {
  sha: string
  shortSha: string
  author: string
  authorEmail: string
  date: string
  relativeDate: string
  subject: string
}

export const $gitLog = atom<GitCommit[]>([])

export async function refreshLog(limit = 50): Promise<void> {
  const api = gitApi()
  const cwd = currentCwd()

  if (!api?.log || !cwd) {
    $gitLog.set([])

    return
  }

  try {
    const result = await api.log(cwd, { limit })

    if (result.ok) {
      $gitLog.set((result.commits ?? []) as GitCommit[])
    }
  } catch {
    /* leave previous */
  }
}

export async function fetchCommitDiff(sha: string): Promise<string> {
  const api = gitApi()
  const cwd = currentCwd()

  if (!api?.commitDiff || !cwd) {
    return ''
  }

  try {
    const result = await api.commitDiff(cwd, sha)

    return result.ok ? (result.diff ?? '') : ''
  } catch {
    return ''
  }
}

// ─── Stash ───────────────────────────────────────────────────────────────────

export interface GitStash {
  ref: string
  subject: string
}

export const $gitStashes = atom<GitStash[]>([])

export async function refreshStashes(): Promise<void> {
  const api = gitApi()
  const cwd = currentCwd()

  if (!api?.stashList || !cwd) {
    $gitStashes.set([])

    return
  }

  try {
    const result = await api.stashList(cwd)

    if (result.ok) {
      $gitStashes.set((result.stashes ?? []) as GitStash[])
    }
  } catch {
    /* leave previous */
  }
}

export function stashPush(options?: { includeUntracked?: boolean; message?: string }) {
  return mutate(cwd => gitApi()!.stashPush(cwd, options)).then(async ok => {
    await refreshStashes()

    return ok
  })
}

export function stashAction(action: 'apply' | 'drop' | 'pop', ref: string) {
  return mutate(cwd => gitApi()!.stashAction(cwd, action, ref)).then(async ok => {
    await refreshStashes()

    return ok
  })
}

// ─── Hunk-level staging ──────────────────────────────────────────────────────

// Stage (reverse=false) or unstage (reverse=true) a single hunk via an
// apply-able patch fragment built by lib/git-diff.buildHunkPatch.
export function applyHunk(patch: string, reverse: boolean) {
  return mutate(cwd => gitApi()!.applyHunk(cwd, patch, { reverse }))
}

// Refresh everything the panel shows (status + branches + log + stashes).
export async function refreshAll(): Promise<void> {
  await Promise.all([refreshGitStatus(), refreshBranches(), refreshLog(), refreshStashes()])
}

// Refresh whenever the active workspace cwd changes.
let lastCwd = ''
$currentCwd.subscribe(cwd => {
  const next = (cwd ?? '').trim()

  if (next !== lastCwd) {
    lastCwd = next
    void refreshGitStatus()
    void refreshBranches()
  }
})
