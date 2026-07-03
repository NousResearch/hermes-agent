import { atom } from 'nanostores'

import type {
  HermesGitLogEntry,
  HermesGitShowEntry,
  HermesRepoStatus,
  HermesReviewFile,
  HermesReviewList
} from '@/global'
import { desktopGit } from '@/lib/desktop-git'
import { $currentCwd } from './session'

// Source-control state for the desktop Source Control panel. The main process
// owns all git execution (electron/git-scm.cjs); this store holds the latest
// status/files/log snapshots, busy/error flags, and a refresh layer that
// guards against stale writes on rapid repo switching.

export const $scmStatus = atom<HermesRepoStatus | null>(null)
export const $scmFiles = atom<HermesReviewFile[]>([])
export const $scmLog = atom<HermesGitLogEntry[]>([])
export const $scmBusy = atom(false)
export const $scmError = atom<string | null>(null)
export const $scmLoaded = atom(false)
export const $scmIsRepo = atom(true)

// Sidebar mode: 'explorer' | 'sourceControl'. Toggled by Ctrl+Shift+G.
export const $scmVisible = atom(false)

export function showSourceControl(): void {
  $scmVisible.set(true)
  refreshAll()
}

// Commit graph expansion state.
export const $scmExpandedCommits = atom<Set<string>>(new Set())
export const $scmCommitFiles = atom<Record<string, HermesGitShowEntry[]>>({})
export const $scmCommitFilesLoading = atom<Set<string>>(new Set())

function api() {
  return desktopGit()
}

function cwd(): string {
  return $currentCwd.get().trim()
}

// Race guards: each refresh increments a counter; only the latest wins.
let seqStatus = 0
let seqFiles = 0
let seqLog = 0

export async function refreshStatus(): Promise<void> {
  const seq = ++seqStatus
  const dir = cwd()
  if (!dir) { if (seq === seqStatus) { $scmStatus.set(null); $scmIsRepo.set(false) } return }

  const git = api()
  if (!git) { if (seq === seqStatus) { $scmStatus.set(null); $scmIsRepo.set(false) } return }

  $scmBusy.set(true)
  try {
    const s = await git.repoStatusGraph(dir)
    if (seq === seqStatus) { $scmStatus.set(s); $scmIsRepo.set(true) }
  } catch {
    if (seq === seqStatus) { $scmStatus.set(null); $scmIsRepo.set(false) }
  } finally {
    if (seq === seqStatus) $scmBusy.set(false)
    $scmLoaded.set(true)
  }
}

export async function refreshFiles(): Promise<void> {
  const seq = ++seqFiles
  const dir = cwd()
  if (!dir) { if (seq === seqFiles) { $scmFiles.set([]); $scmIsRepo.set(false) } return }

  const git = api()
  if (!git) { if (seq === seqFiles) { $scmFiles.set([]); $scmIsRepo.set(false) } return }

  try {
    const result: HermesReviewList = await git.changedFiles(dir)
    if (seq === seqFiles) { $scmFiles.set(result.files); $scmIsRepo.set(true) }
  } catch {
    if (seq === seqFiles) { $scmFiles.set([]); $scmIsRepo.set(false) }
  }
}

export async function refreshLog(): Promise<void> {
  const seq = ++seqLog
  const dir = cwd()
  if (!dir) { if (seq === seqLog) $scmLog.set([]); return }

  const git = api()
  if (!git) { if (seq === seqLog) $scmLog.set([]); return }

  try {
    const entries = await git.log(dir, 50)
    if (seq === seqLog) $scmLog.set(entries)
  } catch {
    if (seq === seqLog) $scmLog.set([])
  }
}

export function refreshAll(): void {
  void refreshStatus()
  void refreshFiles()
  void refreshLog()
}

// Toggle commit expansion in the graph. If expanding and not cached, fetches
// the commit's file list. Guards against races via expandedCommitsRef.
const expandedRef = new Set<string>()

export async function toggleCommit(hash: string): Promise<void> {
  const dir = cwd()
  if (!dir) return

  const isOpen = expandedRef.has(hash)
  if (isOpen) {
    expandedRef.delete(hash)
    $scmExpandedCommits.set(new Set(expandedRef))
    return
  }

  expandedRef.add(hash)
  $scmExpandedCommits.set(new Set(expandedRef))

  // Already cached — no fetch needed.
  if ($scmCommitFiles.get()[hash]) return

  $scmCommitFilesLoading.set(new Set($scmCommitFilesLoading.get()).add(hash))

  const git = api()
  if (!git) return

  try {
    const files = await git.show(dir, hash)
    if (expandedRef.has(hash)) {
      $scmCommitFiles.set({ ...$scmCommitFiles.get(), [hash]: files })
    }
  } catch {
    if (expandedRef.has(hash)) {
      $scmCommitFiles.set({ ...$scmCommitFiles.get(), [hash]: [] })
    }
  } finally {
    const next = new Set($scmCommitFilesLoading.get())
    next.delete(hash)
    $scmCommitFilesLoading.set(next)
  }
}

export function collapseAllCommits(): void {
  expandedRef.clear()
  $scmExpandedCommits.set(new Set())
  $scmCommitFiles.set({})
  $scmCommitFilesLoading.set(new Set())
}

// Clear graph state when the repo changes so stale files don't show.
export function clearGraphState(): void {
  collapseAllCommits()
}

// Auto-refresh whenever the active workspace cwd changes.
let lastCwd = ''
$currentCwd.subscribe(next => {
  const trimmed = (next ?? '').trim()
  if (trimmed !== lastCwd) {
    lastCwd = trimmed
    clearGraphState()
    refreshAll()
  }
})

export * as scm from './git'
