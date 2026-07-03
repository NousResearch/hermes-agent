import { atom } from 'nanostores'

import type { HermesGitDivergence, HermesGitLogEntry, HermesGitShowEntry } from '@/global'
import { desktopGit } from '@/lib/desktop-git'
import { $currentCwd } from './session'
import { $reviewOpen, $reviewFiles } from './review'

// Commit-graph state for the review pane's graph section. The main process
// owns all git execution (electron/git-scm.cjs); this store holds the latest
// log snapshot + expanded-commit file lists, with race guards for rapid
// repo switching.

export const $graphLog = atom<HermesGitLogEntry[]>([])
export const $graphExpanded = atom<Set<string>>(new Set())
export const $graphCommitFiles = atom<Record<string, HermesGitShowEntry[]>>({})
export const $graphLoading = atom<Set<string>>(new Set())
export const $graphDivergence = atom<HermesGitDivergence | null>(null)

const GRAPH_MIN_HEIGHT = 80
const GRAPH_MAX_HEIGHT = 480
const GRAPH_DEFAULT_HEIGHT = 220

export const $graphHeight = atom(GRAPH_DEFAULT_HEIGHT)

export function setGraphHeight(h: number): void {
  $graphHeight.set(Math.max(GRAPH_MIN_HEIGHT, Math.min(GRAPH_MAX_HEIGHT, h)))
}

export function resetGraphHeight(): void {
  $graphHeight.set(GRAPH_DEFAULT_HEIGHT)
}

let seqLog = 0

function cwd(): string {
  return $currentCwd.get().trim()
}

export async function refreshGraph(): Promise<void> {
  const seq = ++seqLog
  const dir = cwd()
  if (!dir) { if (seq === seqLog) { $graphLog.set([]); $graphDivergence.set(null) } return }

  const git = desktopGit()
  if (!git) { if (seq === seqLog) { $graphLog.set([]); $graphDivergence.set(null) } return }

  try {
    const [entries, div] = await Promise.all([
      git.log(dir, 50),
      git.divergence(dir).catch(() => null)
    ])
    if (seq === seqLog) {
      $graphLog.set(entries)
      $graphDivergence.set(div)
    }
  } catch {
    if (seq === seqLog) { $graphLog.set([]); $graphDivergence.set(null) }
  }
}

const expandedRef = new Set<string>()

export async function toggleCommit(hash: string): Promise<void> {
  const dir = cwd()
  if (!dir) return

  if (expandedRef.has(hash)) {
    expandedRef.delete(hash)
    $graphExpanded.set(new Set(expandedRef))
    return
  }

  expandedRef.add(hash)
  $graphExpanded.set(new Set(expandedRef))

  if ($graphCommitFiles.get()[hash]) return

  $graphLoading.set(new Set($graphLoading.get()).add(hash))

  const git = desktopGit()
  if (!git) return

  try {
    const files = await git.show(dir, hash)
    if (expandedRef.has(hash)) {
      $graphCommitFiles.set({ ...$graphCommitFiles.get(), [hash]: files })
    }
  } catch {
    if (expandedRef.has(hash)) {
      $graphCommitFiles.set({ ...$graphCommitFiles.get(), [hash]: [] })
    }
  } finally {
    const next = new Set($graphLoading.get())
    next.delete(hash)
    $graphLoading.set(next)
  }
}

export function collapseGraph(): void {
  expandedRef.clear()
  $graphExpanded.set(new Set())
  $graphCommitFiles.set({})
  $graphLoading.set(new Set())
  $graphDivergence.set(null)
}

// Refresh graph when the review pane opens or the repo changes.
$reviewOpen.subscribe(open => {
  if (open) void refreshGraph()
})

$currentCwd.subscribe(() => {
  if ($reviewOpen.get()) {
    collapseGraph()
    void refreshGraph()
  }
})
