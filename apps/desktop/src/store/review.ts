import { atom, computed } from 'nanostores'

import { SIDEBAR_COLLAPSE_MEDIA_QUERY } from '@/app/layout-constants'
import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import type { HermesReviewFile, HermesReviewShipInfo } from '@/global'
import { matchesQuery } from '@/hooks/use-media-query'
import { desktopGit } from '@/lib/desktop-git'
import { isExcludedPath } from '@/lib/excluded-paths'
import { requestOneShot } from '@/lib/oneshot'
import { Codecs, persistentAtom } from '@/lib/persisted'

import { refreshRepoStatus } from './coding-status'
import { $busy, $currentCwd } from './session'
import { $workspaceChangeTick } from './workspace-events'

// State for the review pane: the working-tree changed-file list, the selected
// file's diff, and the git mutations (stage / unstage / revert). The active
// session's cwd is the repo; the pane reads git as the source of truth, the
// same bounded "re-probe on structural edges" model as the coding rail.
//
// Scope is always "uncommitted" — Hermes' flow is agent edits you review BEFORE
// committing, so branch/last-turn scopes are almost always empty here (unlike
// Codex, which commits per turn). We show the one view that's always populated.

// Must match the review <Pane id> in desktop-controller (the forced-reveal
// event is addressed by pane id).
export const REVIEW_PANE_ID = 'review'

const OPEN_KEY = 'hermes.desktop.reviewOpen'
const COMMIT_DEFAULT_KEY = 'hermes.desktop.reviewCommitDefault'
const TREE_MODE_KEY = 'hermes.desktop.reviewTreeMode'
const SELECTED_KEY = 'hermes.desktop.reviewSelectedPath'
const REVIEW_REFRESH_DEBOUNCE_MS = 100
const SHIP_INFO_STALE_MS = 30_000

// Persisted so the pane stays open across reloads (like the other rail panes).
export const $reviewOpen = persistentAtom(OPEN_KEY, false, Codecs.bool)

// The split-button's remembered default action ('commit' | 'commitPush').
export type CommitAction = 'commit' | 'commitPush'

export const $reviewCommitDefault = persistentAtom<CommitAction>(COMMIT_DEFAULT_KEY, 'commit', {
  decode: raw => (raw === 'commitPush' ? 'commitPush' : 'commit'),
  encode: value => value
})

// Changed-file layout: a flat path list (VS Code's default) or a folder tree.
export type ReviewTreeMode = 'list' | 'tree'

export const $reviewTreeMode = persistentAtom<ReviewTreeMode>(TREE_MODE_KEY, 'tree', {
  decode: raw => (raw === 'list' ? 'list' : 'tree'),
  encode: value => value
})

export function toggleReviewTreeMode(): void {
  $reviewTreeMode.set($reviewTreeMode.get() === 'tree' ? 'list' : 'tree')
}

export const $reviewFiles = atom<HermesReviewFile[]>([])
export const $reviewLoading = atom(false)
// False when the active session isn't in a local git repo (detached/fresh chat,
// remote backend). Lets the pane say "not a repo" instead of stranding on a
// skeleton or implying a clean repo with "no changes".
export const $reviewIsRepo = atom(true)

// Largest single-file churn (added + removed) in the current diff. Drives the
// per-row data bars: each file's bar is its churn relative to this max, so the
// biggest file fills the row and the rest scale down against it.
export const $reviewMaxChurn = computed($reviewFiles, files =>
  files.reduce((max, file) => Math.max(max, file.added + file.removed), 0)
)
// Persisted so a relaunch restores the file you were diffing (its diff is
// re-fetched in refreshReview once the file is confirmed still changed).
export const $reviewSelectedPath = persistentAtom<null | string>(SELECTED_KEY, null, Codecs.nullableText)
export const $reviewDiff = atom<null | string>(null)
export const $reviewDiffLoading = atom(false)

// Ship state: gh availability + this branch's PR, and a busy flag for the
// commit/push/PR action bar (disables buttons + shows progress).
export const $reviewShipInfo = atom<HermesReviewShipInfo>({ ghReady: false, pr: null })
export const $reviewShipBusy = atom(false)

// True while a commit message is being generated (drives the input's spinner).
export const $reviewCommitMsgBusy = atom(false)

const repoCwd = (): null | string => $currentCwd.get()?.trim() || null

type ReviewBridge = NonNullable<NonNullable<NonNullable<Window['hermesDesktop']>['git']>['review']>
let reviewRefreshSeq = 0
let reviewRefreshTimer: ReturnType<typeof setTimeout> | null = null
let shipInfoSeq = 0
let shipInfoLastCheckedAt = 0

// The two things every review op needs: the repo cwd + the IPC bridge. Null when
// either is missing (no session, remote backend), so callers bail in one line.
function reviewCtx(): { cwd: string; review: ReviewBridge } | null {
  const cwd = repoCwd()
  const review = desktopGit()?.review

  return cwd && review ? { cwd, review } : null
}

// ── Reads ────────────────────────────────────────────────────────────────────

export async function refreshReview(): Promise<void> {
  const ctx = reviewCtx()
  const seq = (reviewRefreshSeq += 1)

  if (!$reviewOpen.get() || !ctx) {
    $reviewFiles.set([])
    $reviewIsRepo.set(Boolean(ctx))

    // Critical: clear loading on the no-cwd / not-a-repo path too. It's set
    // true (optimistically) before a refresh is scheduled, so skipping it here
    // strands the pane on a forever-skeleton for a fresh, detached chat.
    if (seq === reviewRefreshSeq) {
      $reviewLoading.set(false)
    }

    return
  }

  const { cwd, review } = ctx

  $reviewIsRepo.set(true)
  $reviewLoading.set(true)

  try {
    const result = await review.list(cwd, 'uncommitted', null)

    // Ignore a result that resolved after the cwd moved on.
    if (seq !== reviewRefreshSeq || repoCwd() !== cwd) {
      return
    }

    // Hide dep/build/cache dirs and OS noise even when the repo tracks them —
    // .gitignored paths are already dropped upstream by `git status`.
    const files = result.files.filter(file => !isExcludedPath(file.path))

    $reviewFiles.set(files)

    // Drop the selection if the file is gone (staged away, reverted) so the diff
    // pane doesn't strand on a ghost; otherwise lazily fetch its diff so a
    // restored (persisted) selection re-renders on boot.
    const selected = $reviewSelectedPath.get()
    const selectedFile = selected ? files.find(file => file.path === selected) : null

    if (selected && !selectedFile) {
      clearReviewSelection()
    } else if (selectedFile && $reviewDiff.get() === null) {
      void selectReviewFile(selectedFile)
    }
  } catch {
    if (seq === reviewRefreshSeq) {
      $reviewFiles.set([])
    }
  } finally {
    if (seq === reviewRefreshSeq) {
      $reviewLoading.set(false)
    }
  }
}

function scheduleReviewRefresh(): void {
  if (!$reviewOpen.get()) {
    return
  }

  if (reviewRefreshTimer) {
    clearTimeout(reviewRefreshTimer)
  }

  reviewRefreshTimer = setTimeout(() => {
    reviewRefreshTimer = null
    void refreshReview()
  }, REVIEW_REFRESH_DEBOUNCE_MS)
}

export async function selectReviewFile(file: HermesReviewFile): Promise<void> {
  $reviewSelectedPath.set(file.path)

  const ctx = reviewCtx()

  if (!ctx) {
    $reviewDiff.set(null)

    return
  }

  $reviewDiffLoading.set(true)

  try {
    const diff = await ctx.review.diff(ctx.cwd, file.path, 'uncommitted', null, file.staged)

    if ($reviewSelectedPath.get() === file.path) {
      $reviewDiff.set(diff || '')
    }
  } catch {
    if ($reviewSelectedPath.get() === file.path) {
      $reviewDiff.set('')
    }
  } finally {
    if ($reviewSelectedPath.get() === file.path) {
      $reviewDiffLoading.set(false)
    }
  }
}

export function clearReviewSelection(): void {
  $reviewSelectedPath.set(null)
  $reviewDiff.set(null)
  $reviewDiffLoading.set(false)
}

// ── View state ───────────────────────────────────────────────────────────────

export async function refreshShipInfo(): Promise<void> {
  const ctx = reviewCtx()
  const seq = (shipInfoSeq += 1)

  if (!ctx) {
    $reviewShipInfo.set({ ghReady: false, pr: null })

    return
  }

  try {
    const info = await ctx.review.shipInfo(ctx.cwd)

    if (seq === shipInfoSeq && repoCwd() === ctx.cwd) {
      $reviewShipInfo.set(info)
      shipInfoLastCheckedAt = Date.now()
    }
  } catch {
    if (seq === shipInfoSeq) {
      $reviewShipInfo.set({ ghReady: false, pr: null })
      shipInfoLastCheckedAt = Date.now()
    }
  }
}

function refreshShipInfoIfStale(): void {
  if (Date.now() - shipInfoLastCheckedAt > SHIP_INFO_STALE_MS) {
    void refreshShipInfo()
  }
}

export function openReview(): void {
  $reviewOpen.set(true)
  void refreshReview()
  void refreshShipInfo()
}

export function closeReview(): void {
  $reviewOpen.set(false)
  clearReviewSelection()
}

export function toggleReview(): void {
  // Narrow width: the pane is a collapsed overlay (like the sidebar under ⌘B).
  // Make sure its data is loaded, then slide it in/out via the forced-reveal pin
  // — never the docked open state, which a 0px track would render invisibly.
  if (matchesQuery(SIDEBAR_COLLAPSE_MEDIA_QUERY)) {
    if (!$reviewOpen.get()) {
      openReview()
    }

    window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id: REVIEW_PANE_ID } }))

    return
  }

  if ($reviewOpen.get()) {
    closeReview()
  } else {
    openReview()
  }
}

// ── Mutations ────────────────────────────────────────────────────────────────

// Run a git mutation then re-sync both the review list and the rail's +/- (the
// working tree changed). A failure is swallowed by the caller's notify wrapper.
async function afterMutation(): Promise<void> {
  await refreshReview()
  void refreshRepoStatus()

  const selected = $reviewSelectedPath.get()
  const file = selected ? $reviewFiles.get().find(f => f.path === selected) : null

  // Re-fetch the open diff (staging flips which diff — cached vs worktree).
  if (file) {
    void selectReviewFile(file)
  }
}

export async function stageReviewFile(path: null | string): Promise<void> {
  await desktopGit()?.review?.stage(repoCwd() ?? '', path)
  await afterMutation()
}

export async function unstageReviewFile(path: null | string): Promise<void> {
  await desktopGit()?.review?.unstage(repoCwd() ?? '', path)
  await afterMutation()
}

export async function revertReviewFile(path: null | string): Promise<void> {
  await desktopGit()?.review?.revert(repoCwd() ?? '', path)
  await afterMutation()
}

// Revert is destructive (discards working-tree edits with no undo), so it always
// routes through a confirm dialog. The target is `{ path }` where `path === null`
// means "revert all"; `undefined` means no confirm is open. We wrap the path in
// an object so the `null` ("all") case is distinguishable from "closed".
export const $reviewRevertTarget = atom<{ path: null | string } | undefined>(undefined)

/** Open the revert confirm for a single file, or `null` for all changes. */
export function requestRevert(path: null | string): void {
  $reviewRevertTarget.set({ path })
}

export function cancelRevert(): void {
  $reviewRevertTarget.set(undefined)
}

/** Confirm the pending revert (closes the dialog, then performs it). */
export async function confirmRevert(): Promise<void> {
  const target = $reviewRevertTarget.get()

  $reviewRevertTarget.set(undefined)

  if (target) {
    await revertReviewFile(target.path)
  }
}

// ── Ship flow (commit / push / PR) ───────────────────────────────────────────

// Serialize ship actions behind one busy flag so the bar can't double-fire.
async function runShip<T>(action: () => Promise<T>): Promise<T> {
  $reviewShipBusy.set(true)

  try {
    return await action()
  } finally {
    $reviewShipBusy.set(false)
  }
}

export async function commitChanges(message: string, opts: { push?: boolean } = {}): Promise<void> {
  const ctx = reviewCtx()

  if (!ctx || !message.trim()) {
    return
  }

  await runShip(async () => {
    await ctx.review.commit(ctx.cwd, message.trim(), Boolean(opts.push))
    await refreshReview()
    void refreshRepoStatus()
    void refreshShipInfo()
  })
}

// Monotonic token: each generation captures one; Stop (or a newer press) bumps
// it, so a stale resolve is ignored. The model call can't be aborted
// server-side — we just drop its result and free the UI immediately.
let commitGenSeq = 0

/** Abandon any in-flight commit-message generation and re-enable the input. */
export function cancelCommitMessage(): void {
  commitGenSeq += 1
  $reviewCommitMsgBusy.set(false)
}

// Draft a commit message from the working-tree diff via a one-off LLM request
// (outside the conversation — no history, no cache break). `previous` is the
// current box text: handing it back as "don't repeat this" makes a re-press a
// real regen even on greedy / temperature-pinned models. Throws so the UI toasts.
export async function generateCommitMessage(previous = ''): Promise<string> {
  const ctx = reviewCtx()

  if (!ctx?.review.commitContext) {
    return ''
  }

  const gen = (commitGenSeq += 1)
  const live = () => gen === commitGenSeq

  $reviewCommitMsgBusy.set(true)

  try {
    const { diff, recent } = await ctx.review.commitContext(ctx.cwd)

    if (!live() || !diff.trim()) {
      return ''
    }

    const text = await requestOneShot({
      template: 'commit_message',
      temperature: 0.8,
      variables: { avoid: previous, diff, recent_commits: recent }
    })

    return live() ? text : ''
  } finally {
    if (live()) {
      $reviewCommitMsgBusy.set(false)
    }
  }
}

export async function pushChanges(): Promise<void> {
  const ctx = reviewCtx()

  if (!ctx) {
    return
  }

  await runShip(async () => {
    await ctx.review.push(ctx.cwd)
    void refreshShipInfo()
  })
}

// PR button: open the existing PR in the browser, or create one (pushing first)
// then open it. Caller gates this on shipInfo.ghReady.
export async function createOrOpenPr(): Promise<void> {
  const ctx = reviewCtx()

  if (!ctx) {
    return
  }

  const existing = $reviewShipInfo.get().pr

  if (existing?.url) {
    void window.hermesDesktop?.openExternal?.(existing.url)

    return
  }

  await runShip(async () => {
    const { url } = await ctx.review.createPr(ctx.cwd)

    if (url) {
      void window.hermesDesktop?.openExternal?.(url)
    }

    void refreshShipInfo()
  })
}

// ── Triggers (module-scope, mirror coding-status.ts) ─────────────────────────

// A file-mutating tool finished (event-driven, not polled) → refresh the open
// pane's changed-file list. gh/PR re-check is NOT here (gh is slow); it runs on
// the settle edge below.
$workspaceChangeTick.subscribe(() => {
  if ($reviewOpen.get()) {
    scheduleReviewRefresh()
  }
})

// Turn settled: final list refresh + the slower gh/PR re-check.
let prevBusy = $busy.get()

$busy.subscribe(busy => {
  if (prevBusy && !busy && $reviewOpen.get()) {
    scheduleReviewRefresh()
    refreshShipInfoIfStale()
  }

  prevBusy = busy
})

// The active session's cwd changed → the repo changed under the pane. Clear the
// stale file list + selection up front so the pane drops straight to its loading
// skeleton instead of blipping the previous repo's diff into the new one.
$currentCwd.subscribe(() => {
  if ($reviewOpen.get()) {
    clearReviewSelection()
    $reviewFiles.set([])
    $reviewLoading.set(true)
    scheduleReviewRefresh()
    void refreshShipInfo()
  }
})

// An outside terminal may have changed the tree while we were away.
if (typeof window !== 'undefined') {
  window.addEventListener('focus', () => {
    if ($reviewOpen.get()) {
      scheduleReviewRefresh()
      refreshShipInfoIfStale()
    }
  })
}
