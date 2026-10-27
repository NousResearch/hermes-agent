import { atom, computed } from 'nanostores'

import { SIDEBAR_COLLAPSE_MEDIA_QUERY } from '@/app/layout-constants'
import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { isPaneVisible, revealTreePane } from '@/components/pane-shell/tree/store'
import type { HermesReviewFile, HermesReviewScope, HermesReviewShipInfo } from '@/global'
import { matchesQuery } from '@/hooks/use-media-query'
import { desktopGit } from '@/lib/desktop-git'
import { isExcludedPath } from '@/lib/excluded-paths'
import { requestOneShot } from '@/lib/oneshot'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { modeBound } from '@/store/interface-mode'

import { refreshRepoStatus, repoStatusForCwd } from './coding-status'
import { noteAreaClosed, recordFeatureUse } from './desktop-metrics'
import { stampSessionPrBranch } from './pull-requests'
import { $busy, $currentCwd, $selectedStoredSessionId, $sessions } from './session'
import { $sessionStates } from './session-states'
import { $workspaceChangeTick } from './workspace-events'

// State for the review pane: the working-tree changed-file list, the selected
// file's diff, and the git mutations (stage / unstage / revert). The active
// session's cwd is the repo; the pane reads git as the source of truth, the
// same bounded "re-probe on structural edges" model as the coding rail.
//
// Diff scope is selectable: 'uncommitted' (the review-before-commit default),
// 'branch' (committed work vs the trunk merge-base), and 'lastTurn' (everything
// since the current turn began, driven by the $reviewTurnBase HEAD capture).
// Mutations and the commit/PR ship bar are uncommitted-only.

// Must match the review <Pane id> in desktop-controller (the forced-reveal
// event is addressed by pane id).
export const REVIEW_PANE_ID = 'review'

const OPEN_KEY = 'hermes.desktop.reviewOpen'
const COMMIT_DEFAULT_KEY = 'hermes.desktop.reviewCommitDefault'
const TREE_MODE_KEY = 'hermes.desktop.reviewTreeMode'
const SCOPE_KEY = 'hermes.desktop.reviewScope'
const SELECTED_KEY = 'hermes.desktop.reviewSelectedPath'
const REVIEW_REFRESH_DEBOUNCE_MS = 100
const SHIP_INFO_STALE_MS = 30_000

// Persisted so the pane stays open across reloads (like the other rail panes).
// Simple mode rests it closed without touching the preference; ⌘G still opens
// it for the session.
const $reviewOpenPref = persistentAtom(OPEN_KEY, false, Codecs.bool)

export const $reviewOpen = modeBound('reviewOpen', $reviewOpenPref, open => $reviewOpenPref.set(open))

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

// Diff scope for the review pane. 'uncommitted' (staged+unstaged+untracked) is
// the review-before-commit default; 'branch' shows committed work vs the trunk
// merge-base; 'lastTurn' shows everything (committed + uncommitted) since the
// most recent turn began. Persisted like the tree-mode toggle.
export const $reviewScope = persistentAtom<HermesReviewScope>(SCOPE_KEY, 'uncommitted', {
  decode: raw => (raw === 'branch' || raw === 'lastTurn' ? raw : 'uncommitted'),
  encode: value => value
})

// HEAD sha per repo cwd, captured when the most recent turn started in that
// repo — the baseRef for the 'lastTurn' scope. Keyed by cwd so a pinned tile
// worktree or a background session gets its own baseline instead of inheriting
// the foreground cwd's. Empty until a turn has been observed for a given cwd.
//
// Normalization contract: keys are written from `state.cwd.trim()` at capture
// and read through `repoCwd()` (which trims too). Both sides ultimately carry
// the same gateway-reported absolute path for a given repo, so they match
// byte-for-byte; any other divergence (trailing slash, symlink, case) would
// silently miss and degrade lastTurn to empty rather than mislead.
//
// Capture is best-effort and inherently racy: HEAD is resolved asynchronously
// after `busy` flips, so a turn that commits immediately can beat the reply
// and bake its own commits into the baseline (lastTurn then shows slightly
// less than the turn changed). Accepted — the alternative is blocking turn
// start on a git probe.
export const $reviewTurnBase = atom<Record<string, string>>({})

// Cap on tracked baselines. Bounded by repos-worked-in in practice, but a
// long-lived session shouldn't hoard stale entries for abandoned worktrees:
// beyond the cap, the least-recently-captured cwd is evicted.
const MAX_TURN_BASES = 8

function recordTurnBase(cwd: string, sha: string): void {
  const prev = $reviewTurnBase.get()

  if (prev[cwd] === sha) {
    return
  }

  // Re-insert the cwd so object key order tracks recency (oldest first),
  // then evict from the front beyond the cap.
  const rest = Object.entries(prev).filter(([key]) => key !== cwd)

  while (rest.length >= MAX_TURN_BASES) {
    rest.shift()
  }

  rest.push([cwd, sha])
  $reviewTurnBase.set(Object.fromEntries(rest))
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

// The pane's repo scope. Null = follow the ACTIVE session's cwd (the classic
// behavior). A tile's rail opens the pane pinned to ITS worktree instead —
// tiles can sit in different worktrees than main, and reviewing "the diff I'm
// looking at" must mean that tile's repo, not whatever main happens to be on.
export const $reviewScopeCwd = atom<null | string>(null)
// The composer target that opened the pane. The review pane is a shared
// surface, but its "let the agent ship it" action must return to the session
// whose worktree the user is reviewing, not broadcast to every mounted tile.
export const $reviewScopeTarget = atom('main')

/** The repo the pane is reading right now: its pinned scope, else the active
 *  session's cwd. Exported for pane helpers that join repo-relative paths. */
export const reviewRepoCwd = (): null | string => $reviewScopeCwd.get()?.trim() || $currentCwd.get()?.trim() || null

const repoCwd = reviewRepoCwd

type ReviewBridge = NonNullable<NonNullable<NonNullable<Window['hermesDesktop']>['git']>['review']>
let reviewRefreshSeq = 0
let reviewRefreshTimer: ReturnType<typeof setTimeout> | null = null
let reviewSelectionSeq = 0
let shipInfoSeq = 0
let shipInfoLastCheckedAt = 0

// The two things every review op needs: the repo cwd + the IPC bridge. Null when
// either is missing (no session, remote backend), so callers bail in one line.
function reviewCtx(): { cwd: string; review: ReviewBridge } | null {
  const cwd = repoCwd()
  const review = desktopGit()?.review

  return cwd && review ? { cwd, review } : null
}

// The scope + base ref for the current read. 'lastTurn' diffing is baseRef-
// driven (Electron recomputes the merge-base itself for 'branch'), so only the
// last-turn baseline — looked up for the pane's own repo cwd — is passed.
function reviewReadParams(): { scope: HermesReviewScope; baseRef: null | string } {
  const scope = $reviewScope.get()

  if (scope !== 'lastTurn') {
    return { scope, baseRef: null }
  }

  const cwd = repoCwd()

  return { scope, baseRef: cwd ? ($reviewTurnBase.get()[cwd] ?? null) : null }
}

// ── Reads ────────────────────────────────────────────────────────────────────

export async function refreshReview(): Promise<boolean> {
  // A direct refresh supersedes an already-queued debounce for the same live
  // context. Without this, openReviewForPath() can start a list request only
  // for the timer to overtake it before the requested file is selected.
  if (reviewRefreshTimer) {
    clearTimeout(reviewRefreshTimer)
    reviewRefreshTimer = null
  }

  const intentCwd = repoCwd()
  const ctx = reviewCtx()
  const seq = (reviewRefreshSeq += 1)
  const selectionSeq = reviewSelectionSeq
  const ownsList = () => seq === reviewRefreshSeq && repoCwd() === intentCwd

  if (!$reviewOpen.get() || !ctx) {
    if (ownsList()) {
      $reviewFiles.set([])
      $reviewIsRepo.set(Boolean(ctx))

      // Critical: clear loading on the no-cwd / not-a-repo path too. It's set
      // true (optimistically) before a refresh is scheduled, so skipping it here
      // strands the pane on a forever-skeleton for a fresh, detached chat.
      $reviewLoading.set(false)
    }

    return ownsList()
  }

  const { cwd, review } = ctx

  $reviewIsRepo.set(true)
  $reviewLoading.set(true)

  try {
    const { scope, baseRef } = reviewReadParams()
    const result = await review.list(cwd, scope, baseRef)

    // Ignore a result that resolved after the cwd moved on.
    if (!ownsList()) {
      return false
    }

    // Hide dep/build/cache dirs and OS noise even when the repo tracks them —
    // .gitignored paths are already dropped upstream by `git status`.
    const files = result.files.filter(file => !isExcludedPath(file.path))

    $reviewFiles.set(files)

    // Drop the selection if the file is gone (staged away, reverted) so the diff
    // pane doesn't strand on a ghost; otherwise lazily fetch its diff so a
    // restored (persisted) selection re-renders on boot.
    if (selectionSeq === reviewSelectionSeq) {
      const selected = $reviewSelectedPath.get()
      const selectedFile = selected ? files.find(file => file.path === selected) : null

      if (selected && !selectedFile) {
        clearReviewSelection()
      } else if (selectedFile && $reviewDiff.get() === null) {
        void selectReviewFile(selectedFile)
      }
    }

    return true
  } catch {
    if (ownsList()) {
      $reviewFiles.set([])
    }

    return ownsList()
  } finally {
    if (ownsList()) {
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

  // Scheduling is itself a newer list intent. Invalidate in-flight work now,
  // not when the debounce fires, so an older catch/finally cannot mutate the
  // repository that is waiting for the timer.
  reviewRefreshSeq += 1

  reviewRefreshTimer = setTimeout(() => {
    reviewRefreshTimer = null
    void refreshReview()
  }, REVIEW_REFRESH_DEBOUNCE_MS)
}

export async function selectReviewFile(file: HermesReviewFile): Promise<void> {
  const seq = (reviewSelectionSeq += 1)
  $reviewSelectedPath.set(file.path)

  const intentCwd = repoCwd()
  const ctx = reviewCtx()

  const ownsSelection = () =>
    seq === reviewSelectionSeq && $reviewSelectedPath.get() === file.path && repoCwd() === intentCwd

  if (!ctx) {
    if (ownsSelection()) {
      $reviewDiff.set(null)
    }

    return
  }

  $reviewDiffLoading.set(true)

  try {
    const { scope, baseRef } = reviewReadParams()
    const diff = await ctx.review.diff(ctx.cwd, file.path, scope, baseRef, file.staged)

    if (ownsSelection()) {
      $reviewDiff.set(diff || '')
    }
  } catch {
    if (ownsSelection()) {
      $reviewDiff.set('')
    }
  } finally {
    if (ownsSelection()) {
      $reviewDiffLoading.set(false)
    }
  }
}

export function clearReviewSelection(): void {
  reviewSelectionSeq += 1
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

/** Open the pane scoped to `scopeCwd` (a tile's worktree), or to the active
 *  session's cwd when null — see `$reviewScopeCwd`. Keep the originating
 *  composer target alongside it for agent-ship actions. */
export function openReview(scopeCwd: null | string = null, scopeTarget = 'main'): void {
  $reviewScopeCwd.set(scopeCwd?.trim() || null)
  $reviewScopeTarget.set(scopeTarget.trim() || 'main')
  $reviewOpen.set(true)
  recordFeatureUse('review_pane')
  void refreshReview()
  void refreshShipInfo()
}

export function closeReview(): void {
  noteAreaClosed('review_pane')
  $reviewOpen.set(false)
  $reviewScopeCwd.set(null)
  $reviewScopeTarget.set('main')
  clearReviewSelection()
}

export function toggleReview(scopeCwd: null | string = null, scopeTarget = 'main'): void {
  // Narrow width: the pane is a collapsed overlay (like the sidebar under ⌘B).
  // Make sure its data is loaded, then slide it in/out via the forced-reveal pin
  // — never the docked open state, which a 0px track would render invisibly.
  if (matchesQuery(SIDEBAR_COLLAPSE_MEDIA_QUERY)) {
    const target = scopeTarget.trim() || 'main'

    const originChanged =
      ($reviewScopeCwd.get() ?? null) !== (scopeCwd?.trim() || null) || $reviewScopeTarget.get() !== target

    if (!$reviewOpen.get() || originChanged) {
      openReview(scopeCwd, target)
    }

    window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id: REVIEW_PANE_ID } }))

    return
  }

  // Ask the TREE, not `$reviewOpen`. The store stays true while the pane sits
  // behind a sibling tab in the right column or inside a minimized zone, so a
  // boolean flip spent the press re-asserting a value it already held and ⌘G
  // read as a dead key. `revealReview` fronts and un-minimizes; only close when
  // the diff is genuinely the thing on screen.
  if (isPaneVisible(REVIEW_PANE_ID)) {
    closeReview()
  } else {
    revealReview(scopeCwd, scopeTarget)
  }
}

/**
 * Open the review pane and bring it into view. Unlike `toggleReview` this never
 * closes an already-open pane — it's the "take me to the diff" entry point used
 * by the transcript's changed-files card.
 */
export function revealReview(scopeCwd: null | string = null, scopeTarget = 'main'): void {
  const wasOpen = $reviewOpen.get()
  const target = scopeTarget.trim() || 'main'

  if (!wasOpen) {
    openReview(scopeCwd, target)
  } else if (($reviewScopeCwd.get() ?? null) !== (scopeCwd?.trim() || null) || $reviewScopeTarget.get() !== target) {
    // Already open but on another worktree's diff — re-home it. The scope
    // subscription below clears the stale list and re-probes. Keep the
    // originating composer target alongside the cwd for the agent-ship action.
    $reviewScopeCwd.set(scopeCwd?.trim() || null)
    $reviewScopeTarget.set(target)
  }

  if (matchesQuery(SIDEBAR_COLLAPSE_MEDIA_QUERY)) {
    // The reveal pin is a toggle, so only fire it when the overlay isn't
    // already slid in — otherwise "show me the diff" would hide the pane.
    if (!wasOpen) {
      window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id: REVIEW_PANE_ID } }))
    }

    return
  }

  revealTreePane(REVIEW_PANE_ID)
}

/** The changed file matching a tool-reported path (absolute or repo-relative). */
function matchReviewFile(files: readonly HermesReviewFile[], path: string): HermesReviewFile | undefined {
  const target = path.replace(/\\/g, '/').replace(/\/+$/, '')

  if (!target) {
    return undefined
  }

  return files.find(file => {
    const candidate = file.path.replace(/\\/g, '/')

    return candidate === target || target.endsWith(`/${candidate}`) || candidate.endsWith(`/${target}`)
  })
}

/**
 * Open the review pane on one file's diff. The path comes from a tool call, so
 * it may be absolute while git reports repo-relative — match on the tail.
 */
export async function openReviewForPath(
  path: string,
  scopeCwd: null | string = null,
  scopeTarget = 'main'
): Promise<void> {
  revealReview(scopeCwd, scopeTarget)
  const cwd = repoCwd()
  const refreshed = await refreshReview()

  if (!refreshed || repoCwd() !== cwd) {
    return
  }

  const file = matchReviewFile($reviewFiles.get(), path)

  if (file) {
    await selectReviewFile(file)

    return
  }

  // The click promised this file's diff, but the pane's scope can't show it:
  // a 'lastTurn' baseline lost to a restart (the turn-base map is in-memory),
  // a 'branch' base that doesn't resolve, or changes the agent already
  // committed. Each dead-ends the pane on "No diffs" while the transcript
  // card — fed by tool events — still lists the file. Try the other read
  // scopes before giving up so the click lands somewhere real; the pane stays
  // coherent because list, selection and diff all come from that scope.
  const originalScope = $reviewScope.get()

  for (const scope of ['uncommitted', 'branch', 'lastTurn'] as const) {
    if (scope === originalScope) {
      continue
    }

    $reviewScope.set(scope)
    const ok = await refreshReview()

    if (!ok || repoCwd() !== cwd) {
      return
    }

    const match = matchReviewFile($reviewFiles.get(), path)

    if (match) {
      await selectReviewFile(match)

      return
    }
  }

  // Nothing lists the file (stale card); leave the user's scope untouched.
  $reviewScope.set(originalScope)
  await refreshReview()
}

// ── Mutations ────────────────────────────────────────────────────────────────

// Run a git mutation then re-sync both the review list and the rail's +/- (the
// working tree changed). A failure is swallowed by the caller's notify wrapper.
async function afterMutation(): Promise<void> {
  await refreshReview()
  void refreshRepoStatus(repoCwd())

  const selected = $reviewSelectedPath.get()
  const file = selected ? $reviewFiles.get().find(f => f.path === selected) : null

  // Re-fetch the open diff (staging flips which diff — cached vs worktree).
  if (file) {
    void selectReviewFile(file)
  }
}

export async function stageReviewFile(path: null | string): Promise<void> {
  if ($reviewScope.get() !== 'uncommitted') {
    return
  }

  await desktopGit()?.review?.stage(repoCwd() ?? '', path)
  await afterMutation()
}

export async function unstageReviewFile(path: null | string): Promise<void> {
  if ($reviewScope.get() !== 'uncommitted') {
    return
  }

  await desktopGit()?.review?.unstage(repoCwd() ?? '', path)
  await afterMutation()
}

export async function revertReviewFile(path: null | string): Promise<void> {
  if ($reviewScope.get() !== 'uncommitted') {
    return
  }

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
    void refreshRepoStatus(repoCwd())
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

    // The session recorded its branch when it started; the checkout may have
    // moved since, so bind the conversation to the branch the PR actually came
    // from — otherwise a session that began on trunk badges whatever else lives
    // on trunk, or nothing.
    const session = $sessions.get().find(s => s.id === $selectedStoredSessionId.get())
    const branch = repoStatusForCwd(ctx.cwd).get()?.branch

    if (session?.git_repo_root && branch) {
      stampSessionPrBranch(session.id, session.git_repo_root, branch)
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

// Turn settled: final list refresh + the slower gh/PR re-check. `$busy` is the
// ACTIVE session's flag, so this settle edge only fires for the foreground turn.
let prevBusy = $busy.get()

$busy.subscribe(busy => {
  if (prevBusy && !busy && $reviewOpen.get()) {
    scheduleReviewRefresh()
    refreshShipInfoIfStale()
  }

  prevBusy = busy
})

// Last-turn baseline capture: HEAD per repo cwd at the moment a turn started in
// that repo. Driven off $sessionStates (every session — tiles and background
// sessions included) rather than $busy, which only mirrors the foreground: a
// pinned tile worktree or a background turn would otherwise inherit the wrong
// (or no) baseline. Best-effort — a non-repo / remote backend resolves null and
// lastTurn simply shows empty for that cwd.
let prevBusyByRuntime: Record<string, boolean> = Object.fromEntries(
  Object.entries($sessionStates.get()).map(([runtimeId, state]) => [runtimeId, state.busy])
)

$sessionStates.subscribe(states => {
  const nextBusy: Record<string, boolean> = {}

  for (const [runtimeId, state] of Object.entries(states)) {
    nextBusy[runtimeId] = state.busy
    const wasBusy = prevBusyByRuntime[runtimeId] ?? false

    if (state.busy && !wasBusy) {
      const cwd = state.cwd?.trim()

      if (cwd) {
        void desktopGit()
          ?.review?.revParse(cwd, 'HEAD')
          .then(sha => {
            if (sha) {
              recordTurnBase(cwd, sha)
            }
          })
      }
    }
  }

  prevBusyByRuntime = nextBusy
})

// The pane's repo moved under it. For the classic (unscoped) pane that's the
// active session's cwd changing; for a scoped pane it's a re-home to another
// tile's worktree — and a main-pane cwd change is deliberately IGNORED while
// scoped, so switching sessions in main can't yank the diff you're reviewing.
// Either way: clear the stale file list + selection up front so the pane drops
// straight to its loading skeleton instead of blipping the previous repo's
// diff into the new one.
function onReviewRepoMoved(): void {
  if ($reviewOpen.get()) {
    clearReviewSelection()
    $reviewFiles.set([])
    $reviewLoading.set(true)
    scheduleReviewRefresh()
    void refreshShipInfo()
  }
}

$currentCwd.subscribe(() => {
  if (!$reviewScopeCwd.get()) {
    onReviewRepoMoved()
  }
})

let prevScopeCwd = $reviewScopeCwd.get()

$reviewScopeCwd.subscribe(scope => {
  if (scope !== prevScopeCwd) {
    prevScopeCwd = scope
    onReviewRepoMoved()
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
