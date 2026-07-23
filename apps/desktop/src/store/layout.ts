import { atom, computed, type ReadableAtom, type WritableAtom } from 'nanostores'

import { SIDEBAR_COLLAPSE_MEDIA_QUERY } from '@/app/layout-constants'
import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { matchesQuery } from '@/hooks/use-media-query'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { arraysEqual, insertUniqueId, readKey } from '@/lib/storage'

import { $paneStates, ensurePaneRegistered, setPaneOpen, setPaneWidthOverride, togglePane } from './panes'

export const SIDEBAR_DEFAULT_WIDTH = 237
export const SIDEBAR_MAX_WIDTH = 360
// Open at the same width as the sessions sidebar so the two rails match, but
// allow shrinking well below that (~30% under the old 14rem floor) for users who
// want a narrow tree.
export const FILE_BROWSER_DEFAULT_WIDTH = `${SIDEBAR_DEFAULT_WIDTH}px`
export const FILE_BROWSER_MIN_WIDTH = '10rem'
export const FILE_BROWSER_MAX_WIDTH = '20rem'

export const SIDEBAR_SESSIONS_PAGE_SIZE = 50

const SIDEBAR_PINNED_STORAGE_KEY = 'hermes.desktop.pinnedSessions'
const SIDEBAR_AGENTS_GROUPED_STORAGE_KEY = 'hermes.desktop.agentsGroupedByWorkspace'
const SIDEBAR_CRON_OPEN_STORAGE_KEY = 'hermes.desktop.sidebarCronOpen'
const SIDEBAR_MESSAGING_OPEN_STORAGE_KEY = 'hermes.desktop.sidebarMessagingOpen'
const SIDEBAR_SESSION_ORDER_STORAGE_KEY = 'hermes.desktop.sessionOrder'
const SIDEBAR_SESSION_ORDER_MANUAL_STORAGE_KEY = 'hermes.desktop.sessionOrder.manual'
const SIDEBAR_WORKSPACE_ORDER_STORAGE_KEY = 'hermes.desktop.workspaceOrder'
const SIDEBAR_WORKSPACE_PARENT_ORDER_STORAGE_KEY = 'hermes.desktop.workspaceParentOrder'
const SIDEBAR_PROJECT_ORDER_STORAGE_KEY = 'hermes.desktop.projectOrder'
const SIDEBAR_WORKSPACE_COLLAPSED_STORAGE_KEY = 'hermes.desktop.workspaceCollapsed'
const SIDEBAR_WORKSPACE_NODE_OPEN_STORAGE_KEY = 'hermes.desktop.workspaceNodeOpen'
const SIDEBAR_DISMISSED_AUTO_PROJECTS_STORAGE_KEY = 'hermes.desktop.dismissedAutoProjects'
const SIDEBAR_DISMISSED_WORKTREES_STORAGE_KEY = 'hermes.desktop.dismissedWorktrees'
const PANES_FLIPPED_STORAGE_KEY = 'hermes.desktop.panesFlipped'
const RIGHT_RAIL_ACTIVE_TAB_STORAGE_KEY = 'hermes.desktop.rightRailActiveTab'

export const CHAT_SIDEBAR_PANE_ID = 'chat-sidebar'
export const FILE_BROWSER_PANE_ID = 'file-browser'
export const PREVIEW_PANE_ID = 'preview'
export const RIGHT_RAIL_PREVIEW_TAB_ID = 'preview'

export type RightRailTabId = typeof RIGHT_RAIL_PREVIEW_TAB_ID | `file:${string}`

ensurePaneRegistered(CHAT_SIDEBAR_PANE_ID, { open: true })
ensurePaneRegistered(FILE_BROWSER_PANE_ID, { open: false })
ensurePaneRegistered(PREVIEW_PANE_ID, { open: true })

export const $sidebarOpen: ReadableAtom<boolean> = computed(
  $paneStates,
  states => states[CHAT_SIDEBAR_PANE_ID]?.open ?? true
)

export const $fileBrowserOpen: ReadableAtom<boolean> = computed(
  $paneStates,
  states => states[FILE_BROWSER_PANE_ID]?.open ?? false
)

// Persisted so a relaunch reopens the same rail tab. A restored file-tab id with
// no matching tab is reconciled back to the preview tab in the preview store.
export const $rightRailActiveTabId = persistentAtom<RightRailTabId>(
  RIGHT_RAIL_ACTIVE_TAB_STORAGE_KEY,
  RIGHT_RAIL_PREVIEW_TAB_ID,
  { decode: raw => raw as RightRailTabId, encode: tabId => tabId }
)

export const $sidebarWidth: ReadableAtom<number> = computed($paneStates, states => {
  const override = states[CHAT_SIDEBAR_PANE_ID]?.widthOverride

  return typeof override === 'number' ? override : SIDEBAR_DEFAULT_WIDTH
})

export const $pinnedSessionIds = persistentAtom(SIDEBAR_PINNED_STORAGE_KEY, [] as string[], Codecs.stringArray)
export const $sidebarSessionOrderIds = persistentAtom(
  SIDEBAR_SESSION_ORDER_STORAGE_KEY,
  [] as string[],
  Codecs.stringArray
)
export const $sidebarSessionOrderManual = persistentAtom(SIDEBAR_SESSION_ORDER_MANUAL_STORAGE_KEY, false, Codecs.bool)
export const $sidebarWorkspaceOrderIds = persistentAtom(
  SIDEBAR_WORKSPACE_ORDER_STORAGE_KEY,
  [] as string[],
  Codecs.stringArray
)
// Order of the top-level repo "parent" groups in the worktree tree (worktrees
// within a parent reuse $sidebarWorkspaceOrderIds).
export const $sidebarWorkspaceParentOrderIds = persistentAtom(
  SIDEBAR_WORKSPACE_PARENT_ORDER_STORAGE_KEY,
  [] as string[],
  Codecs.stringArray
)
// Manual drag-order of projects in the overview. Empty = the deterministic
// default sort (active first, explicit before auto, by recency); once the user
// drags a project their order wins (orderByIds surfaces new projects on top).
export const $sidebarProjectOrderIds = persistentAtom(
  SIDEBAR_PROJECT_ORDER_STORAGE_KEY,
  [] as string[],
  Codecs.stringArray
)
// Explicit open/collapse state for sidebar workspace nodes AND review file-tree
// folders, keyed by stable node id (repo root / worktree path / `review:<path>`).
// A stored value is the user's EXPLICIT choice (true = open, false = collapsed);
// an absent id falls back to the caller's `defaultOpen`.
//
// We store the RESOLVED boolean, NOT an XOR against the default (the old
// `workspaceCollapsed` set did the latter). The XOR was buggy for any node
// whose default *flips*: a worktree lane defaults collapsed while empty and
// open once it holds a session, so an explicit expand of an empty lane silently
// re-read as a "collapse" the moment the lane gained a row — collapsing the very
// lane the user had just opened to work in. An absolute value survives that flip.
export const $sidebarWorkspaceNodeOpen = persistentAtom<Record<string, boolean>>(
  SIDEBAR_WORKSPACE_NODE_OPEN_STORAGE_KEY,
  migrateWorkspaceCollapsedIds(),
  Codecs.json<Record<string, boolean>>(raw => {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
      return {}
    }

    return Object.fromEntries(
      Object.entries(raw).filter((entry): entry is [string, boolean] => typeof entry[1] === 'boolean')
    )
  })
)

// One-time migration off the old XOR `workspaceCollapsed` string[]. Every id in
// it was a deviation from a DEFAULT-OPEN node (repos + file-tree folders, whose
// default never flips), so it maps cleanly to `collapsed` (false). The rare
// empty-worktree-lane "expand" record maps to `false` too, which just returns
// that lane to its default-collapsed state — self-healing, not a regression.
function migrateWorkspaceCollapsedIds(): Record<string, boolean> {
  if (readKey(SIDEBAR_WORKSPACE_NODE_OPEN_STORAGE_KEY) !== null) {
    return {}
  }

  const raw = readKey(SIDEBAR_WORKSPACE_COLLAPSED_STORAGE_KEY)

  if (raw === null) {
    return {}
  }

  try {
    const ids = JSON.parse(raw) as unknown

    if (!Array.isArray(ids)) {
      return {}
    }

    return Object.fromEntries(
      ids.filter((id): id is string => typeof id === 'string' && id.length > 0).map(id => [id, false])
    )
  } catch {
    return {}
  }
}

// Auto-derived (git-repo) projects the user has dismissed ("deleted") from the
// overview. Keyed by repo-root path; persisted so they stay hidden. Explicit
// projects are deleted for real instead — this only declutters the auto tier.
export const $dismissedAutoProjectIds = persistentAtom(
  SIDEBAR_DISMISSED_AUTO_PROJECTS_STORAGE_KEY,
  [] as string[],
  Codecs.stringArray
)
// Worktree rows removed from the UI after a `git worktree remove`. The on-disk
// dir is gone but historical sessions still reference its path, so we hide the
// row by id (worktree path) to keep "remove" feeling real.
export const $dismissedWorktreeIds = persistentAtom(
  SIDEBAR_DISMISSED_WORKTREES_STORAGE_KEY,
  [] as string[],
  Codecs.stringArray
)
export const $sidebarPinsOpen = atom(true)
export const $sidebarRecentsOpen = atom(true)
// Cron-job sessions live in their own section below recents, collapsed by
// default (it only renders at all when cron sessions exist) so the
// scheduler's `[IMPORTANT: …]` first-message previews don't spam recents.
export const $sidebarCronOpen = persistentAtom(SIDEBAR_CRON_OPEN_STORAGE_KEY, false, Codecs.bool)
// Messaging platform sections collapse by default (they can be numerous and
// tall). We persist the ids the user has *explicitly expanded*, so the default
// stays collapsed unless they've opened a platform before.
export const $sidebarMessagingOpenIds = persistentAtom(
  SIDEBAR_MESSAGING_OPEN_STORAGE_KEY,
  [] as string[],
  Codecs.stringArray
)
export const $sidebarAgentsGrouped = persistentAtom(SIDEBAR_AGENTS_GROUPED_STORAGE_KEY, false, Codecs.bool)
// When true, the sessions sidebar moves to the right and the file browser +
// preview rail move to the left — a mirror of the default layout.
export const $panesFlipped = persistentAtom(PANES_FLIPPED_STORAGE_KEY, false, Codecs.bool)
export const $isSidebarResizing = atom(false)
export const $sessionsLimit = atom(SIDEBAR_SESSIONS_PAGE_SIZE)

// Resolve a node's open state against its default (absent = follow default).
export function workspaceNodeOpen(id: string, defaultOpen = true): boolean {
  return $sidebarWorkspaceNodeOpen.get()[id] ?? defaultOpen
}

// Force a node open/collapsed. Stable across a default flip — used by "+ new
// session" to reveal the lane it targets and keep it open once it's populated.
export function setWorkspaceNodeOpen(id: string, open: boolean): void {
  const current = $sidebarWorkspaceNodeOpen.get()

  if (current[id] === open) {
    return
  }

  $sidebarWorkspaceNodeOpen.set({ ...current, [id]: open })
}

// Toggle a repo/worktree/file-tree node relative to its current resolved state.
export function toggleWorkspaceNodeCollapsed(id: string, defaultOpen = true): void {
  setWorkspaceNodeOpen(id, !workspaceNodeOpen(id, defaultOpen))
}

// Dismiss ("delete") an auto-derived project from the overview.
export function dismissAutoProject(id: string): void {
  const current = $dismissedAutoProjectIds.get()

  if (!current.includes(id)) {
    $dismissedAutoProjectIds.set([...current, id])
  }
}

// Hide a worktree row after it's been removed via git.
export function dismissWorktree(id: string): void {
  const current = $dismissedWorktreeIds.get()

  if (!current.includes(id)) {
    $dismissedWorktreeIds.set([...current, id])
  }
}

// A hidden worktree becomes visible again as soon as the user explicitly starts
// or opens work there (for example, selecting an already-checked-out branch).
export function restoreWorktree(id: string): void {
  const current = $dismissedWorktreeIds.get()

  if (current.includes(id)) {
    $dismissedWorktreeIds.set(current.filter(worktreeId => worktreeId !== id))
  }
}

export function setSidebarWidth(width: number) {
  const bounded = Math.min(SIDEBAR_MAX_WIDTH, Math.max(SIDEBAR_DEFAULT_WIDTH, width))
  setPaneWidthOverride(CHAT_SIDEBAR_PANE_ID, bounded)
}

// Below the collapse breakpoint a collapsible rail leaves the grid and lives as
// a hover/pin overlay, so open/toggle must route through the reveal event — the
// docked `open` flag renders a 0px track invisibly. Centralised here so every
// caller (titlebar, keybinds, session-search, reveal-file) inherits it instead
// of re-deriving the narrow branch. Returns true when it handled the intent.
function revealNarrowPane(id: string, mode: 'close' | 'open' | 'toggle'): boolean {
  if (typeof window === 'undefined' || !matchesQuery(SIDEBAR_COLLAPSE_MEDIA_QUERY)) {
    return false
  }

  window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id, mode } }))

  return true
}

export function setSidebarOpen(open: boolean) {
  setPaneOpen(CHAT_SIDEBAR_PANE_ID, open)
  revealNarrowPane(CHAT_SIDEBAR_PANE_ID, open ? 'open' : 'close')
}

export function toggleSidebarOpen() {
  if (!revealNarrowPane(CHAT_SIDEBAR_PANE_ID, 'toggle')) {
    togglePane(CHAT_SIDEBAR_PANE_ID)
  }
}

export function toggleFileBrowserOpen() {
  if (!revealNarrowPane(FILE_BROWSER_PANE_ID, 'toggle')) {
    togglePane(FILE_BROWSER_PANE_ID)
  }
}

export function setFileBrowserOpen(open: boolean) {
  setPaneOpen(FILE_BROWSER_PANE_ID, open)
  revealNarrowPane(FILE_BROWSER_PANE_ID, open ? 'open' : 'close')
}

// "Reveal this file in the file-browser tree" — an absolute path the tree
// subscribes to, expanding ancestor folders and selecting/scrolling to it. Reset
// to null by the tree once consumed.
export const $revealInTreeRequest = atom<null | string>(null)

export function revealFileInTree(path: string): void {
  setFileBrowserOpen(true)
  $revealInTreeRequest.set(path)
}

// Hotkey → focus the sessions search field. Opens the sidebar first, then lets
// the field (which only mounts when the sidebar is open) subscribe + focus.
export const SESSION_SEARCH_FOCUS_EVENT = 'hermes:focus-session-search'

export function requestSessionSearchFocus() {
  setSidebarOpen(true)

  if (typeof window !== 'undefined') {
    window.setTimeout(() => window.dispatchEvent(new CustomEvent(SESSION_SEARCH_FOCUS_EVENT)), 0)
  }
}

export function togglePanesFlipped() {
  $panesFlipped.set(!$panesFlipped.get())
}

export function selectRightRailTab(id: RightRailTabId) {
  $rightRailActiveTabId.set(id)
}

export function setSidebarPinsOpen(open: boolean) {
  $sidebarPinsOpen.set(open)
}

export function setSidebarRecentsOpen(open: boolean) {
  $sidebarRecentsOpen.set(open)
}

export function setSidebarCronOpen(open: boolean) {
  $sidebarCronOpen.set(open)
}

export function toggleSidebarMessagingOpen(sourceId: string) {
  const current = $sidebarMessagingOpenIds.get()

  $sidebarMessagingOpenIds.set(
    current.includes(sourceId) ? current.filter(id => id !== sourceId) : [...current, sourceId]
  )
}

export function setSidebarAgentsGrouped(grouped: boolean) {
  $sidebarAgentsGrouped.set(grouped)
}

// Write an order list only when it actually changed, so an identical drag
// result keeps the same array reference and subscribers don't churn.
function setOrderIds($atom: WritableAtom<string[]>, ids: string[]) {
  if (!arraysEqual($atom.get(), ids)) {
    $atom.set(ids)
  }
}

export function setSidebarSessionOrderIds(ids: string[]) {
  setOrderIds($sidebarSessionOrderIds, ids)
}

export function setSidebarSessionOrderManual(manual: boolean) {
  if ($sidebarSessionOrderManual.get() !== manual) {
    $sidebarSessionOrderManual.set(manual)
  }
}

export function setSidebarWorkspaceOrderIds(ids: string[]) {
  setOrderIds($sidebarWorkspaceOrderIds, ids)
}

export function setSidebarWorkspaceParentOrderIds(ids: string[]) {
  setOrderIds($sidebarWorkspaceParentOrderIds, ids)
}

export function setSidebarProjectOrderIds(ids: string[]) {
  setOrderIds($sidebarProjectOrderIds, ids)
}

export function setSidebarResizing(resizing: boolean) {
  $isSidebarResizing.set(resizing)
}

export function pinSession(sessionId: string, index?: number) {
  const prev = $pinnedSessionIds.get()

  setOrderIds($pinnedSessionIds, insertUniqueId(prev, sessionId, index ?? prev.filter(id => id !== sessionId).length))
}

export function unpinSession(sessionId: string) {
  setOrderIds(
    $pinnedSessionIds,
    $pinnedSessionIds.get().filter(id => id !== sessionId)
  )
}

// Replace the whole pinned order at once (drag-reorder hands back the new order
// rather than a single move). Keep only ids that are actually pinned so a stale
// row can't smuggle an unpinned id into the store.
export function setPinnedSessionOrder(ids: string[]) {
  const prev = $pinnedSessionIds.get()
  const pinned = new Set(prev)
  const next = ids.filter(id => pinned.has(id))

  if (next.length === prev.length && !arraysEqual(prev, next)) {
    $pinnedSessionIds.set(next)
  }
}

export function bumpSessionsLimit(step: number = SIDEBAR_SESSIONS_PAGE_SIZE) {
  const safeStep = Math.max(1, Math.floor(step))
  $sessionsLimit.set($sessionsLimit.get() + safeStep)
}

export function resetSessionsLimit() {
  if ($sessionsLimit.get() !== SIDEBAR_SESSIONS_PAGE_SIZE) {
    $sessionsLimit.set(SIDEBAR_SESSIONS_PAGE_SIZE)
  }
}
