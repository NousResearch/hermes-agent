import { atom, computed, type ReadableAtom } from 'nanostores'

import {
  arraysEqual,
  insertUniqueId,
  persistBoolean,
  persistStringArray,
  storedBoolean,
  storedStringArray
} from '@/lib/storage'

import { $paneStates, ensurePaneRegistered, setPaneOpen, setPaneWidthOverride, togglePane } from './panes'

export const SIDEBAR_DEFAULT_WIDTH = 237
export const SIDEBAR_MAX_WIDTH = 360
// Open at the same width as the sessions sidebar so the two rails match.
export const FILE_BROWSER_DEFAULT_WIDTH = `${SIDEBAR_DEFAULT_WIDTH}px`
export const FILE_BROWSER_MIN_WIDTH = '14rem'
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
const PANES_FLIPPED_STORAGE_KEY = 'hermes.desktop.panesFlipped'

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

export const $rightRailActiveTabId = atom<RightRailTabId>(RIGHT_RAIL_PREVIEW_TAB_ID)

export const $sidebarWidth: ReadableAtom<number> = computed($paneStates, states => {
  const override = states[CHAT_SIDEBAR_PANE_ID]?.widthOverride

  return typeof override === 'number' ? override : SIDEBAR_DEFAULT_WIDTH
})

export const $pinnedSessionIds = atom(storedStringArray(SIDEBAR_PINNED_STORAGE_KEY))
export const $sidebarSessionOrderIds = atom(storedStringArray(SIDEBAR_SESSION_ORDER_STORAGE_KEY))
export const $sidebarSessionOrderManual = atom(storedBoolean(SIDEBAR_SESSION_ORDER_MANUAL_STORAGE_KEY, false))
export const $sidebarWorkspaceOrderIds = atom(storedStringArray(SIDEBAR_WORKSPACE_ORDER_STORAGE_KEY))
// Order of the top-level repo "parent" groups in the worktree tree (worktrees
// within a parent reuse $sidebarWorkspaceOrderIds).
export const $sidebarWorkspaceParentOrderIds = atom(storedStringArray(SIDEBAR_WORKSPACE_PARENT_ORDER_STORAGE_KEY))
export const $sidebarPinsOpen = atom(true)
// Set by the PaneShell hover-reveal overlay while the sidebar is collapsed; kept
// true the whole time it's a floating overlay (not just while shown) so the
// consumer mounts contents off-screen, ready to slide. ChatSidebar mounts its
// rows on `sidebarOpen || this`.
export const $sidebarOverlayMounted = atom(false)
export const $sidebarRecentsOpen = atom(true)
// Cron-job sessions live in their own section below recents, collapsed by
// default (it only renders at all when cron sessions exist) so the
// scheduler's `[IMPORTANT: …]` first-message previews don't spam recents.
export const $sidebarCronOpen = atom(storedBoolean(SIDEBAR_CRON_OPEN_STORAGE_KEY, false))
// Messaging platform sections collapse by default (they can be numerous and
// tall). We persist the ids the user has *explicitly expanded*, so the default
// stays collapsed unless they've opened a platform before.
export const $sidebarMessagingOpenIds = atom<string[]>(storedStringArray(SIDEBAR_MESSAGING_OPEN_STORAGE_KEY))
export const $sidebarAgentsGrouped = atom(storedBoolean(SIDEBAR_AGENTS_GROUPED_STORAGE_KEY, false))
// When true, the sessions sidebar moves to the right and the file browser +
// preview rail move to the left — a mirror of the default layout.
export const $panesFlipped = atom(storedBoolean(PANES_FLIPPED_STORAGE_KEY, false))
export const $isSidebarResizing = atom(false)
export const $sessionsLimit = atom(SIDEBAR_SESSIONS_PAGE_SIZE)

$pinnedSessionIds.subscribe(ids => persistStringArray(SIDEBAR_PINNED_STORAGE_KEY, [...ids]))
$sidebarCronOpen.subscribe(open => persistBoolean(SIDEBAR_CRON_OPEN_STORAGE_KEY, open))
$sidebarMessagingOpenIds.subscribe(ids => persistStringArray(SIDEBAR_MESSAGING_OPEN_STORAGE_KEY, [...ids]))
$sidebarSessionOrderIds.subscribe(ids => persistStringArray(SIDEBAR_SESSION_ORDER_STORAGE_KEY, [...ids]))
$sidebarSessionOrderManual.subscribe(manual => persistBoolean(SIDEBAR_SESSION_ORDER_MANUAL_STORAGE_KEY, manual))
$sidebarWorkspaceOrderIds.subscribe(ids => persistStringArray(SIDEBAR_WORKSPACE_ORDER_STORAGE_KEY, [...ids]))
$sidebarWorkspaceParentOrderIds.subscribe(ids =>
  persistStringArray(SIDEBAR_WORKSPACE_PARENT_ORDER_STORAGE_KEY, [...ids])
)
$sidebarAgentsGrouped.subscribe(grouped => persistBoolean(SIDEBAR_AGENTS_GROUPED_STORAGE_KEY, grouped))
$panesFlipped.subscribe(flipped => persistBoolean(PANES_FLIPPED_STORAGE_KEY, flipped))

export function setSidebarWidth(width: number) {
  const bounded = Math.min(SIDEBAR_MAX_WIDTH, Math.max(SIDEBAR_DEFAULT_WIDTH, width))
  setPaneWidthOverride(CHAT_SIDEBAR_PANE_ID, bounded)
}

export function setSidebarOpen(open: boolean) {
  setPaneOpen(CHAT_SIDEBAR_PANE_ID, open)
}

export function toggleSidebarOpen() {
  togglePane(CHAT_SIDEBAR_PANE_ID)
}

export function toggleFileBrowserOpen() {
  togglePane(FILE_BROWSER_PANE_ID)
}

export function setFileBrowserOpen(open: boolean) {
  setPaneOpen(FILE_BROWSER_PANE_ID, open)
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

export function setSidebarOverlayMounted(mounted: boolean) {
  $sidebarOverlayMounted.set(mounted)
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

export function setSidebarSessionOrderIds(ids: string[]) {
  if (!arraysEqual($sidebarSessionOrderIds.get(), ids)) {
    $sidebarSessionOrderIds.set(ids)
  }
}

export function setSidebarSessionOrderManual(manual: boolean) {
  if ($sidebarSessionOrderManual.get() !== manual) {
    $sidebarSessionOrderManual.set(manual)
  }
}

export function setSidebarWorkspaceOrderIds(ids: string[]) {
  if (!arraysEqual($sidebarWorkspaceOrderIds.get(), ids)) {
    $sidebarWorkspaceOrderIds.set(ids)
  }
}

export function setSidebarWorkspaceParentOrderIds(ids: string[]) {
  if (!arraysEqual($sidebarWorkspaceParentOrderIds.get(), ids)) {
    $sidebarWorkspaceParentOrderIds.set(ids)
  }
}

export function setSidebarResizing(resizing: boolean) {
  $isSidebarResizing.set(resizing)
}

export function pinSession(sessionId: string, index?: number) {
  const prev = $pinnedSessionIds.get()
  const next = insertUniqueId(prev, sessionId, index ?? prev.filter(id => id !== sessionId).length)

  if (!arraysEqual(prev, next)) {
    $pinnedSessionIds.set(next)
  }
}

export function unpinSession(sessionId: string) {
  const prev = $pinnedSessionIds.get()
  const next = prev.filter(id => id !== sessionId)

  if (!arraysEqual(prev, next)) {
    $pinnedSessionIds.set(next)
  }
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
