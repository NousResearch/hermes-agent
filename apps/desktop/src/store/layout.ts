import { atom, computed, type ReadableAtom } from 'nanostores'

import { setSessionPinned as patchSessionPinned } from '@/hermes'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { arraysEqual, readKey, writeKey } from '@/lib/storage'
import { activeGateway } from '@/store/gateway'

import { $paneStates, ensurePaneRegistered, setPaneOpen, setPaneWidthOverride, togglePane } from './panes'
import { $activeSessionId, $selectedStoredSessionId, $sessions, setSessions } from './session'
import { broadcastSessionsChanged } from './session-sync'

export const SIDEBAR_DEFAULT_WIDTH = 237
export const SIDEBAR_MAX_WIDTH = 360
// Open at the same width as the sessions sidebar so the two rails match, but
// allow shrinking well below that (~30% under the old 14rem floor) for users who
// want a narrow tree.
export const FILE_BROWSER_DEFAULT_WIDTH = `${SIDEBAR_DEFAULT_WIDTH}px`
export const FILE_BROWSER_MIN_WIDTH = '10rem'
export const FILE_BROWSER_MAX_WIDTH = '20rem'

export const SIDEBAR_SESSIONS_PAGE_SIZE = 50

export const SIDEBAR_PINNED_STORAGE_KEY = 'hermes.desktop.pinnedSessions'
const SIDEBAR_AGENTS_GROUPED_STORAGE_KEY = 'hermes.desktop.agentsGroupedByWorkspace'
const SIDEBAR_CRON_OPEN_STORAGE_KEY = 'hermes.desktop.sidebarCronOpen'
const SIDEBAR_MESSAGING_OPEN_STORAGE_KEY = 'hermes.desktop.sidebarMessagingOpen'
const SIDEBAR_SESSION_ORDER_STORAGE_KEY = 'hermes.desktop.sessionOrder'
const SIDEBAR_SESSION_ORDER_MANUAL_STORAGE_KEY = 'hermes.desktop.sessionOrder.manual'
const SIDEBAR_WORKSPACE_ORDER_STORAGE_KEY = 'hermes.desktop.workspaceOrder'
const SIDEBAR_WORKSPACE_PARENT_ORDER_STORAGE_KEY = 'hermes.desktop.workspaceParentOrder'
const SIDEBAR_PROJECT_ORDER_STORAGE_KEY = 'hermes.desktop.projectOrder'
const SIDEBAR_WORKSPACE_COLLAPSED_STORAGE_KEY = 'hermes.desktop.workspaceCollapsed'
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

const $legacyPinnedSessionIds = atom(readLegacyPinnedSessionIds())

export const $pinnedSessionIds: ReadableAtom<string[]> = computed(
  [$sessions, $legacyPinnedSessionIds],
  (sessions, legacyPinnedSessionIds) => {
    const serverCapable = sessions.some(session => 'pinned' in session)

    if (!serverCapable) {
      return legacyPinnedSessionIds
    }

    const ids: string[] = []

    for (const session of sessions) {
      if (!session.pinned) {
        continue
      }

      const id = pinIdForSession(session)

      if (!ids.includes(id)) {
        ids.push(id)
      }
    }

    return ids
  }
)
export const $sidebarSessionOrderIds = persistentAtom(
  SIDEBAR_SESSION_ORDER_STORAGE_KEY,
  [] as string[],
  Codecs.stringArray
)
export const $sidebarSessionOrderManual = persistentAtom(SIDEBAR_SESSION_ORDER_MANUAL_STORAGE_KEY, false, Codecs.bool)
// Manual drag-order of PINNED rows, kept per-device (local) and layered over the
// server-synced pinned SET ($pinnedSessionIds). Pin membership syncs across
// machines; the visual order is a local preference — reordering on one device
// must not reshuffle another. Keyed by durable (lineage-root) pin ids so the
// order survives a session's compression id-change. Empty = default order.
const SIDEBAR_PINNED_ORDER_STORAGE_KEY = 'hermes.desktop.pinnedOrder'
export const $sidebarPinnedOrderIds = persistentAtom(
  SIDEBAR_PINNED_ORDER_STORAGE_KEY,
  [] as string[],
  Codecs.stringArray
)

export function setSidebarPinnedOrderIds(ids: string[]) {
  if (!arraysEqual($sidebarPinnedOrderIds.get(), ids)) {
    $sidebarPinnedOrderIds.set(ids)
  }
}

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
// Repo/worktree nodes that the user has explicitly COLLAPSED. Absent = open, so
// a project's folders auto-open when you enter it (and persist your collapses
// across reloads). Keyed by stable node id (repo root / worktree path).
export const $sidebarWorkspaceCollapsedIds = persistentAtom(
  SIDEBAR_WORKSPACE_COLLAPSED_STORAGE_KEY,
  [] as string[],
  Codecs.stringArray
)
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
// Set by the PaneShell hover-reveal overlay while the sidebar is collapsed; kept
// true the whole time it's a floating overlay (not just while shown) so the
// consumer mounts contents off-screen, ready to slide. ChatSidebar mounts its
// rows on `sidebarOpen || this`.
export const $sidebarOverlayMounted = atom(false)
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

// Toggle a repo/worktree node's persisted collapse state (absent = open).
export function toggleWorkspaceNodeCollapsed(id: string): void {
  const current = $sidebarWorkspaceCollapsedIds.get()

  $sidebarWorkspaceCollapsedIds.set(current.includes(id) ? current.filter(nodeId => nodeId !== id) : [...current, id])
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

export function setSidebarProjectOrderIds(ids: string[]) {
  if (!arraysEqual($sidebarProjectOrderIds.get(), ids)) {
    $sidebarProjectOrderIds.set(ids)
  }
}

export function setSidebarResizing(resizing: boolean) {
  $isSidebarResizing.set(resizing)
}

export function readLegacyPinnedSessionIds(): string[] {
  const raw = readKey(SIDEBAR_PINNED_STORAGE_KEY)

  if (!raw) {
    return []
  }

  try {
    return Codecs.stringArray.decode(raw)
  } catch {
    return []
  }
}

function clearLegacyPinnedSessionIds() {
  writeKey(SIDEBAR_PINNED_STORAGE_KEY, null)
  $legacyPinnedSessionIds.set([])
}

export async function migrateLegacyPinnedSessions(
  sessions: Array<{ id: string; _lineage_root_id?: null | string; pinned?: boolean }>,
  pushPinned: (sessionId: string) => Promise<void>,
  log: Pick<Console, 'info'> = console
): Promise<number> {
  const legacyIds = readLegacyPinnedSessionIds()

  if (legacyIds.length === 0) {
    return 0
  }

  const serverPinnedIds = new Set(sessions.filter(session => session.pinned).map(pinIdForSession))
  const pushedIds = legacyIds.filter((id, index) => legacyIds.indexOf(id) === index && !serverPinnedIds.has(id))

  await Promise.all(pushedIds.map(id => pushPinned(id)))
  clearLegacyPinnedSessionIds()
  log.info('server-side pinned sessions migration push count', {
    pushed: pushedIds.length,
    total: legacyIds.length
  })

  return pushedIds.length
}

function sessionMatchesPinId(session: { id: string; _lineage_root_id?: null | string }, pinId: string): boolean {
  return session.id === pinId || session._lineage_root_id === pinId || pinIdForSession(session) === pinId
}

function pinIdForSession(session: { id: string; _lineage_root_id?: null | string }): string {
  return session._lineage_root_id || session.id
}

function markSessionPinned(pinId: string, pinned: boolean) {
  setSessions(prev =>
    prev.map(session => (sessionMatchesPinId(session, pinId) ? { ...session, pinned } : session))
  )
}

export async function setSessionPinned(sessionId: string, pinned: boolean): Promise<void> {
  const session = $sessions.get().find(s => sessionMatchesPinId(s, sessionId))
  const pinId = session ? pinIdForSession(session) : sessionId
  const previous = $sessions.get()

  markSessionPinned(pinId, pinned)

  try {
    const selectedStoredSessionId = $selectedStoredSessionId.get()

    const runtimeId =
      selectedStoredSessionId && (selectedStoredSessionId === sessionId || selectedStoredSessionId === pinId)
        ? $activeSessionId.get()
        : null

    const gateway = runtimeId ? activeGateway() : null

    if (gateway && runtimeId) {
      try {
        await gateway.request('session.pin', { session_id: runtimeId, pinned })
        broadcastSessionsChanged()

        return
      } catch (err) {
        console.warn('session.pin RPC failed; falling back to REST', err)
      }
    }

    await patchSessionPinned(pinId, pinned, session?.profile)
    broadcastSessionsChanged()
  } catch (err) {
    setSessions(previous)
    throw err
  }
}

export function pinSession(sessionId: string): Promise<void> {
  return setSessionPinned(sessionId, true)
}

export function unpinSession(sessionId: string): Promise<void> {
  return setSessionPinned(sessionId, false)
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
