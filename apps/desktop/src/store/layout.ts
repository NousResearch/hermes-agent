import { atom, computed, type ReadableAtom, type WritableAtom } from 'nanostores'

import { SIDEBAR_COLLAPSE_MEDIA_QUERY } from '@/app/layout-constants'
import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { isPaneVisible, revealTreePane } from '@/components/pane-shell/tree/store'
import { matchesQuery } from '@/hooks/use-media-query'
import { connectionScopedAtom } from '@/lib/connection-scoped'
import { type Codec, Codecs, persistentAtom } from '@/lib/persisted'
import { arraysEqual, insertUniqueId, readKey } from '@/lib/storage'

import { $paneStates, ensurePaneRegistered, setPaneOpen, setPaneWidthOverride, togglePane } from './panes'
import { $showAllProfiles, setShowAllProfiles } from './profile'
import type { PullRequestBucket } from './pull-requests'
import type { SessionStatusBucket } from './session-dot-state'

export const SIDEBAR_DEFAULT_WIDTH = 237
export const SIDEBAR_MAX_WIDTH = 360
// Open at the same width as the sessions sidebar so the two rails match, but
// allow shrinking well below that (~30% under the old 14rem floor) for users who
// want a narrow tree.
export const FILE_BROWSER_DEFAULT_WIDTH = `${SIDEBAR_DEFAULT_WIDTH}px`
export const FILE_BROWSER_MIN_WIDTH = '10rem'
export const FILE_BROWSER_MAX_WIDTH = '20rem'

export const SIDEBAR_SESSIONS_PAGE_SIZE = 50
// How deep the list reaches once a filter is on. A filter that only searches
// the loaded page answers the wrong question — "6 merged PRs" really meant "6
// among the last 50 rows" — so narrowing the view widens the window it reads.
export const SIDEBAR_FILTERED_PAGE_SIZE = 300

const SIDEBAR_PINNED_STORAGE_KEY = 'hermes.desktop.pinnedSessions'
const SIDEBAR_AGENTS_GROUPED_STORAGE_KEY = 'hermes.desktop.agentsGroupedByWorkspace'
const SIDEBAR_CRON_OPEN_STORAGE_KEY = 'hermes.desktop.sidebarCronOpen'
const SIDEBAR_MESSAGING_OPEN_STORAGE_KEY = 'hermes.desktop.sidebarMessagingOpen'
const SIDEBAR_SESSION_ORDER_STORAGE_KEY = 'hermes.desktop.sessionOrder'
const SIDEBAR_SESSION_ORDER_MANUAL_STORAGE_KEY = 'hermes.desktop.sessionOrder.manual'
const SIDEBAR_GROUPING_STORAGE_KEY = 'hermes.desktop.sidebarGrouping'
const SIDEBAR_ALL_PROFILES_GROUPING_STORAGE_KEY = 'hermes.desktop.sidebarGrouping.allProfiles'
const SIDEBAR_ALL_PROFILES_AGENTS_GROUPED_STORAGE_KEY = 'hermes.desktop.sidebarAgentsGrouped.allProfiles'
const SIDEBAR_SORT_KEY_STORAGE_KEY = 'hermes.desktop.sidebarSortKey'
const SIDEBAR_ROW_META_STORAGE_KEY = 'hermes.desktop.sidebarRowMeta'
const SIDEBAR_CARD_ROWS_STORAGE_KEY = 'hermes.desktop.sidebarCardRows'
const SIDEBAR_STATUS_FILTER_STORAGE_KEY = 'hermes.desktop.sidebarStatusFilter'
const SIDEBAR_SHOW_ARCHIVED_STORAGE_KEY = 'hermes.desktop.sidebarShowArchived'
const SIDEBAR_PROJECT_FILTER_STORAGE_KEY = 'hermes.desktop.sidebarProjectFilter'
const SIDEBAR_PROFILE_FILTER_STORAGE_KEY = 'hermes.desktop.sidebarProfileFilter'
const SIDEBAR_PR_FILTER_STORAGE_KEY = 'hermes.desktop.sidebarPrFilter'
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
/** The file tree's id in the LAYOUT TREE — distinct from the pane-state id
 *  above, which keys its open/width record. Toggles need both. */
export const FILES_PANE_ID = 'files'

/** Every rail tab is a preview of something, namespaced by what backs it: a
 *  path on disk, a live URL, or an id into the in-memory artifact registry. */
export type RightRailTabId = `artifact:${string}` | `file:${string}` | `url:${string}`

ensurePaneRegistered(CHAT_SIDEBAR_PANE_ID, { open: true })
ensurePaneRegistered(FILE_BROWSER_PANE_ID, { open: false })

export const $sidebarOpen: ReadableAtom<boolean> = computed(
  $paneStates,
  states => states[CHAT_SIDEBAR_PANE_ID]?.open ?? true
)

export const $fileBrowserOpen: ReadableAtom<boolean> = computed(
  $paneStates,
  states => states[FILE_BROWSER_PANE_ID]?.open ?? false
)

// Persisted so a relaunch reopens the same rail tab. Null when the rail has no
// tabs; a restored id with no matching tab is reconciled in the preview store.
export const $rightRailActiveTabId = persistentAtom<RightRailTabId | null>(RIGHT_RAIL_ACTIVE_TAB_STORAGE_KEY, null, {
  decode: raw => (raw ? (raw as RightRailTabId) : null),
  encode: tabId => tabId ?? ''
})

export const $sidebarWidth: ReadableAtom<number> = computed($paneStates, states => {
  const override = states[CHAT_SIDEBAR_PANE_ID]?.widthOverride

  return typeof override === 'number' ? override : SIDEBAR_DEFAULT_WIDTH
})

// Pins and the manual session order are CONNECTION-scoped, not global: they
// are lists of session ids owned by one gateway's state.db, and multiple
// windows in this app can be connected to different gateways while sharing
// one localStorage area. A global key here is how one gateway's pins bleed
// into another window's sidebar (#77318). The local connection keeps the
// bare legacy key; remote connections get their own namespaced keys.
export const $pinnedSessionIds = connectionScopedAtom(SIDEBAR_PINNED_STORAGE_KEY, [] as string[], Codecs.stringArray)
export const $sidebarSessionOrderIds = connectionScopedAtom(
  SIDEBAR_SESSION_ORDER_STORAGE_KEY,
  [] as string[],
  Codecs.stringArray
)
export const $sidebarSessionOrderManual = connectionScopedAtom(
  SIDEBAR_SESSION_ORDER_MANUAL_STORAGE_KEY,
  false,
  Codecs.bool
)
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
// The Project-grouping flag, per scope like the grouping atoms below it: one
// global bool here meant picking Project inside a workspace also flipped the
// all-profiles view into the project tree (and leaving it there wiped the
// workspace's choice) — the "sidebar forgets my grouping every time I switch
// workspaces" bug. The flat key keeps its historical name so an existing
// choice survives the update.
const $sidebarFlatAgentsGrouped = persistentAtom(SIDEBAR_AGENTS_GROUPED_STORAGE_KEY, false, Codecs.bool)

const $sidebarAllProfilesAgentsGrouped = persistentAtom(
  SIDEBAR_ALL_PROFILES_AGENTS_GROUPED_STORAGE_KEY,
  false,
  Codecs.bool
)

/** Whether the CURRENT scope shows the project tree (reads the scope's own
 *  flag, so each workspace and the all-profiles view remember it separately). */
export const $sidebarAgentsGrouped: ReadableAtom<boolean> = computed(
  [$showAllProfiles, $sidebarFlatAgentsGrouped, $sidebarAllProfilesAgentsGrouped],
  (showAll, flat, allProfiles) => (showAll ? allProfiles : flat)
)

/** How the recents list is divided. `date` is the sidebar's long-standing
 *  default (Today / Yesterday / Last week dividers). `profile` only means
 *  anything while the sidebar is showing every profile at once. */
export type SidebarGrouping = 'date' | 'profile' | 'project' | 'status'
/** What ranks rows within whatever grouping is active. */
export type SidebarOrdering = 'cost' | 'created' | 'manual' | 'status' | 'tokens' | 'updated'
/** The sort keys the menu offers; `manual` is entered by dragging, not picked. */
export type SidebarSortKey = Exclude<SidebarOrdering, 'manual'>
/** Optional per-row metadata the user can switch on. `preview` is card-only:
 *  the one-line row has nowhere to put a second line. */
export type SidebarRowMeta = 'cost' | 'pr' | 'preview' | 'profile' | 'tokens' | 'updated'

function oneOf<T extends string>(values: readonly T[], fallback: T): Codec<T> {
  return {
    decode: raw => (values.includes(raw as T) ? (raw as T) : fallback),
    encode: value => value
  }
}

function listOf<T extends string>(values: readonly T[]): Codec<T[]> {
  return {
    decode: raw => Codecs.stringArray.decode(raw).filter((item): item is T => values.includes(item as T)),
    encode: value => Codecs.stringArray.encode(value)
  }
}

const ROW_META: readonly SidebarRowMeta[] = ['cost', 'pr', 'preview', 'profile', 'tokens', 'updated']
const STATUS_FILTERS: readonly SessionStatusBucket[] = ['needs-input', 'working', 'unread', 'draft', 'idle']
const PR_FILTERS: readonly PullRequestBucket[] = ['open', 'draft', 'merged', 'closed', 'none']
export const SIDEBAR_SORT_KEYS: readonly SidebarSortKey[] = ['updated', 'created', 'status', 'tokens', 'cost']

// `project` deliberately does NOT live here. Entering a project from ⌘K, the
// projects store, or a repo scan flips $sidebarAgentsGrouped directly, so that
// atom stays the single authority for the project view and this one only holds
// the grouping to fall back to when the user leaves it.
const $sidebarFlatGrouping = persistentAtom<SidebarGrouping>(
  SIDEBAR_GROUPING_STORAGE_KEY,
  'date',
  oneOf(['date', 'status'], 'date')
)

// All-profiles keeps its own grouping: `profile` only means anything there, and
// a shared atom would either drag it into a scope where it means nothing or
// reset the choice every time the user flips the rail. Both scopes still ship
// by day — grouping by owner is something you go and pick.
const $sidebarAllProfilesGrouping = persistentAtom<SidebarGrouping>(
  SIDEBAR_ALL_PROFILES_GROUPING_STORAGE_KEY,
  'date',
  oneOf(['date', 'profile', 'status'], 'date')
)

// The sidebar as it ships. Declared once so the atoms below, "Reset to
// defaults" and the "has this view been customized?" check can't drift apart —
// they used to inline the same literals in three places.
const SIDEBAR_DEFAULT_GROUPING: SidebarGrouping = 'date'
const SIDEBAR_DEFAULT_ORDERING: SidebarOrdering = 'updated'
const SIDEBAR_DEFAULT_ROW_META: SidebarRowMeta[] = ['preview', 'updated']

const $sidebarSortKey = persistentAtom<SidebarSortKey>(
  SIDEBAR_SORT_KEY_STORAGE_KEY,
  'updated',
  oneOf(SIDEBAR_SORT_KEYS, 'updated')
)

export const $sidebarRowMeta = persistentAtom<SidebarRowMeta[]>(
  SIDEBAR_ROW_META_STORAGE_KEY,
  SIDEBAR_DEFAULT_ROW_META,
  listOf(ROW_META)
)

/** Inbox style: render the flat list's session rows as three-line cards
 *  (project · age / title / model · size) instead of the one-line row. A
 *  RENDER variant, deliberately not a grouping — it composes with whichever
 *  grouping is active. Off by default; dense tree surfaces never use it. */
export const $sidebarCardRows = persistentAtom(SIDEBAR_CARD_ROWS_STORAGE_KEY, false, Codecs.bool)

export function setSidebarCardRows(on: boolean) {
  $sidebarCardRows.set(on)
}

/** Order-insensitive: the menu appends in click order, so ['tokens','updated']
 *  and ['updated','tokens'] are the same view. */
function sameRowMeta(a: SidebarRowMeta[], b: SidebarRowMeta[]): boolean {
  return a.length === b.length && a.every(id => b.includes(id))
}

// Archived sessions are a separate backend query (`archived: 'only'`), so this
// flag both filters the list and drives the fetch.
export const $sidebarShowArchived = persistentAtom(SIDEBAR_SHOW_ARCHIVED_STORAGE_KEY, false, Codecs.bool)

export const $sidebarStatusFilter = persistentAtom<SessionStatusBucket[]>(
  SIDEBAR_STATUS_FILTER_STORAGE_KEY,
  [],
  listOf(STATUS_FILTERS)
)

// Project ids as `liveSessionProjectId` reports them: an explicit project's id,
// or a repo root path for an auto-promoted one.
export const $sidebarProjectFilter = persistentAtom(
  SIDEBAR_PROJECT_FILTER_STORAGE_KEY,
  [] as string[],
  Codecs.stringArray
)

// Profile names, as `normalizeProfileKey` reports them. Only bites while the
// sidebar is showing every profile — scoped to one, the scope is the filter.
export const $sidebarProfileFilter = persistentAtom(
  SIDEBAR_PROFILE_FILTER_STORAGE_KEY,
  [] as string[],
  Codecs.stringArray
)

// Whether a session's branch has a PR, and in what state. Fetched per repo via
// `gh` (see store/pull-requests), so this is empty on backends without a local
// checkout — the menu hides the submenu rather than offering a dead filter.
export const $sidebarPrFilter = persistentAtom<PullRequestBucket[]>(
  SIDEBAR_PR_FILTER_STORAGE_KEY,
  [],
  listOf(PR_FILTERS)
)

export const $sidebarGrouping: ReadableAtom<SidebarGrouping> = computed(
  [$sidebarAgentsGrouped, $sidebarFlatGrouping, $sidebarAllProfilesGrouping, $showAllProfiles],
  (grouped, flat, allProfiles, showAll) => (grouped ? 'project' : showAll ? allProfiles : flat)
)

// A hand-dragged order outranks any sort key — dragging IS how you pick manual,
// so the menu reflects that rather than offering a fourth way to say it.
export const $sidebarOrdering: ReadableAtom<SidebarOrdering> = computed(
  [$sidebarSessionOrderManual, $sidebarSortKey],
  (manual, key) => (manual ? 'manual' : key)
)

export const $sidebarFiltersActive: ReadableAtom<boolean> = computed(
  [$sidebarStatusFilter, $sidebarProjectFilter, $sidebarProfileFilter, $sidebarPrFilter, $sidebarShowArchived],
  (statuses, projects, profiles, prs, archived) =>
    statuses.length > 0 || projects.length > 0 || profiles.length > 0 || prs.length > 0 || archived
)

/** Anything at all moved off the shipped view — what makes a reset worth
 *  offering. Broader than `$sidebarFiltersActive`, which only knows about what
 *  hides rows, not about how they're grouped, sorted or labelled. */
export const $sidebarViewCustomized: ReadableAtom<boolean> = computed(
  [$sidebarGrouping, $sidebarOrdering, $sidebarRowMeta, $sidebarCardRows, $sidebarFiltersActive],
  (grouping, ordering, rowMeta, cardRows, filtersActive) =>
    grouping !== SIDEBAR_DEFAULT_GROUPING ||
    ordering !== SIDEBAR_DEFAULT_ORDERING ||
    !sameRowMeta(rowMeta, SIDEBAR_DEFAULT_ROW_META) ||
    cardRows ||
    filtersActive
)

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

// Fold a whole level shut (or open) in one write — the sidebar's "Collapse all"
// over the project rows. Their lanes keep their own state underneath, so
// re-opening a project shows it as the user left it.
export function setWorkspaceNodesOpen(ids: readonly string[], open: boolean): void {
  $sidebarWorkspaceNodeOpen.set({
    ...$sidebarWorkspaceNodeOpen.get(),
    ...Object.fromEntries(ids.map(id => [id, open]))
  })
}

// Dismiss ("delete") an auto-derived project from the overview.
export function dismissAutoProject(id: string): void {
  const current = $dismissedAutoProjectIds.get()

  if (!current.includes(id)) {
    $dismissedAutoProjectIds.set([...current, id])
  }
}

// Auto projects dismissed from the overview stay out of every surface that
// lists projects (sidebar + ⌘K). Explicit rows never match.
export function filterVisibleProjects<T extends { id: string; isAuto?: boolean }>(
  projects: readonly T[],
  dismissedIds: readonly string[] = $dismissedAutoProjectIds.get()
): T[] {
  if (!dismissedIds.length) {
    return projects as T[]
  }

  const dismissed = new Set(dismissedIds)

  return projects.filter(project => !(project.isAuto && dismissed.has(project.id)))
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
  if (revealNarrowPane(FILE_BROWSER_PANE_ID, 'toggle')) {
    return
  }

  // Ask the TREE, not the pane's boolean. `$fileBrowserOpen` stays true while
  // the tree pane sits behind a sibling tab in the shared right column (the
  // preview rail, the diff) or inside a minimized zone, so ⌘J spent its press
  // re-asserting a value it already held and read as a dead key. Only fold the
  // side when the tree is genuinely the thing on screen; otherwise bring it
  // forward through the reveal path, which fronts and un-minimizes.
  if (!isPaneVisible(FILES_PANE_ID) && $fileBrowserOpen.get()) {
    revealTreePane(FILES_PANE_ID)

    return
  }

  togglePane(FILE_BROWSER_PANE_ID)
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

export function selectRightRailTab(id: RightRailTabId | null) {
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
  // Write the flag the current scope reads — see $sidebarAgentsGrouped.
  ;($showAllProfiles.get() ? $sidebarAllProfilesAgentsGrouped : $sidebarFlatAgentsGrouped).set(grouped)
}

export function setSidebarGrouping(grouping: SidebarGrouping) {
  // Grouping by owner is a request to see every owner, so it turns the
  // all-profiles view on rather than drawing one group around one profile —
  // and every write that follows must target that scope, not the one we were
  // in when the click landed. (The flat scope's atom can't hold 'profile'.)
  if (grouping === 'profile') {
    setShowAllProfiles(true)
    $sidebarAllProfilesAgentsGrouped.set(false)
    $sidebarAllProfilesGrouping.set(grouping)

    return
  }

  setSidebarAgentsGrouped(grouping === 'project')

  if (grouping === 'project') {
    return
  }

  if ($showAllProfiles.get()) {
    $sidebarAllProfilesGrouping.set(grouping)

    return
  }

  $sidebarFlatGrouping.set(grouping)
}

export function setSidebarOrdering(ordering: SidebarOrdering) {
  if (ordering === 'manual') {
    setSidebarSessionOrderManual(true)

    return
  }

  // Picking a sort key is the only way back out of a hand-dragged order, so it
  // has to drop the saved sequence as well as the flag.
  setSidebarSessionOrderManual(false)
  setSidebarSessionOrderIds([])
  $sidebarSortKey.set(ordering)
}

export function setSidebarShowArchived(show: boolean) {
  $sidebarShowArchived.set(show)
}

function toggleIn<T extends string>($atom: WritableAtom<T[]>, value: T) {
  const current = $atom.get()

  $atom.set(current.includes(value) ? current.filter(item => item !== value) : [...current, value])
}

export function toggleSidebarRowMeta(meta: SidebarRowMeta) {
  toggleIn($sidebarRowMeta, meta)
}

export function toggleSidebarStatusFilter(status: SessionStatusBucket) {
  toggleIn($sidebarStatusFilter, status)
}

export function toggleSidebarProjectFilter(projectId: string) {
  toggleIn($sidebarProjectFilter, projectId)
}

export function toggleSidebarProfileFilter(profile: string) {
  toggleIn($sidebarProfileFilter, profile)
}

export function toggleSidebarPrFilter(bucket: PullRequestBucket) {
  toggleIn($sidebarPrFilter, bucket)
}

function clearSidebarFilters() {
  $sidebarStatusFilter.set([])
  $sidebarProjectFilter.set([])
  $sidebarProfileFilter.set([])
  $sidebarPrFilter.set([])
  $sidebarShowArchived.set(false)
}

/** Every knob the filter menu owns, back to the sidebar as it ships. Ordering
 *  goes through its setter so a hand-dragged sequence is dropped along with it. */
export function resetSidebarView() {
  setSidebarGrouping(SIDEBAR_DEFAULT_GROUPING)
  // Both scopes, not just the one on screen: each keeps its own grouping (and
  // its own Project flag), so a reset that left the other customized would
  // hand it back on the next flip.
  $sidebarFlatGrouping.set(SIDEBAR_DEFAULT_GROUPING)
  $sidebarAllProfilesGrouping.set(SIDEBAR_DEFAULT_GROUPING)
  $sidebarFlatAgentsGrouped.set(false)
  $sidebarAllProfilesAgentsGrouped.set(false)
  setSidebarOrdering(SIDEBAR_DEFAULT_ORDERING)
  $sidebarRowMeta.set(SIDEBAR_DEFAULT_ROW_META)
  $sidebarCardRows.set(false)
  clearSidebarFilters()
}

/** Whether anything on screen needs PR data — the gate on shelling out to
 *  `gh`. Nobody pays for a network call per repo to render a view that would
 *  not show a PR anywhere. */
export const $sidebarPrDataWanted: ReadableAtom<boolean> = computed(
  [$sidebarRowMeta, $sidebarPrFilter],
  (rowMeta, prFilter) => rowMeta.includes('pr') || prFilter.length > 0
)

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

// Apply a new pinned order from a drag. The dragged list only holds the pins
// that currently RESOLVE to a loaded row, so this is a permutation of a subset:
// re-slot the ids it names into the positions they already occupied, leaving
// any pin it doesn't mention (row not loaded yet) exactly where it was.
// Requiring both lists to be the same length instead let one unresolved pin
// silently discard the whole reorder.
export function setPinnedSessionOrder(ids: string[]) {
  const prev = $pinnedSessionIds.get()
  const pinned = new Set(prev)
  const moving = ids.filter(id => pinned.has(id))

  if (!moving.length) {
    return
  }

  const movingSet = new Set(moving)
  const next = [...prev]
  let cursor = 0

  prev.forEach((id, index) => {
    if (movingSet.has(id)) {
      next[index] = moving[cursor++]
    }
  })

  setOrderIds($pinnedSessionIds, next)
}

export function bumpSessionsLimit(step: number = SIDEBAR_SESSIONS_PAGE_SIZE) {
  const safeStep = Math.max(1, Math.floor(step))
  $sessionsLimit.set($sessionsLimit.get() + safeStep)
}

/** Raise the window to at least `floor`, never shrinking it. Returns true when
 *  it moved, so the caller knows a refetch is worth it. */
export function raiseSessionsLimit(floor: number): boolean {
  if ($sessionsLimit.get() >= floor) {
    return false
  }

  $sessionsLimit.set(floor)

  return true
}

export function resetSessionsLimit() {
  if ($sessionsLimit.get() !== SIDEBAR_SESSIONS_PAGE_SIZE) {
    $sessionsLimit.set(SIDEBAR_SESSIONS_PAGE_SIZE)
  }
}
