import { KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core'
import { sortableKeyboardCoordinates } from '@dnd-kit/sortable'
import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation } from 'react-router'

import { PlatformAvatar } from '@/app/messaging/platform-icon'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ContextMenu, ContextMenuContent, ContextMenuTrigger } from '@/components/ui/context-menu'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { KbdGroup } from '@/components/ui/kbd'
import { SearchField } from '@/components/ui/search-field'
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem
} from '@/components/ui/sidebar'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import { useContributions } from '@/contrib/react/use-contributions'
import { searchSessions, type SessionInfo, type SessionSearchResult } from '@/hermes'
import { useI18n } from '@/i18n'
import { comboTokens } from '@/lib/keybinds/combo'
import { resolveProfileColor } from '@/lib/profile-color'
import { sessionMatchesSearch } from '@/lib/session-search'
import { normalizeSessionSource, sessionSourceLabel } from '@/lib/session-source'
import { cn } from '@/lib/utils'
import { $cronJobs } from '@/store/cron'
import { $bindings } from '@/store/keybinds'
import {
  $dismissedAutoProjectIds,
  $panesFlipped,
  $pinnedSessionIds,
  $sidebarCardRows,
  $sidebarCronOpen,
  $sidebarFiltersActive,
  $sidebarGrouping,
  $sidebarMessagingOpenIds,
  $sidebarOrdering,
  $sidebarPinsOpen,
  $sidebarPrDataWanted,
  $sidebarPrFilter,
  $sidebarProfileFilter,
  $sidebarProjectFilter,
  $sidebarProjectOrderIds,
  $sidebarRecentsOpen,
  $sidebarSessionOrderIds,
  $sidebarSessionOrderManual,
  $sidebarShowArchived,
  $sidebarStatusFilter,
  $sidebarWorkspaceOrderIds,
  $sidebarWorkspaceParentOrderIds,
  filterVisibleProjects,
  pinSession,
  SESSION_SEARCH_FOCUS_EVENT,
  setPinnedSessionOrder,
  setSidebarCronOpen,
  setSidebarPinsOpen,
  setSidebarProjectOrderIds,
  setSidebarRecentsOpen,
  setSidebarSessionOrderIds,
  setSidebarSessionOrderManual,
  setSidebarWorkspaceOrderIds,
  setSidebarWorkspaceParentOrderIds,
  SIDEBAR_SESSIONS_PAGE_SIZE,
  toggleSidebarMessagingOpen,
  unpinSession
} from '@/store/layout'
import {
  $newChatProfile,
  $profileColors,
  $profiles,
  $profileScope,
  ALL_PROFILES,
  normalizeProfileKey
} from '@/store/profile'
import {
  $activeProjectId,
  $projects,
  $projectScope,
  $projectTree,
  $projectTreeLoading,
  $removedSessionIds,
  $reposScanning,
  ALL_PROJECTS,
  enterProject,
  exitProjectScope,
  fetchProjectSessions,
  openProjectCreate,
  refreshProjects,
  refreshProjectTree,
  refreshWorktrees,
  scanAndRecordRepos
} from '@/store/projects'
import {
  $prBranchBySession,
  $pullRequestsByBranch,
  pullRequestBucket,
  recoverSessionPullRequests,
  refreshPullRequests,
  sessionPrKey
} from '@/store/pull-requests'
import { openRouteTile } from '@/store/route-tiles'
import {
  $cronSessions,
  $currentCwd,
  $gatewayState,
  $messagingPlatformTotals,
  $messagingSessions,
  $messagingTruncated,
  $sessionProfilesTruncated,
  $sessions,
  $sessionsLoading,
  sessionPinId,
  setCurrentCwd
} from '@/store/session'
import { $sessionDotStateById, sessionStatusBucket } from '@/store/session-dot-state'
import { $focusedStoredSessionId, $workingSessionIds, type SplitDir } from '@/store/session-states'
import { $archivedSessions, loadArchivedSessions } from '@/store/sidebar-archive'
import { $sidebarSessionRankIds } from '@/store/sidebar-sort'

import {
  type AppView,
  ARTIFACTS_ROUTE,
  CRON_ROUTE,
  MESSAGING_ROUTE,
  SIDEBAR_NAV_AREA,
  type SidebarNavContribution,
  SKILLS_ROUTE
} from '../../routes'
import type { SidebarNavItem } from '../../types'

import { SidebarCronJobsSection } from './cron-jobs-section'
import { SidebarFilterMenu } from './filter-menu'
import { SidebarLoadMoreRow } from './load-more-row'
import { orderByIds, reconcileOrderIds, resolveManualSessionOrderIds, sameIds } from './order'
import { ProfileRail } from './profile-switcher'
import { ProjectDialog } from './project-dialog'
import {
  excludeProjectSessions,
  liveSessionProjectId,
  orderProjectsByIds,
  overlayLiveLanes,
  overlayLivePreviews,
  PROJECT_PREVIEW_COUNT,
  ProjectBackRow,
  ProjectMenu,
  projectTreeCwd,
  sessionRecency as sessionTime,
  type SidebarProjectTree,
  type SidebarSessionGroup,
  type SidebarWorkspaceTree,
  sortProjectsForOverview,
  StartWorkButton,
  useRepoWorktreeMap
} from './projects'
import { WorktreeDialog } from './projects/worktree-dialog'
import { SidebarBlankState, SidebarPinnedEmptyState, SidebarSessionSkeletons } from './section-states'
import { buildSessionByAnyId, resolvePinnedSessions } from './session-index'
import { SidebarSessionsSection, VIRTUALIZE_THRESHOLD } from './sessions-section'
import { CONTEXT_SPLIT_KIT, SplitSubmenu } from './split-submenu'

// Non-session groups (messaging platforms) stay compact: show a few rows up
// front, reveal more in larger steps on demand. Keeps a busy platform from
// dominating the sidebar before the user asks to see it.
const NON_SESSION_INITIAL_ROWS = 3
const NON_SESSION_LOAD_STEP = 10

// How long after connecting to warm the project tree for someone who isn't in
// the grouped view. Long enough that the flat list — the thing actually on
// screen — has the connection to itself first.
const PROJECT_TREE_WARM_MS = 2_000

const SIDEBAR_NAV: SidebarNavItem[] = [
  {
    id: 'new-session',
    label: '',
    icon: props => <Codicon name="robot" {...props} />,
    action: 'new-session',
    keybindActionId: 'session.new'
  },
  {
    id: 'skills',
    label: '',
    icon: props => <Codicon name="symbol-misc" {...props} />,
    route: SKILLS_ROUTE,
    keybindActionId: 'nav.skills'
  },
  {
    id: 'messaging',
    label: '',
    icon: props => <Codicon name="comment" {...props} />,
    route: MESSAGING_ROUTE,
    keybindActionId: 'nav.messaging'
  },
  {
    id: 'artifacts',
    label: '',
    icon: props => <Codicon name="files" {...props} />,
    route: ARTIFACTS_ROUTE,
    keybindActionId: 'nav.artifacts'
  },
  {
    id: 'cron',
    label: '',
    icon: props => <Codicon name="watch" {...props} />,
    route: CRON_ROUTE,
    keybindActionId: 'nav.cron'
  }
]

// Two modes via the `compact` height variant (styles.css):
//   tall    → each section is shrink-0, capped, its own scroller; Sessions is flex-1.
//   compact → COMPACT_FLAT drops the caps so the whole stack scrolls as one.
// Sections stay shrink-0 so none can be squeezed below its content and bleed onto
// the next — the flexbox `min-height: auto` overlap trap that caused the bug.
const COMPACT_FLAT = 'compact:max-h-none compact:overflow-visible'

// Vertical scroll only — never a horizontal bar from glow bleed, long titles,
// etc. The bar itself only shows while the pointer is in the list.
const SCROLL_Y = 'overflow-y-auto overflow-x-hidden overscroll-contain scrollbar-fade'

// The outer list reserves its bar's width whether or not one is showing, so
// filtering or collapsing a section doesn't reflow every row sideways. Only the
// outer one: nested scrollers would each reserve their own and stack the inset.
const SCROLL_GUTTER = '[scrollbar-gutter:stable]'

// A non-session group's scroll body: own scroller when tall, flattened when compact.
const GROUP_BODY = cn(SCROLL_Y, COMPACT_FLAT)

// Section-header action icons stay hidden until the whole header row is hovered
// (group/section lives on SidebarSectionHeader), mirroring the artifacts/file
// browser header affordances. focus-visible keeps them keyboard-reachable.
const HEADER_ACTION_BTN =
  'text-(--ui-text-tertiary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/section:opacity-100 focus-visible:opacity-100'

// The view toggle (overview group toggle / in-project back) is the one control
// that stays visible at all times — it's the stable navigation affordance, not
// a hover-revealed action.
const HEADER_NAV_BTN =
  'text-(--ui-text-tertiary) opacity-70 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground hover:opacity-100 focus-visible:opacity-100'

// FTS results cover sessions that aren't in the loaded page; synthesize a
// minimal SessionInfo so they render in the same row component (resume works
// by id; the snippet stands in for the preview).
function searchResultToSession(result: SessionSearchResult): SessionInfo {
  const ts = result.session_started ?? Date.now() / 1000

  return {
    archived: false,
    cwd: null,
    ended_at: null,
    id: result.session_id,
    _lineage_root_id: result.lineage_root ?? null,
    input_tokens: 0,
    is_active: false,
    last_active: ts,
    message_count: 0,
    model: result.model ?? null,
    output_tokens: 0,
    preview: result.snippet?.trim() || null,
    source: result.source ?? null,
    started_at: ts,
    title: null,
    tool_call_count: 0
  }
}

interface ChatSidebarProps extends React.ComponentProps<typeof Sidebar> {
  currentView: AppView
  onNavigate: (item: SidebarNavItem) => void
  onLoadMoreSessions: () => Promise<void> | void
  onLoadMoreMessaging?: (platform: string) => Promise<void> | void
  onResumeSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
  onArchiveSession: (sessionId: string) => void
  onBranchSession: (sessionId: string) => void
  onNewSessionInWorkspace: (path: null | string) => void
  /** Create a brand-new session and open it as a tile on `dir`. */
  onNewSessionSplit: (dir: SplitDir) => void
  onManageCronJob: (jobId: string) => void
  onTriggerCronJob: (jobId: string) => Promise<void>
}

export function ChatSidebar({
  currentView,
  onNavigate,
  onLoadMoreSessions,
  onLoadMoreMessaging,
  onResumeSession,
  onDeleteSession,
  onArchiveSession,
  onBranchSession,
  onNewSessionInWorkspace,
  onNewSessionSplit,
  onManageCronJob,
  onTriggerCronJob
}: ChatSidebarProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const { pathname } = useLocation()
  // Contributed nav rows (plugins pairing a page with a sidebar entry) render
  // below the built-ins with the same chrome; active = at their route.
  const navContributions = useContributions(SIDEBAR_NAV_AREA)

  const contributedNav = useMemo<SidebarNavItem[]>(
    () =>
      navContributions.flatMap(c => {
        const data = c.data as Partial<SidebarNavContribution> | undefined

        if (!data?.path?.startsWith('/') || !data.label) {
          return []
        }

        const codicon = data.codicon || 'plug'

        return [
          {
            id: c.id,
            label: data.label,
            icon: (props: { className?: string }) => <Codicon name={codicon} {...props} />,
            route: data.path
          }
        ]
      }),
    [navContributions]
  )

  const panesFlipped = useStore($panesFlipped)
  const grouping = useStore($sidebarGrouping)
  const ordering = useStore($sidebarOrdering)
  const statusFilter = useStore($sidebarStatusFilter)
  const projectFilter = useStore($sidebarProjectFilter)
  const profileFilter = useStore($sidebarProfileFilter)
  const prFilter = useStore($sidebarPrFilter)
  const prDataWanted = useStore($sidebarPrDataWanted)
  const prBranchOverrides = useStore($prBranchBySession)
  const pullRequests = useStore($pullRequestsByBranch)
  const filtersActive = useStore($sidebarFiltersActive)
  const showArchived = useStore($sidebarShowArchived)
  const cardRows = useStore($sidebarCardRows)
  const archivedSessions = useStore($archivedSessions)
  const dotStates = useStore($sessionDotStateById)
  // The active sort key as an id order. The flat list applies it within its
  // dividers; groups apply it to their own lanes.
  const sortOrderIds = useStore($sidebarSessionRankIds)
  const agentsGrouped = grouping === 'project'
  const pinnedSessionIds = useStore($pinnedSessionIds)
  const pinsOpen = useStore($sidebarPinsOpen)
  const agentsOpen = useStore($sidebarRecentsOpen)
  const cronOpen = useStore($sidebarCronOpen)
  // The sidebar highlight tracks the FOCUSED session — the interacted tile's
  // tab, else the main selection — so it stays 1:1 with whatever tab is active.
  const selectedSessionId = useStore($focusedStoredSessionId)
  const sessions = useStore($sessions)
  const cronSessions = useStore($cronSessions)
  const cronJobs = useStore($cronJobs)
  const messagingSessions = useStore($messagingSessions)
  const messagingPlatformTotals = useStore($messagingPlatformTotals)
  const messagingTruncated = useStore($messagingTruncated)
  const sessionsLoading = useStore($sessionsLoading)
  const sessionProfilesTruncated = useStore($sessionProfilesTruncated)
  const profiles = useStore($profiles)
  const profileColors = useStore($profileColors)
  const profileScope = useStore($profileScope)
  // Only surface the profile switcher when more than one profile exists, so
  // single-profile users see the unchanged sidebar.
  const multiProfile = profiles.length > 1
  // Gate ALL-profiles grouping on multiProfile too: if a user drops back to one
  // profile while scope is still ALL (persisted), the rail is hidden and they'd
  // otherwise be stuck in the grouped view with no way out.
  const showAllProfiles = multiProfile && profileScope === ALL_PROFILES
  const agentOrderIds = useStore($sidebarSessionOrderIds)
  const agentOrderManual = useStore($sidebarSessionOrderManual)
  const workspaceOrderIds = useStore($sidebarWorkspaceOrderIds)
  const workspaceParentOrderIds = useStore($sidebarWorkspaceParentOrderIds)
  const projectOrderIds = useStore($sidebarProjectOrderIds)
  const projects = useStore($projects)
  const projectTree = useStore($projectTree)
  const projectTreeLoading = useStore($projectTreeLoading)
  const removedSessionIds = useStore($removedSessionIds)
  const reposScanning = useStore($reposScanning)
  const activeProjectId = useStore($activeProjectId)
  const projectScope = useStore($projectScope)
  const currentCwd = useStore($currentCwd)
  const gatewayState = useStore($gatewayState)
  const dismissedAutoProjects = useStore($dismissedAutoProjectIds)
  const newSessionCombo = useStore($bindings)['session.new']?.[0]
  const newSessionKbd = newSessionCombo ? comboTokens(newSessionCombo) : []
  const [searchQuery, setSearchQuery] = useState('')
  const [serverMatches, setServerMatches] = useState<SessionSearchResult[]>([])
  const [searchPending, setSearchPending] = useState(false)
  const [newSessionKbdFlash, setNewSessionKbdFlash] = useState(false)
  const [messagingLoadMorePending, setMessagingLoadMorePending] = useState<Record<string, boolean>>({})
  const [recentsLoadMorePending, setRecentsLoadMorePending] = useState(false)
  const messagingOpenIds = useStore($sidebarMessagingOpenIds)
  // Per-platform count of rows currently revealed (starts at NON_SESSION_INITIAL_ROWS).
  const [messagingVisible, setMessagingVisible] = useState<Record<string, number>>({})
  const searchInputRef = useRef<HTMLInputElement>(null)
  const trimmedQuery = searchQuery.trim()

  // Hotkey (session.focusSearch) → focus the field once it's mounted.
  useEffect(() => {
    const onFocus = () => searchInputRef.current?.focus({ preventScroll: true })

    window.addEventListener(SESSION_SEARCH_FOCUS_EVENT, onFocus)

    return () => window.removeEventListener(SESSION_SEARCH_FOCUS_EVENT, onFocus)
  }, [])

  // Flash the ⌘N hint full-opacity (no transition) for the press, so hitting
  // the shortcut visibly pings its affordance in the sidebar.
  useEffect(() => {
    let timeout: ReturnType<typeof setTimeout> | undefined

    const onShortcut = () => {
      setNewSessionKbdFlash(true)
      clearTimeout(timeout)
      timeout = setTimeout(() => setNewSessionKbdFlash(false), 140)
    }

    window.addEventListener('hermes:new-session-shortcut', onShortcut)

    return () => {
      window.removeEventListener('hermes:new-session-shortcut', onShortcut)
      clearTimeout(timeout)
    }
  }, [])

  const activeSidebarSessionId = currentView === 'chat' ? selectedSessionId : null

  const dndSensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  )

  // Profile scope = the "workspace switcher" context. Concrete scope shows only
  // that profile's sessions (clean rows, no per-row tags); ALL fans every
  // profile in, grouped by profile below. Single-profile users land here with
  // scope === their only profile, so nothing is filtered out.
  // Archived rows are excluded from the sessions query, so Archived is a view of
  // its own set rather than a filter over this one — a flat list of archived
  // rows, no project tree, no date or status dividers.
  const scopedSessions = useMemo(() => {
    const pool = showArchived ? archivedSessions : sessions

    return showAllProfiles ? pool : pool.filter(s => normalizeProfileKey(s.profile) === profileScope)
  }, [sessions, archivedSessions, showArchived, showAllProfiles, profileScope])

  // One predicate for the status/project filters, so the flat list and the
  // project lanes narrow by the same rule. A project lane holds rows the loaded
  // page may not, so it has to be answerable per session rather than by
  // membership in the filtered set.
  const sessionMatchesFilters = useCallback(
    (session: SessionInfo) => {
      if (statusFilter.length && !statusFilter.includes(sessionStatusBucket(dotStates[session.id]))) {
        return false
      }

      // Narrowing to a few of the profiles on screen. Scoped to one profile the
      // list is already that profile's, so a stale selection can't blank it.
      if (showAllProfiles && profileFilter.length && !profileFilter.includes(normalizeProfileKey(session.profile))) {
        return false
      }

      if (prFilter.length) {
        const key = sessionPrKey(session)

        if (!prFilter.includes(pullRequestBucket(key ? pullRequests[key] : undefined))) {
          return false
        }
      }

      // Same membership the sidebar groups and colors by, so a filtered row
      // lands in the lane the user picked it from.
      return !projectFilter.length || projectFilter.includes(liveSessionProjectId(session, projects) ?? '')
    },
    [statusFilter, projectFilter, profileFilter, showAllProfiles, prFilter, pullRequests, projects, dotStates]
  )

  const filtersNarrow =
    statusFilter.length > 0 ||
    projectFilter.length > 0 ||
    prFilter.length > 0 ||
    (showAllProfiles && profileFilter.length > 0)

  const visibleSessions = useMemo(
    () => (filtersNarrow ? scopedSessions.filter(sessionMatchesFilters) : scopedSessions),
    [scopedSessions, filtersNarrow, sessionMatchesFilters]
  )

  // Recents by activity (last_active || started_at). User send stamps
  // last_active immediately. Ordering by status doesn't sort here — it re-slots
  // rows *inside* whatever dividers are on, via sortOrderIds below — so the
  // date buckets stay chronological either way.
  const sortedSessions = useMemo(
    () => [...visibleSessions].sort((a, b) => sessionTime(b) - sessionTime(a)),
    [visibleSessions]
  )

  // Index sessions by every id a pin might be stored under — recents, cron,
  // AND messaging, since all three can be pinned (see session-index.ts).
  const sessionByAnyId = useMemo(
    () => buildSessionByAnyId(visibleSessions, cronSessions, messagingSessions),
    [visibleSessions, cronSessions, messagingSessions]
  )

  // Local pin ids first (hand-picked order), then server-flagged pins the
  // local set doesn't know about — a backend `pinned=1` row must never be
  // invisible just because localStorage is cold or was clobbered (#85969).
  const pinnedSessions = useMemo(
    () =>
      resolvePinnedSessions(pinnedSessionIds, sessionByAnyId, [
        ...visibleSessions,
        ...cronSessions,
        ...messagingSessions
      ]),
    [pinnedSessionIds, sessionByAnyId, visibleSessions, cronSessions, messagingSessions]
  )

  // Every id a pin is reachable under: the raw stored ids, plus BOTH identities
  // of each session we resolved one to. A pin is stored on the durable lineage
  // root, but the lists that must filter it out are fed from three independent
  // fetches (recents, the messaging slice, the backend project tree) and each
  // can surface the same conversation under either its live tip or its root.
  // Comparing one identity against the other is how a pinned session ended up
  // rendered twice — once in Pinned, once in its project group.
  const pinnedIdentitySet = useMemo(() => {
    const ids = new Set(pinnedSessionIds)

    for (const session of pinnedSessions) {
      ids.add(session.id)

      if (session._lineage_root_id) {
        ids.add(session._lineage_root_id)
      }
    }

    return ids
  }, [pinnedSessionIds, pinnedSessions])

  // A pinned session belongs to the Pinned section and nowhere else, so every
  // other list filters it out. Match on either identity the row carries — a
  // backend snapshot can surface either side of a compression tip rotation.
  const isPinnedSession = useCallback(
    (session: SessionInfo) =>
      pinnedIdentitySet.has(session.id) ||
      (session._lineage_root_id != null && pinnedIdentitySet.has(session._lineage_root_id)),
    [pinnedIdentitySet]
  )

  // What the project tree drops: pins (they live in their own section) plus
  // anything the active filters exclude, so filtering works the same whether
  // you're looking at the flat list or the lanes.
  const isHiddenFromProjects = useCallback(
    (session: SessionInfo) => isPinnedSession(session) || (filtersNarrow && !sessionMatchesFilters(session)),
    [isPinnedSession, filtersNarrow, sessionMatchesFilters]
  )

  // Full-text search across *all* sessions (not just the loaded page) so 699
  // sessions stay findable. Debounced; loaded sessions are matched instantly
  // client-side and merged ahead of the server hits.
  useEffect(() => {
    if (!trimmedQuery) {
      setServerMatches([])
      setSearchPending(false)

      return
    }

    let cancelled = false

    setSearchPending(true)

    const id = window.setTimeout(() => {
      void searchSessions(trimmedQuery)
        .then(res => {
          if (!cancelled) {
            setServerMatches(res.results)
          }
        })
        .catch(() => undefined)
        .finally(() => {
          if (!cancelled) {
            setSearchPending(false)
          }
        })
    }, 200)

    return () => {
      cancelled = true
      window.clearTimeout(id)
    }
  }, [trimmedQuery])

  const searchResults = useMemo(() => {
    if (!trimmedQuery) {
      return []
    }

    const out = new Map<string, SessionInfo>()

    for (const s of sortedSessions) {
      if (sessionMatchesSearch(s, trimmedQuery)) {
        out.set(s.id, s)
      }
    }

    for (const match of serverMatches) {
      if (out.has(match.session_id)) {
        continue
      }

      const loaded = sessionByAnyId.get(match.session_id)
      out.set(match.session_id, loaded ?? searchResultToSession(match))
    }

    return [...out.values()]
  }, [trimmedQuery, sortedSessions, serverMatches, sessionByAnyId])

  const unpinnedAgentSessions = useMemo(
    () => sortedSessions.filter(s => !isPinnedSession(s)),
    [sortedSessions, isPinnedSession]
  )

  useEffect(() => {
    const next = resolveManualSessionOrderIds(
      unpinnedAgentSessions.map(s => s.id),
      agentOrderIds,
      agentOrderManual
    )

    if (!next.length && agentOrderManual) {
      setSidebarSessionOrderManual(false)
    }

    if (!next.length && agentOrderIds.length) {
      setSidebarSessionOrderIds([])

      return
    }

    if (next.length && !sameIds(next, agentOrderIds)) {
      setSidebarSessionOrderIds(next)
    }
  }, [agentOrderIds, agentOrderManual, unpinnedAgentSessions])

  // Recents render in recency order. The hand-picked order is layered on per
  // date group inside the section (orderRowsWithinGroups) rather than baked
  // into the list here, so a drag ranks a row among its own day's chats
  // instead of flattening the whole sidebar into an undated manual mode.
  const agentSessions = unpinnedAgentSessions

  // Recents are local-only: messaging-platform sessions are fetched as their
  // own slice ($messagingSessions) and rendered in self-managed per-platform
  // sections below, so there is no source-grouping magic to untangle here.
  //
  // Workspace grouping is a `project -> repo -> lane -> sessions` tree computed
  // authoritatively on the backend (projects.tree). Parents reorder via
  // workspaceParentOrderIds; worktrees within a parent via workspaceOrderIds.
  const worktreeGroupingActive = agentsGrouped && !showArchived
  const gatewayReady = gatewayState === 'open'

  // The backend project tree is a structural snapshot, NOT a per-message feed.
  // Refresh it on structural edges only — entering the grouped view, a profile
  // switch, gateway (re)connect — plus the once-per-run disk scan. Live session
  // changes between refreshes are reflected by the in-memory overlay
  // (overlayLiveLanes / overlayLivePreviews) off `$sessions`, so a turn
  // completing does NOT re-run the heavy list_sessions_rich scan. Project
  // mutations refresh the tree from their own store actions.
  useEffect(() => {
    if (!gatewayReady) {
      return
    }

    if (worktreeGroupingActive) {
      void refreshProjects()

      // The all-profiles tree is served off every profile's databases at once
      // and deliberately leaves discovery out — a repo with no sessions is the
      // same repo in every profile, so scanning here would multiply empty lanes
      // by the profile count and write the result into profiles the user isn't
      // driving.
      if (showAllProfiles) {
        void refreshProjectTree()

        return
      }

      // Paint the list from the fast tree fetch (explicit projects + repos from
      // existing sessions / the backend cache) FIRST, then kick off the heavy
      // home-dir git crawl so newly-discovered repos fold in afterward — instead
      // of the crawl blocking the first render.
      void refreshProjectTree().finally(() => void scanAndRecordRepos())

      return
    }

    // Flat view: warm the tree in the background anyway. Fetching it only on
    // the switch meant the first switch of every run paid for the whole round
    // trip behind a skeleton, and the menu's Project filter had nothing to
    // list until you'd visited the grouped view at least once.
    const warm = window.setTimeout(() => void refreshProjectTree(), PROJECT_TREE_WARM_MS)

    return () => window.clearTimeout(warm)
  }, [worktreeGroupingActive, showAllProfiles, profileScope, gatewayReady])

  // Sessions the branch join can't answer for get one look at their own
  // transcript — a `gh pr create` in there names the PR outright. Backfills
  // whatever is loaded, whether or not the badge is on: gating it on the badge
  // meant switching PR on showed a half-empty list until a second pass caught
  // up. One request per batch of never-scanned rows, and the scanned set makes
  // that batch empty from the second pass on, so this settles to nothing.
  useEffect(() => {
    if (!gatewayReady) {
      return
    }

    const warm = window.setTimeout(() => void recoverSessionPullRequests(scopedSessions), PROJECT_TREE_WARM_MS)

    return () => window.clearTimeout(warm)
  }, [gatewayReady, scopedSessions])

  // PR state is only fetched for someone who asked to see it — the badge or the
  // filter — and it asks about the branches on screen, so the answer can't be
  // crowded out by a busy repo's newer PRs.
  const prLookupsByRepo = useMemo(() => {
    if (!prDataWanted) {
      return {}
    }

    const byRepo: Record<string, string[]> = {}

    for (const session of scopedSessions) {
      // The row's own key, so a session bound to a branch (or a PR number) it
      // was stamped with asks about THAT, not the branch it started on.
      const [root, lookup] = sessionPrKey(session)?.split('\n') ?? []

      if (root && lookup && !byRepo[root]?.includes(lookup)) {
        byRepo[root] = [...(byRepo[root] ?? []), lookup]
      }
    }

    return byRepo
    // prBranchOverrides is what `sessionPrKey` reads through — a recovered PR
    // has to re-ask with the key it just learned.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prDataWanted, scopedSessions, prBranchOverrides])

  // A stable identity for "the same question as last time", so a re-render that
  // rebuilds the map doesn't re-ask GitHub.
  const prQueryKey = JSON.stringify(
    Object.entries(prLookupsByRepo)
      .map(([root, lookups]) => [root, [...lookups].sort()] as const)
      .sort(([a], [b]) => a.localeCompare(b))
  )

  useEffect(() => {
    if (prQueryKey === '[]') {
      return
    }

    const byRepo = Object.fromEntries(JSON.parse(prQueryKey) as [string, string[]][])

    void refreshPullRequests(byRepo)

    // A PR opens, merges or gets closed on github.com, not in here — so like
    // the project tree, re-pull when the window comes back. The store's own
    // staleness window keeps a flurry of focus events to one call per repo.
    const onActive = () => {
      if (document.visibilityState !== 'hidden') {
        void refreshPullRequests(byRepo)
      }
    }

    window.addEventListener('focus', onActive)
    document.addEventListener('visibilitychange', onActive)

    return () => {
      window.removeEventListener('focus', onActive)
      document.removeEventListener('visibilitychange', onActive)
    }
  }, [prQueryKey])

  // Out-of-band repo changes (a `git init` / `rm -rf` in another terminal) emit
  // no git events, so — like every git GUI — re-pull on window focus / tab
  // visibility instead of stranding the tree until a hard reload. The tree
  // fetch is cheap and runs every focus (picks up explicit create/delete +
  // session regrouping); the heavy disk crawl that surfaces brand-new repos is
  // throttled. Agent-driven changes already refresh via $workspaceChangeTick.
  useEffect(() => {
    if (!worktreeGroupingActive || !gatewayReady) {
      return
    }

    let lastScanAt = 0
    const SCAN_THROTTLE_MS = 30_000

    const onActive = () => {
      if (document.visibilityState === 'hidden') {
        return
      }

      void refreshProjects()
      void refreshProjectTree()

      // Discovery stays off while browsing every profile, for the reason the
      // first fetch leaves it out.
      if (showAllProfiles) {
        return
      }

      const now = Date.now()

      if (now - lastScanAt >= SCAN_THROTTLE_MS) {
        lastScanAt = now
        void scanAndRecordRepos(true)
      }
    }

    window.addEventListener('focus', onActive)
    document.addEventListener('visibilitychange', onActive)

    return () => {
      window.removeEventListener('focus', onActive)
      document.removeEventListener('visibilitychange', onActive)
    }
  }, [worktreeGroupingActive, showAllProfiles, gatewayReady])

  // Apply the persisted repo + worktree orders to a project's repo subtrees.
  const orderRepos = useCallback(
    (repos: SidebarWorkspaceTree[]): SidebarWorkspaceTree[] =>
      orderByIds(repos, parent => parent.id, workspaceParentOrderIds).map(parent => ({
        ...parent,
        groups: orderByIds(parent.groups, group => group.id, workspaceOrderIds)
      })),
    [workspaceParentOrderIds, workspaceOrderIds]
  )

  // ── Projects: the single top-level model (authoritative, from the backend) ──
  // `projects.tree` already unifies explicit projects + auto repos and folds
  // linked worktrees under their main repo. The desktop only layers local view
  // state on top: dismissed auto-projects, persisted repo/lane order, and the
  // overview sort. Membership is the backend tree's — never re-derived here.
  const projectModel = useMemo<SidebarProjectTree[]>(() => {
    const sorted = sortProjectsForOverview(
      filterVisibleProjects(projectTree, dismissedAutoProjects)
        // A filtered-out project drops its whole lane, header included — hiding
        // only its rows would leave a row of empty folders behind.
        .filter(project => !projectFilter.length || projectFilter.includes(project.id))
        .map(project =>
          excludeProjectSessions(
            {
              ...project,
              // Home is synthetic, so its name is ours to translate — every
              // other label is a repo basename or a name the user typed.
              label: project.isNoProject ? s.projects.home : project.label,
              repos: orderRepos(project.repos)
            },
            isHiddenFromProjects
          )
        ),
      activeProjectId
    )

    // Layer the user's manual drag-order on top of the deterministic sort. Empty
    // (default) returns `sorted` untouched; projects the user hasn't ordered yet
    // keep their sorted position rather than jumping the hand-picked list.
    return orderProjectsByIds(sorted, projectOrderIds)
  }, [
    projectTree,
    dismissedAutoProjects,
    orderRepos,
    activeProjectId,
    projectFilter,
    projectOrderIds,
    isHiddenFromProjects,
    s
  ])

  // The overview only renders in grouped mode; the model stays live regardless
  // so scoping is consistent across views.
  const agentProjectTree = worktreeGroupingActive ? projectModel : undefined

  // ── Project switcher (drill-in) ────────────────────────────────────────────
  // Grouped, single-profile view is a project switcher: ALL_PROJECTS shows the
  // overview (a list you click into); a concrete scope means you've "entered" a
  // project, so the Sessions list shows ONLY that project's worktrees/sessions.
  const projectsActive = Boolean(agentProjectTree?.length)

  // The overview node for the entered project (structure + counts, empty lanes).
  const overviewEnteredProject =
    projectsActive && projectScope !== ALL_PROJECTS
      ? agentProjectTree?.find(node => node.id === projectScope)
      : undefined

  const inProject = Boolean(overviewEnteredProject)
  const enteredProjectId = overviewEnteredProject?.id

  // Entering a project lazily hydrates its full lanes (repo -> lane -> sessions)
  // from the backend — same grouping/ids as the overview, just with rows.
  const [enteredProjectTree, setEnteredProjectTree] = useState<SidebarProjectTree | null>(null)

  useEffect(() => {
    if (!enteredProjectId || !gatewayReady) {
      setEnteredProjectTree(null)

      return
    }

    let cancelled = false

    void fetchProjectSessions(enteredProjectId).then(project => {
      if (!cancelled) {
        setEnteredProjectTree(project)
      }
    })

    return () => {
      cancelled = true
    }
    // `projectTree` in deps: re-hydrate after a tree refresh so the entered view
    // stays current with new/ended sessions.
  }, [enteredProjectId, gatewayReady, projectTree])

  // Prefer the hydrated tree; fall back to the overview node (empty lanes) while
  // the drill-in fetch is in flight, so the header/structure render immediately.
  const enteredProject = useMemo<SidebarProjectTree | undefined>(() => {
    if (!overviewEnteredProject) {
      return undefined
    }

    const hydrated =
      enteredProjectTree && enteredProjectTree.id === overviewEnteredProject.id
        ? enteredProjectTree
        : overviewEnteredProject

    // The live-session overlay (creates/evictions) is applied per-repo in
    // RepoFlatSection, AFTER the visual git-worktree lanes are merged in (so
    // out-of-tree worktrees can be placed). Here we just order the snapshot and
    // drop pinned rows — the hydrated lanes come straight from the backend, so
    // they haven't been through projectModel's filter.
    // The label comes from the overview node either way — that's the model's
    // presentation copy (Home is translated there), not the raw payload's.
    return excludeProjectSessions(
      { ...hydrated, label: overviewEnteredProject.label, repos: orderRepos(hydrated.repos) },
      isHiddenFromProjects
    )
  }, [overviewEnteredProject, enteredProjectTree, orderRepos, isHiddenFromProjects])

  // Overlay live `$sessions` onto the entered project so a just-created session
  // (which the backend snapshot hasn't folded in yet) counts as content and
  // renders immediately — same optimistic layer as the overview previews. The
  // backend now seeds each project folder as an (empty) repo, so the overlay
  // always has a lane to place a new in-project session into.
  const enteredProjectContent = useMemo(
    () => (enteredProject ? overlayLiveLanes(enteredProject, agentSessions, removedSessionIds) : undefined),
    [enteredProject, agentSessions, removedSessionIds]
  )

  const scopedRepoPaths = useMemo(
    () =>
      enteredProject ? enteredProject.repos.map(repo => repo.path).filter((path): path is string => Boolean(path)) : [],
    [enteredProject]
  )

  // git worktree list is a VISUAL-only enhancer (empty lanes); never membership.
  const inEnteredProject = Boolean(enteredProject && !showAllProfiles)
  const [scopedRepoWorktrees] = useRepoWorktreeMap(scopedRepoPaths, inEnteredProject)

  // Re-probe worktree lanes on out-of-band git changes the renderer can't see.
  // A turn can `git worktree add/remove` in the terminal (e.g. you ask Hermes to
  // "remove that worktree"), and the window never blurs during an in-app chat,
  // so nothing would otherwise re-run the visual probe. Re-sync when a working
  // session settles (its turn finished) or the window refocuses (an external
  // terminal may have changed things) — only while a project is entered, and
  // only the cheap per-repo `git worktree list`, never the heavy tree scan.
  //
  // Listened to rather than rendered from: a settling turn is a side effect,
  // and reading it with `useStore` repainted this whole component — every
  // section, every row — on each status edge, to run an effect that touches no
  // markup. The rows subscribe to their own status, so nothing above them needs
  // to re-render for one of them to change color.
  useEffect(() => {
    if (!inEnteredProject) {
      return
    }

    let previous = $workingSessionIds.get()

    return $workingSessionIds.listen(working => {
      // A session leaving the working set means its turn just completed.
      const aTurnSettled = previous.some(id => !working.includes(id))

      previous = working

      if (aTurnSettled) {
        refreshWorktrees()
      }
    })
  }, [inEnteredProject])

  useEffect(() => {
    if (!inEnteredProject) {
      return
    }

    const onFocus = () => refreshWorktrees()
    window.addEventListener('focus', onFocus)

    return () => window.removeEventListener('focus', onFocus)
  }, [inEnteredProject])

  const lastProjectCwdSyncRef = useRef<null | string>(null)

  const syncProjectCwd = useCallback(
    (project: SidebarProjectTree) => {
      const target = projectTreeCwd(project)

      if (target && target !== currentCwd) {
        setCurrentCwd(target)
      }
    },
    [currentCwd]
  )

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (!inProject || !enteredProject) {
      lastProjectCwdSyncRef.current = null

      return
    }

    if (lastProjectCwdSyncRef.current === enteredProject.id) {
      return
    }

    syncProjectCwd(enteredProject)
    lastProjectCwdSyncRef.current = enteredProject.id
  }, [inProject, enteredProject, syncProjectCwd])

  // A persisted scope can go stale (project archived/removed, or a profile
  // switch swapped the whole catalog). Once projects have loaded, drop back to
  // the overview if the scoped id is gone.
  useEffect(() => {
    if (projectScope !== ALL_PROJECTS && projectsActive && !enteredProject) {
      exitProjectScope()
    }
  }, [projectScope, projectsActive, enteredProject])

  // The project overview (drill-in list) vs. the entered project's content.
  const projectOverview = projectsActive && !inProject ? agentProjectTree : undefined

  // Preview rows come from the backend tree (each project carries its
  // most-recent sessions), overlaid with live $sessions so a just-created
  // session shows under its project instantly (and with its working arc),
  // matching the flat Recents list. Keyed by project id for the rows.
  const overviewPreviews = useMemo<Record<string, SessionInfo[]>>(
    () =>
      overlayLivePreviews(projectOverview ?? [], agentSessions, projects, PROJECT_PREVIEW_COUNT, {
        removed: removedSessionIds,
        // Rank before the trim, so "3 priciest in this project" isn't "3 most
        // recent, priciest first".
        rankIds: sortOrderIds
      }),
    [projectOverview, agentSessions, projects, removedSessionIds, sortOrderIds]
  )

  const onEnterProject = useCallback(
    (id: string) => {
      const project = projectModel.find(node => node.id === id)

      if (project) {
        syncProjectCwd(project)
      }

      enterProject(id)
    },
    [projectModel, syncProjectCwd]
  )

  // The Sessions section is a project switcher in grouped mode: its label reads
  // "Sessions" when flat, "Projects" at the overview, and the project's name
  // once you've entered one.
  const sessionsLabel =
    inProject && enteredProject ? enteredProject.label : worktreeGroupingActive ? s.projects.sectionLabel : s.sessions

  // Mirror the section's skeleton gate (projectsLoading + nothing to show yet):
  // while the skeleton is up there's no point also spinning the header count.
  const projectsSkeletonVisible =
    worktreeGroupingActive &&
    projectTreeLoading &&
    !projectOverview?.length &&
    !(inProject && (enteredProject?.sessionCount ?? 0) > 0)

  const runKeyedLoad = useCallback(
    (
      key: string,
      load: ((key: string) => Promise<void> | void) | undefined,
      setPending: React.Dispatch<React.SetStateAction<Record<string, boolean>>>
    ) => {
      if (!load) {
        return
      }

      setPending(prev => ({ ...prev, [key]: true }))

      void Promise.resolve(load(key))
        .catch(() => undefined)
        .finally(() => setPending(({ [key]: _done, ...rest }) => rest))
    },
    []
  )

  const loadMoreForMessaging = useCallback(
    (platform: string) => runKeyedLoad(platform, onLoadMoreMessaging, setMessagingLoadMorePending),
    [onLoadMoreMessaging, runKeyedLoad]
  )

  // Reveal another batch of a platform's rows; fetch from the backend too if we
  // run past what's loaded and more remain on disk.
  const revealMoreMessaging = (platform: string, loaded: number, hasMore: boolean) => {
    const next = (messagingVisible[platform] ?? NON_SESSION_INITIAL_ROWS) + NON_SESSION_LOAD_STEP

    setMessagingVisible(prev => ({ ...prev, [platform]: next }))

    if (next > loaded && hasMore) {
      loadMoreForMessaging(platform)
    }
  }

  // Each messaging platform is its own self-managed section: split the
  // separately-fetched messaging slice by source, newest platform first, rows
  // within a platform by recency. Per-platform totals (when a "load more" has
  // resolved them) drive the count + whether more remain on disk.
  const messagingGroups = useMemo<MessagingSection[]>(() => {
    if (!messagingSessions.length) {
      return []
    }

    const bySource = new Map<string, SessionInfo[]>()
    // Rows this platform owns that the Pinned section is showing instead. The
    // backend's per-platform total counts them, so discount it or "load more"
    // promises rows that will never appear.
    const pinnedBySource = new Map<string, number>()

    for (const session of messagingSessions) {
      const sourceId = normalizeSessionSource(session.source)

      if (!sourceId) {
        continue
      }

      if (isPinnedSession(session)) {
        pinnedBySource.set(sourceId, (pinnedBySource.get(sourceId) ?? 0) + 1)

        continue
      }

      const list = bySource.get(sourceId) ?? []
      list.push(session)
      bySource.set(sourceId, list)
    }

    return [...bySource.entries()]
      .map(([sourceId, list]) => {
        const ordered = [...list].sort((a, b) => sessionTime(b) - sessionTime(a))
        const known = messagingPlatformTotals[sourceId]
        const unpinnedKnown = known == null ? null : Math.max(0, known - (pinnedBySource.get(sourceId) ?? 0))
        const total = Math.max(ordered.length, unpinnedKnown ?? 0)

        return {
          // Known exact total → more exist iff total exceeds loaded; otherwise
          // the seed fetch was capped, so assume more until a per-platform load
          // resolves the count.
          hasMore: unpinnedKnown != null ? unpinnedKnown > ordered.length : messagingTruncated,
          label: sessionSourceLabel(sourceId) ?? sourceId,
          sessions: ordered,
          sourceId,
          total
        }
      })
      .sort((a, b) => sessionTime(b.sessions[0]) - sessionTime(a.sessions[0]))
  }, [messagingSessions, messagingPlatformTotals, messagingTruncated, isPinnedSession])

  // Grouping by profile: one collapsible group per profile, color on the header
  // (not on every row). Default profile floats to the top, the rest alpha.
  // Only reachable while the sidebar is showing every profile — scoped to one,
  // it would draw a single group around the whole list.
  const profileGrouped = showAllProfiles && grouping === 'profile'

  const profileGroups = useMemo<SidebarSessionGroup[] | undefined>(() => {
    if (!profileGrouped) {
      return undefined
    }

    const groups = new Map<string, SidebarSessionGroup>()

    for (const session of agentSessions) {
      const key = normalizeProfileKey(session.profile)

      const group = groups.get(key) ?? {
        color: resolveProfileColor(key, profileColors),
        id: key,
        label: key,
        mode: 'profile',
        path: null,
        sessions: []
      }

      group.sessions.push(session)

      groups.set(key, group)
    }

    // default (root) first, then the rest alphabetically.
    return [...groups.values()].sort((a, b) =>
      a.id === 'default' ? -1 : b.id === 'default' ? 1 : a.label.localeCompare(b.label)
    )
  }, [profileGrouped, agentSessions, profileColors])

  // The flat Sessions list always shows ALL recent sessions; Projects is a
  // parallel grouped view, not a filter on this one — nothing is hidden here.
  const displayAgentSessions = agentSessions

  // Pagination is scope-aware. In "All profiles" mode it tracks the global
  // unified set; scoped to one profile it tracks that profile's own truncation
  // flag — otherwise a huge default profile keeps "Load more" stuck on while
  // you browse a small one. The backend reports whether its page was capped
  // rather than an exact count, so no COUNT(*) runs per refresh.
  const loadedSessionCount = showAllProfiles ? sessions.length : scopedSessions.length

  // The archived view is its own (single, capped) query — paging the live
  // sessions list from under it would just fold un-archived rows back in.
  const hasMoreSessions =
    !showArchived &&
    (showAllProfiles
      ? Object.values(sessionProfilesTruncated).some(Boolean)
      : Boolean(sessionProfilesTruncated[profileScope]))

  const displayRecentsCountRef = useRef(0)
  const loadedRecentsCountRef = useRef(0)
  displayRecentsCountRef.current = displayAgentSessions.length
  loadedRecentsCountRef.current = loadedSessionCount

  const onLoadMoreRecents = useCallback(async () => {
    if (recentsLoadMorePending) {
      return
    }

    setRecentsLoadMorePending(true)

    try {
      const startVisible = displayRecentsCountRef.current
      const targetVisible = startVisible + SIDEBAR_SESSIONS_PAGE_SIZE
      let lastLoaded = loadedRecentsCountRef.current

      // Project-less recents can be sparse in the global recent stream (because
      // project-scoped sessions are filtered out in the UI). Keep paging until
      // we actually reveal a full page of visible rows, or the backend window
      // stops growing.
      for (let attempt = 0; attempt < 6; attempt += 1) {
        await Promise.resolve(onLoadMoreSessions())
        await new Promise<void>(resolve => window.requestAnimationFrame(() => resolve()))

        const visibleNow = displayRecentsCountRef.current
        const loadedNow = loadedRecentsCountRef.current

        if (visibleNow >= targetVisible) {
          break
        }

        if (loadedNow <= lastLoaded) {
          break
        }

        lastLoaded = loadedNow
      }
    } finally {
      setRecentsLoadMorePending(false)
    }
  }, [onLoadMoreSessions, recentsLoadMorePending])

  // Archived rows are excluded from the sessions query, so the view has to
  // fetch its own set.
  useEffect(() => {
    if (showArchived) {
      void loadArchivedSessions()
    }
  }, [showArchived])

  // Ranking by size is a question about the whole list ("what did I burn money
  // on"), so it drops the calendar dividers and ranks globally — "Today" above
  // the priciest session you have ever had would be a lie. Time- and
  // state-based keys stay bucketed, where they read correctly per day.
  const rankedGlobally = ordering === 'cost' || ordering === 'tokens'

  const displayAgentGroups = profileGroups

  // The recents list owns its own (virtualized) scroll container only when it's a
  // long flat list. In that case it must keep its scroller even in short mode, so
  // we don't flatten it (flattening would defeat virtualization). Short flat lists
  // and grouped views (profile groups or the worktree tree) flatten into the
  // single outer scroll instead.
  // Whichever grouping is active, the flat set of repo subtrees on screen — the
  // single source for reconciling repo/worktree order, whether repos hang off
  // the bare tree or are nested under projects.
  const activeRepoTrees = useMemo<SidebarWorkspaceTree[]>(
    () => (agentProjectTree ? agentProjectTree.flatMap(project => project.repos) : []),
    [agentProjectTree]
  )

  // Mirror the section's own virtualization inputs (the props it receives),
  // not the raw tree cache: agentProjectTree persists after leaving Project
  // grouping, and keying on it here while the section keys on projectOverview
  // (which is nulled the moment grouping changes) left the two disagreeing —
  // wrapper classes built for a virtualized list around a non-virtual one.
  // Entered-project content is the third prop that suppresses virtualization.
  const recentsVirtualizes =
    !displayAgentGroups?.length &&
    !projectOverview?.length &&
    !(inProject && enteredProjectContent) &&
    displayAgentSessions.length >= VIRTUALIZE_THRESHOLD

  // Keep the persisted parent + worktree orders reconciled with what's on screen:
  // freshly-seen repos/worktrees surface at the top, vanished ones drop out of
  // the saved order.
  useEffect(() => {
    if (!activeRepoTrees.length) {
      return
    }

    const nextParents = reconcileOrderIds(
      activeRepoTrees.map(parent => parent.id),
      workspaceParentOrderIds
    )

    if (!sameIds(nextParents, workspaceParentOrderIds)) {
      setSidebarWorkspaceParentOrderIds(nextParents)
    }

    const nextWorktrees = reconcileOrderIds(
      activeRepoTrees.flatMap(parent => parent.groups.map(group => group.id)),
      workspaceOrderIds
    )

    if (!sameIds(nextWorktrees, workspaceOrderIds)) {
      setSidebarWorkspaceOrderIds(nextWorktrees)
    }
  }, [activeRepoTrees, workspaceParentOrderIds, workspaceOrderIds])

  // Skeletons mean "still loading", so they key off the UNFILTERED set. Keyed
  // off the filtered one, a filter that matches nothing showed skeletons on
  // every background refresh instead of the empty state.
  const showSessionSkeletons = sessionsLoading && scopedSessions.length === 0

  // Filtered down to nothing still renders the section: the empty state is what
  // tells you the filter — not an empty account — is why the list is bare.
  const showSessionSections =
    showSessionSkeletons || filtersActive || sortedSessions.length > 0 || projectModel.length > 0

  // The sidebar's session-area mode — exposed as data-attributes so custom
  // skins can target project mode (overview vs. entered), archived, or search
  // without relying on internal class names. `data-sessions-project` carries
  // the entered project's id for per-project targeting.
  const sessionsMode: 'archived' | 'flat' | 'project' | 'projects' | 'search' = trimmedQuery
    ? 'search'
    : showArchived
      ? 'archived'
      : inProject
        ? 'project'
        : worktreeGroupingActive
          ? 'projects'
          : 'flat'

  // Each reorderable list reports its OWN new id order; persisting is a direct,
  // typed write — no id-prefix sniffing to figure out which level moved.
  const reorderSessions = (ids: string[]) => {
    setSidebarSessionOrderManual(true)
    setSidebarSessionOrderIds(ids)
  }

  // Persist the new project overview order (drag-to-reorder); orderByIds applies
  // it over the default sort, so stale/new ids reconcile on the next render.
  const reorderProjects = (ids: string[]) => setSidebarProjectOrderIds(ids)

  // Sortable rows carry live session ids; the pinned store is keyed by durable
  // (lineage-root) ids, so translate before persisting the new order.
  const reorderPinned = (ids: string[]) =>
    setPinnedSessionOrder(
      ids.map(id => {
        const session = sessionByAnyId.get(id)

        return session ? sessionPinId(session) : id
      })
    )

  return (
    <Sidebar
      className={cn(
        // Visibility is the layout tree's job (a hidden zone is display:none;
        // the narrow overlay renders the live instance) — the sidebar always
        // paints itself fully.
        'relative h-full min-w-0 overflow-hidden border-t-0 border-b-0 text-foreground transition-none',
        panesFlipped ? 'border-l border-r-0' : 'border-r border-l-0',
        'border-(--sidebar-edge-border) bg-(--ui-sidebar-surface-background) opacity-100'
      )}
      collapsible="none"
    >
      <SidebarContent className="gap-0 overflow-hidden bg-transparent px-2.5">
        <SidebarGroup className="shrink-0 p-0 pb-2 pt-[calc(var(--titlebar-height)+0.375rem)]">
          <SidebarGroupContent>
            <SidebarMenu className="gap-px">
              {[...SIDEBAR_NAV, ...contributedNav].map(item => {
                const isInteractive = Boolean(item.action) || Boolean(item.route)

                const active =
                  (item.id === 'skills' && currentView === 'skills') ||
                  (item.id === 'messaging' && currentView === 'messaging') ||
                  (item.id === 'artifacts' && currentView === 'artifacts') ||
                  (item.id === 'cron' && currentView === 'cron') ||
                  // Contributed rows light up at their own route.
                  (Boolean(item.route) && pathname === item.route)

                const isNewSession = item.id === 'new-session'

                const button = (
                  <SidebarMenuButton
                    aria-disabled={!isInteractive}
                    className={cn(
                      // no-drag: these rows sit directly under the titlebar's
                      // [-webkit-app-region:drag] strips (app-shell.tsx), with only
                      // 6px of clearance. Drag regions win hit-testing over DOM
                      // (pointer-events can't override), and on Linux/WSLg the
                      // resolved region has been observed to swallow clicks on the
                      // top rows. Same carve-out as USER_BUBBLE_BASE_CLASS in
                      // thread.tsx.
                      'flex h-7 w-full justify-start gap-2 rounded-md border border-transparent px-2 text-left text-[0.8125rem] font-medium text-(--ui-text-secondary) transition-colors duration-100 ease-out [-webkit-app-region:no-drag] hover:bg-(--ui-control-hover-background) hover:text-foreground hover:transition-none',
                      active &&
                        'border-(--ui-stroke-tertiary) bg-(--ui-control-active-background) text-foreground shadow-none hover:border-(--ui-stroke-tertiary)!',
                      !isInteractive &&
                        'cursor-default hover:border-transparent hover:bg-transparent hover:text-inherit'
                    )}
                    onClick={() => {
                      // A plain new session lands in whatever profile the live
                      // gateway is on (= the active switcher context). null →
                      // no swap. The switcher header is the single place to
                      // change which profile that is.
                      if (isNewSession) {
                        $newChatProfile.set(null)
                      }

                      onNavigate(item)
                    }}
                    tooltip={
                      item.keybindActionId
                        ? {
                            children: (
                              <TipKeybindLabel actionId={item.keybindActionId} text={s.nav[item.id] ?? item.label} />
                            )
                          }
                        : (s.nav[item.id] ?? item.label)
                    }
                    type="button"
                  >
                    <item.icon className="size-4 shrink-0 text-[color-mix(in_srgb,currentColor_72%,transparent)]" />
                    <span className="min-w-0 flex-1 truncate">{s.nav[item.id] ?? item.label}</span>
                    {isNewSession && (
                      <KbdGroup
                        className={cn('ml-auto opacity-55', newSessionKbdFlash && 'opacity-100!')}
                        keys={newSessionKbd}
                        size="sm"
                      />
                    )}
                  </SidebarMenuButton>
                )

                // New session + route-backed pages can open in a split —
                // right-click for the directional "Open in split" submenu.
                return (
                  <SidebarMenuItem key={item.id}>
                    {isNewSession || item.route ? (
                      <ContextMenu>
                        <ContextMenuTrigger asChild>{button}</ContextMenuTrigger>
                        <ContextMenuContent aria-label={s.nav[item.id] ?? item.label}>
                          <SplitSubmenu
                            kit={CONTEXT_SPLIT_KIT}
                            label={s.row.openInSplit}
                            onSplit={dir => {
                              if (isNewSession) {
                                onNewSessionSplit(dir)
                              } else if (item.route) {
                                openRouteTile(item.route, dir)
                              }
                            }}
                          />
                        </ContextMenuContent>
                      </ContextMenu>
                    ) : (
                      button
                    )}
                  </SidebarMenuItem>
                )
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {showSessionSections && (
          <div className="shrink-0 px-2 pb-1 pt-1">
            <SearchField
              aria-label={s.searchAria}
              inputRef={searchInputRef}
              onChange={setSearchQuery}
              placeholder={s.searchPlaceholder}
              value={searchQuery}
            />
          </div>
        )}

        {showSessionSections && (
          <div
            className={cn('flex min-h-0 flex-1 flex-col pb-1.75', SCROLL_Y, SCROLL_GUTTER)}
            data-sessions-mode={sessionsMode}
            data-sessions-project={inProject ? (enteredProjectId ?? undefined) : undefined}
          >
            {trimmedQuery && (
              <SidebarSessionsSection
                activeSessionId={activeSidebarSessionId}
                contentClassName={cn('flex min-h-0 flex-1 flex-col gap-px pb-1.75', SCROLL_Y)}
                emptyState={
                  searchPending ? (
                    <SidebarSessionSkeletons />
                  ) : (
                    <div className="wrap-anywhere grid min-h-24 place-items-center rounded-lg px-2 text-center text-xs text-(--ui-text-tertiary)">
                      {s.noMatch(trimmedQuery)}
                    </div>
                  )
                }
                label={s.results}
                onArchiveSession={onArchiveSession}
                onBranchSession={onBranchSession}
                onDeleteSession={onDeleteSession}
                onResumeSession={onResumeSession}
                onToggle={() => undefined}
                onTogglePin={pinSession}
                open
                pinned={false}
                rootClassName="min-h-32 flex-1 overflow-hidden p-0"
                sessions={searchResults}
                showProfileTags={showAllProfiles}
              />
            )}

            {!trimmedQuery && (
              <SidebarSessionsSection
                activeSessionId={activeSidebarSessionId}
                contentClassName="flex flex-col gap-px rounded-lg pb-2 pt-1"
                dndSensors={dndSensors}
                emptyState={<SidebarPinnedEmptyState />}
                label={s.pinned}
                onArchiveSession={onArchiveSession}
                onBranchSession={onBranchSession}
                onDeleteSession={onDeleteSession}
                onReorderSessions={reorderPinned}
                onResumeSession={onResumeSession}
                onToggle={() => setSidebarPinsOpen(!pinsOpen)}
                onTogglePin={unpinSession}
                open={pinsOpen}
                pinned
                rootClassName="shrink-0 p-0 pb-1"
                sessions={pinnedSessions}
                showProfileTags={showAllProfiles}
                sortable={pinnedSessions.length > 1}
              />
            )}

            {!trimmedQuery && (
              <SidebarSessionsSection
                activeProjectId={activeProjectId}
                activeSessionId={activeSidebarSessionId}
                // Inbox style is a render variant, not a grouping — it rides
                // whichever view is active: flat recents, project lanes, and
                // the overview previews all render the same card.
                card={cardRows}
                collapsible={!inProject}
                contentClassName={cn(
                  'flex min-h-0 flex-1 flex-col gap-px pb-1.75',
                  // The section is the ONE authority on whether the virtual
                  // list owns scrolling: it neutralizes this wrapper scroller
                  // itself (overflow-visible) when it virtualizes. Gating
                  // SCROLL_Y here on index's own parallel guess desynced the
                  // two — a cached project tree flipped this side but not the
                  // section's, leaving the list with no scroller at all and
                  // the recents pane rendering blank under Updated grouping.
                  SCROLL_Y,
                  // Flatten into the single scroll when compact — unless this is the
                  // virtualized long list, which must keep its own scroller.
                  !recentsVirtualizes && COMPACT_FLAT
                )}
                dndSensors={dndSensors}
                emptyState={
                  showSessionSkeletons ? (
                    <SidebarSessionSkeletons />
                  ) : (
                    <div className="grid min-h-16 place-items-center rounded-lg px-2 text-center text-xs text-(--ui-text-tertiary)">
                      {inProject
                        ? s.projectEmpty
                        : filtersActive
                          ? s.noFilterMatches
                          : pinnedSessions.length > 0
                            ? s.allPinned
                            : s.noSessions}
                    </div>
                  )
                }
                footer={
                  // Hidden only when workspace-grouped — those groups page
                  // themselves. Profile groups don't: this one footer fetches the
                  // next page, which grows every profile at once.
                  !agentsGrouped && !showSessionSkeletons && hasMoreSessions ? (
                    <SidebarLoadMoreRow
                      loading={sessionsLoading || recentsLoadMorePending}
                      onClick={() => void onLoadMoreRecents()}
                      // Recents are post-filtered to non-project sessions, so a
                      // backend page size (50) is not a truthful "rows you'll
                      // see" count. Use the generic label instead of a fake N.
                      step={0}
                    />
                  ) : null
                }
                forceEmptyState={showSessionSkeletons}
                // Archived is a plain list, and so is a magnitude-ranked one.
                // Otherwise project lanes stay chronological whatever the flat
                // list does — only the flat list can swap its dividers for
                // WORKING / DONE.
                grouping={showArchived || rankedGlobally ? 'none' : grouping === 'status' ? 'status' : 'date'}
                groups={displayAgentGroups}
                headerAction={
                  inProject && enteredProject ? (
                    <div className="group/workspace flex shrink-0 items-center gap-0.5">
                      {enteredProject.path && <StartWorkButton repoPath={enteredProject.path} />}
                      {/* Home has no folder and no record to rename, theme, or delete. */}
                      {!enteredProject.isNoProject && (
                        <ProjectMenu
                          isActive={enteredProject.id === activeProjectId}
                          onExitScope={exitProjectScope}
                          project={enteredProject}
                          scoped
                        />
                      )}
                      <div className="grid size-6 place-items-center">
                        <Tip label={s.showProjects}>
                          <Button
                            aria-label={s.showProjects}
                            className={HEADER_NAV_BTN}
                            onClick={event => {
                              event.stopPropagation()
                              exitProjectScope()
                            }}
                            size="icon-xs"
                            variant="ghost"
                          >
                            <Codicon name="list-unordered" size="0.75rem" />
                          </Button>
                        </Tip>
                      </div>
                    </div>
                  ) : (
                    <div className="flex shrink-0 items-center gap-0.5">
                      {!showAllProfiles ? (
                        <Tip label={agentsGrouped ? s.projects.newButton : s.nav['new-session']}>
                          <Button
                            aria-label={agentsGrouped ? s.projects.newButton : s.nav['new-session']}
                            className={HEADER_ACTION_BTN}
                            onClick={event => {
                              event.stopPropagation()

                              if (agentsGrouped) {
                                openProjectCreate()
                              } else {
                                onNewSessionInWorkspace(null)
                              }
                            }}
                            size="icon-xs"
                            variant="ghost"
                          >
                            <Codicon name="add" size="0.75rem" />
                          </Button>
                        </Tip>
                      ) : null}
                      <div className="grid size-6 place-items-center">
                        <SidebarFilterMenu className={HEADER_NAV_BTN} />
                      </div>
                    </div>
                  )
                }
                label={sessionsLabel}
                labelMeta={
                  worktreeGroupingActive ? (
                    reposScanning && !projectsSkeletonVisible ? (
                      <GlyphSpinner ariaLabel={s.loading} className="text-[0.6875rem] text-(--ui-text-quaternary)" />
                    ) : undefined
                  ) : undefined
                }
                liveSessions={inProject ? agentSessions : undefined}
                manualOrderIds={agentOrderManual ? agentOrderIds : sortOrderIds}
                onArchiveSession={onArchiveSession}
                onBranchSession={onBranchSession}
                onDeleteSession={onDeleteSession}
                onEnterProject={onEnterProject}
                // Unlike reorder below, this stays on across profiles: a folder
                // is a folder, and the new session lands in the active profile
                // — the same one the composer would have started it in.
                onNewSessionInWorkspace={onNewSessionInWorkspace}
                onReorderProjects={showAllProfiles ? undefined : reorderProjects}
                onReorderSessions={showAllProfiles ? undefined : reorderSessions}
                onResumeSession={onResumeSession}
                onToggle={() => setSidebarRecentsOpen(!agentsOpen)}
                onTogglePin={pinSession}
                open={agentsOpen}
                pinned={false}
                projectBackRow={
                  inProject ? <ProjectBackRow label={s.projects.back} onClick={exitProjectScope} /> : undefined
                }
                projectContent={inProject ? enteredProjectContent : undefined}
                projectOverview={projectOverview}
                projectOverviewPreviews={overviewPreviews}
                projectRepoWorktrees={inProject ? scopedRepoWorktrees : undefined}
                projectsLoading={worktreeGroupingActive ? projectTreeLoading : false}
                removedSessionIds={inProject ? removedSessionIds : undefined}
                rootClassName={cn(
                  'min-h-32 flex-1 overflow-hidden p-0',
                  !recentsVirtualizes && 'compact:min-h-0 compact:flex-none compact:overflow-visible'
                )}
                sessions={displayAgentSessions}
                sortable={!showAllProfiles && agentSessions.length > 1}
              />
            )}

            {!trimmedQuery &&
              !worktreeGroupingActive &&
              messagingGroups.map(group => {
                const visible = messagingVisible[group.sourceId] ?? NON_SESSION_INITIAL_ROWS
                const shownSessions = group.sessions.slice(0, visible)
                // More to show if rows are hidden behind the cap, or the backend
                // still has older threads on disk.
                const canRevealMore = visible < group.sessions.length || group.hasMore

                return (
                  <SidebarSessionsSection
                    activeSessionId={activeSidebarSessionId}
                    contentClassName={cn('flex max-h-56 flex-col gap-px pb-1.75', GROUP_BODY)}
                    emptyState={null}
                    footer={
                      canRevealMore ? (
                        <SidebarLoadMoreRow
                          loading={Boolean(messagingLoadMorePending[group.sourceId])}
                          onClick={() => revealMoreMessaging(group.sourceId, group.sessions.length, group.hasMore)}
                          step={Math.min(NON_SESSION_LOAD_STEP, Math.max(0, group.total - shownSessions.length))}
                        />
                      ) : null
                    }
                    key={group.sourceId}
                    label={group.label}
                    labelIcon={
                      <PlatformAvatar
                        className="size-4 rounded-[4px] text-[0.5625rem] [&_svg]:size-3"
                        platformId={group.sourceId}
                        platformName={group.label}
                      />
                    }
                    onArchiveSession={onArchiveSession}
                    onDeleteSession={onDeleteSession}
                    onResumeSession={onResumeSession}
                    onToggle={() => toggleSidebarMessagingOpen(group.sourceId)}
                    onTogglePin={pinSession}
                    open={messagingOpenIds.includes(group.sourceId)}
                    pinned={false}
                    rootClassName="shrink-0 p-0"
                    sessions={shownSessions}
                  />
                )
              })}

            {!trimmedQuery && !worktreeGroupingActive && cronJobs.length > 0 && (
              <SidebarCronJobsSection
                jobs={cronJobs}
                label={s.cronJobs}
                onManageJob={onManageCronJob}
                onOpenRun={onResumeSession}
                onToggle={() => setSidebarCronOpen(!cronOpen)}
                onTriggerJob={onTriggerCronJob}
                open={cronOpen}
              />
            )}
          </div>
        )}

        {!showSessionSections && <SidebarBlankState onNewProject={openProjectCreate} />}

        <div className="shrink-0 px-0.5 pb-1 pt-0.5">
          <ProfileRail />
        </div>
      </SidebarContent>
      <ProjectDialog />
      {/* One mount for the whole app. The header of WorktreeDialog tells why. */}
      <WorktreeDialog />
    </Sidebar>
  )
}

interface MessagingSection {
  sourceId: string
  label: string
  sessions: SessionInfo[]
  total: number
  hasMore: boolean
}
