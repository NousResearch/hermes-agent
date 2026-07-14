import { KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core'
import { sortableKeyboardCoordinates } from '@dnd-kit/sortable'
import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { PlatformAvatar } from '@/app/messaging/platform-icon'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
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
import { searchSessions, type SessionInfo, type SessionSearchResult } from '@/hermes'
import { useI18n } from '@/i18n'
import { comboTokens } from '@/lib/keybinds/combo'
import { profileColor } from '@/lib/profile-color'
import { sessionMatchesSearch } from '@/lib/session-search'
import { normalizeSessionSource, sessionSourceLabel } from '@/lib/session-source'
import { cn } from '@/lib/utils'
import { $cronJobs } from '@/store/cron'
import {
  $dismissedAutoProjectIds,
  $panesFlipped,
  $pinnedSessionIds,
  $sidebarAgentsGrouped,
  $sidebarCronOpen,
  $sidebarMessagingOpenIds,
  $sidebarOpen,
  $sidebarOverlayMounted,
  $sidebarPinsOpen,
  $sidebarProjectOrderIds,
  $sidebarRecentsOpen,
  $sidebarSessionOrderIds,
  $sidebarSessionOrderManual,
  $sidebarWorkspaceOrderIds,
  $sidebarWorkspaceParentOrderIds,
  pinSession,
  SESSION_SEARCH_FOCUS_EVENT,
  setPinnedSessionOrder,
  setSidebarAgentsGrouped,
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
import { $newChatProfile, $profiles, $profileScope, ALL_PROFILES, normalizeProfileKey } from '@/store/profile'
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
  $cronSessions,
  $currentCwd,
  $gatewayState,
  $messagingPlatformTotals,
  $messagingSessions,
  $messagingTruncated,
  $selectedStoredSessionId,
  $sessionProfileTotals,
  $sessions,
  $sessionsLoading,
  $sessionsTotal,
  $workingSessionIds,
  sessionPinId,
  setCurrentCwd
} from '@/store/session'

import { type AppView, ARTIFACTS_ROUTE, CALENDAR_ROUTE, MESSAGING_ROUTE, SKILLS_ROUTE } from '../../routes'
import type { SidebarNavItem } from '../../types'

import { countLabel } from './chrome'
import { SidebarCronJobsSection } from './cron-jobs-section'
import { SidebarLoadMoreRow } from './load-more-row'
import { orderByIds, reconcileOrderIds, resolveManualSessionOrderIds, sameIds } from './order'
import { ProfileRail } from './profile-switcher'
import { ProjectDialog } from './project-dialog'
import {
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
import { SidebarBlankState, SidebarPinnedEmptyState, SidebarSessionSkeletons } from './section-states'
import { SidebarSessionsSection, VIRTUALIZE_THRESHOLD } from './sessions-section'

// Non-session groups (messaging platforms) stay compact: show a few rows up
// front, reveal more in larger steps on demand. Keeps a busy platform from
// dominating the sidebar before the user asks to see it.
const NON_SESSION_INITIAL_ROWS = 3
const NON_SESSION_LOAD_STEP = 10

const NEW_SESSION_KBD = comboTokens('mod+n')

const SIDEBAR_NAV: SidebarNavItem[] = [
  {
    id: 'new-session',
    label: '',
    icon: props => <Codicon name="robot" {...props} />,
    action: 'new-session'
  },
  {
    id: 'skills',
    label: '',
    icon: props => <Codicon name="symbol-misc" {...props} />,
    route: SKILLS_ROUTE
  },
  { id: 'messaging', label: '', icon: props => <Codicon name="comment" {...props} />, route: MESSAGING_ROUTE },
  { id: 'artifacts', label: '', icon: props => <Codicon name="files" {...props} />, route: ARTIFACTS_ROUTE },
  { id: 'calendar', label: 'Calendar', icon: props => <Codicon name="calendar" {...props} />, route: CALENDAR_ROUTE }
]

// Two modes via the `compact` height variant (styles.css):
//   tall    → each section is shrink-0, capped, its own scroller; Sessions is flex-1.
//   compact → COMPACT_FLAT drops the caps so the whole stack scrolls as one.
// Sections stay shrink-0 so none can be squeezed below its content and bleed onto
// the next — the flexbox `min-height: auto` overlap trap that caused the bug.
const COMPACT_FLAT = 'compact:max-h-none compact:overflow-visible'

// Vertical scroll only — never a horizontal bar from glow bleed, long titles, etc.
const SCROLL_Y = 'overflow-y-auto overflow-x-hidden overscroll-contain'

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
  onLoadMoreProfileSessions?: (profile: string) => Promise<void> | void
  onLoadMoreMessaging?: (platform: string) => Promise<void> | void
  onResumeSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
  onArchiveSession: (sessionId: string) => void
  onBranchSession: (sessionId: string) => void
  onNewSessionInWorkspace: (path: null | string) => void
  onManageCronJob: (jobId: string) => void
  onTriggerCronJob: (jobId: string) => void
}

export function ChatSidebar({
  currentView,
  onNavigate,
  onLoadMoreSessions,
  onLoadMoreProfileSessions,
  onLoadMoreMessaging,
  onResumeSession,
  onDeleteSession,
  onArchiveSession,
  onBranchSession,
  onNewSessionInWorkspace,
  onManageCronJob,
  onTriggerCronJob
}: ChatSidebarProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const sidebarOpen = useStore($sidebarOpen)
  // Collapsed-but-overlay-mounted → render the full sidebar, not just the nav rail.
  const overlayMounted = useStore($sidebarOverlayMounted)
  const contentVisible = sidebarOpen || overlayMounted
  const panesFlipped = useStore($panesFlipped)
  const agentsGrouped = useStore($sidebarAgentsGrouped)
  const pinnedSessionIds = useStore($pinnedSessionIds)
  const pinsOpen = useStore($sidebarPinsOpen)
  const agentsOpen = useStore($sidebarRecentsOpen)
  const cronOpen = useStore($sidebarCronOpen)
  const selectedSessionId = useStore($selectedStoredSessionId)
  const sessions = useStore($sessions)
  const cronSessions = useStore($cronSessions)
  const cronJobs = useStore($cronJobs)
  const messagingSessions = useStore($messagingSessions)
  const messagingPlatformTotals = useStore($messagingPlatformTotals)
  const messagingTruncated = useStore($messagingTruncated)
  const sessionsLoading = useStore($sessionsLoading)
  const sessionsTotal = useStore($sessionsTotal)
  const sessionProfileTotals = useStore($sessionProfileTotals)
  const workingSessionIds = useStore($workingSessionIds)
  const profiles = useStore($profiles)
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
  const [searchQuery, setSearchQuery] = useState('')
  const [serverMatches, setServerMatches] = useState<SessionSearchResult[]>([])
  const [searchPending, setSearchPending] = useState(false)
  const [newSessionKbdFlash, setNewSessionKbdFlash] = useState(false)
  const [profileLoadMorePending, setProfileLoadMorePending] = useState<Record<string, boolean>>({})
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
  const visibleSessions = useMemo(
    () => (showAllProfiles ? sessions : sessions.filter(s => normalizeProfileKey(s.profile) === profileScope)),
    [sessions, showAllProfiles, profileScope]
  )

  // Agent session order is pinned to creation time (started_at), NOT activity —
  // a new message must never float a session to the top. Position only changes
  // for a brand-new session or an explicit manual drag (agentOrderIds).
  const sortedSessions = useMemo(
    () => [...visibleSessions].sort((a, b) => (b.started_at || 0) - (a.started_at || 0)),
    [visibleSessions]
  )

  const workingSessionIdSet = useMemo(() => new Set(workingSessionIds), [workingSessionIds])

  // Index sessions by both their live id and their lineage-root id so a pin
  // stored as the pre-compression root resolves to the live continuation tip.
  const sessionByAnyId = useMemo(() => {
    const map = new Map<string, SessionInfo>()

    // Cron sessions are listed separately but can still be pinned, so index
    // them too — otherwise a pinned cron job can't resolve into the Pinned
    // section. Recents take precedence on id collisions (set last).
    for (const s of [...cronSessions, ...visibleSessions]) {
      map.set(s.id, s)

      if (s._lineage_root_id && !map.has(s._lineage_root_id)) {
        map.set(s._lineage_root_id, s)
      }
    }

    return map
  }, [visibleSessions, cronSessions])

  const pinnedSessions = useMemo(() => {
    const seen = new Set<string>()
    const out: SessionInfo[] = []

    for (const pinId of pinnedSessionIds) {
      const session = sessionByAnyId.get(pinId)

      if (session && !seen.has(session.id)) {
        seen.add(session.id)
        out.push(session)
      }
    }

    return out
  }, [pinnedSessionIds, sessionByAnyId])

  const pinnedRealIdSet = useMemo(() => new Set(pinnedSessions.map(s => s.id)), [pinnedSessions])

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
    () => sortedSessions.filter(s => !pinnedRealIdSet.has(s.id)),
    [sortedSessions, pinnedRealIdSet]
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

  const agentSessions = useMemo(
    () => (agentOrderManual ? orderByIds(unpinnedAgentSessions, s => s.id, agentOrderIds) : unpinnedAgentSessions),
    [unpinnedAgentSessions, agentOrderIds, agentOrderManual]
  )

  // Recents are local-only: messaging-platform sessions are fetched as their
  // own slice ($messagingSessions) and rendered in self-managed per-platform
  // sections below, so there is no source-grouping magic to untangle here.
  //
  // Workspace grouping is a `project -> repo -> lane -> sessions` tree computed
  // authoritatively on the backend (projects.tree). Parents reorder via
  // workspaceParentOrderIds; worktrees within a parent via workspaceOrderIds.
  const worktreeGroupingActive = agentsGrouped && !showAllProfiles
  const gatewayReady = gatewayState === 'open'

  // The backend project tree is a structural snapshot, NOT a per-message feed.
  // Refresh it on structural edges only — entering the grouped view, a profile
  // switch, gateway (re)connect — plus the once-per-run disk scan. Live session
  // changes between refreshes are reflected by the in-memory overlay
  // (overlayLiveLanes / overlayLivePreviews) off `$sessions`, so a turn
  // completing does NOT re-run the heavy list_sessions_rich scan. Project
  // mutations refresh the tree from their own store actions.
  useEffect(() => {
    if (worktreeGroupingActive && gatewayReady) {
      void refreshProjects()
      // Paint the list from the fast tree fetch (explicit projects + repos from
      // existing sessions / the backend cache) FIRST, then kick off the heavy
      // home-dir git crawl so newly-discovered repos fold in afterward — instead
      // of the crawl blocking the first render.
      void refreshProjectTree().finally(() => void scanAndRecordRepos())
    }
  }, [worktreeGroupingActive, profileScope, gatewayReady])

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
  }, [worktreeGroupingActive, gatewayReady])

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
    if (showAllProfiles) {
      return []
    }

    const dismissed = new Set(dismissedAutoProjects)

    const sorted = sortProjectsForOverview(
      projectTree
        .filter(node => !(node.isAuto && dismissed.has(node.id)))
        .map(project => ({ ...project, repos: orderRepos(project.repos) })),
      activeProjectId
    )

    // Layer the user's manual drag-order on top of the deterministic sort. Empty
    // (default) returns `sorted` untouched; new projects surface on top.
    return orderByIds(sorted, project => project.id, projectOrderIds)
  }, [showAllProfiles, projectTree, dismissedAutoProjects, orderRepos, activeProjectId, projectOrderIds])

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
    // out-of-tree worktrees can be placed). Here we just order the snapshot.
    return { ...hydrated, repos: orderRepos(hydrated.repos) }
  }, [overviewEnteredProject, enteredProjectTree, orderRepos])

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
  const prevWorkingIdsRef = useRef<string[]>(workingSessionIds)

  useEffect(() => {
    const prev = prevWorkingIdsRef.current
    prevWorkingIdsRef.current = workingSessionIds

    // A session leaving the working set means its turn just completed.
    const aTurnSettled = prev.some(id => !workingSessionIds.includes(id))

    if (inEnteredProject && aTurnSettled) {
      refreshWorktrees()
    }
  }, [workingSessionIds, inEnteredProject])

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
  // matching the flat Recents list. Keyed by project path for the rows.
  const overviewPreviews = useMemo<Record<string, SessionInfo[]>>(
    () => overlayLivePreviews(projectOverview ?? [], agentSessions, projects, PROJECT_PREVIEW_COUNT, removedSessionIds),
    [projectOverview, agentSessions, projects, removedSessionIds]
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

  const loadMoreForProfileGroup = useCallback(
    (profile: string) => runKeyedLoad(profile, onLoadMoreProfileSessions, setProfileLoadMorePending),
    [onLoadMoreProfileSessions, runKeyedLoad]
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

    for (const session of messagingSessions) {
      const sourceId = normalizeSessionSource(session.source)

      if (!sourceId) {
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
        const total = Math.max(ordered.length, known ?? 0)

        return {
          // Known exact total → more exist iff total exceeds loaded; otherwise
          // the seed fetch was capped, so assume more until a per-platform load
          // resolves the count.
          hasMore: known != null ? known > ordered.length : messagingTruncated,
          label: sessionSourceLabel(sourceId) ?? sourceId,
          sessions: ordered,
          sourceId,
          total
        }
      })
      .sort((a, b) => sessionTime(b.sessions[0]) - sessionTime(a.sessions[0]))
  }, [messagingSessions, messagingPlatformTotals, messagingTruncated])

  // ALL-profiles view: one collapsible group per profile, color on the header
  // (not on every row). Default profile floats to the top, the rest alpha.
  const profileGroups = useMemo<SidebarSessionGroup[] | undefined>(() => {
    if (!showAllProfiles) {
      return undefined
    }

    const groups = new Map<string, SidebarSessionGroup>()

    for (const session of agentSessions) {
      const key = normalizeProfileKey(session.profile)

      const group = groups.get(key) ?? {
        color: profileColor(key),
        id: key,
        label: key,
        mode: 'profile',
        path: null,
        sessions: []
      }

      group.sessions.push(session)

      groups.set(key, group)
    }

    return (
      [...groups.values()]
        .map(group => ({
          ...group,
          loadingMore: Boolean(profileLoadMorePending[group.id]),
          onLoadMore: onLoadMoreProfileSessions ? () => loadMoreForProfileGroup(group.id) : undefined,
          totalCount: Math.max(group.sessions.length, sessionProfileTotals[group.id] ?? 0)
        }))
        // default (root) first, then the rest alphabetically.
        .sort((a, b) => (a.id === 'default' ? -1 : b.id === 'default' ? 1 : a.label.localeCompare(b.label)))
    )
  }, [
    showAllProfiles,
    agentSessions,
    loadMoreForProfileGroup,
    onLoadMoreProfileSessions,
    profileLoadMorePending,
    sessionProfileTotals
  ])

  // The flat Sessions list always shows ALL recent sessions; Projects is a
  // parallel grouped view, not a filter on this one — nothing is hidden here.
  const displayAgentSessions = agentSessions

  // Pagination is scope-aware. In "All profiles" mode it tracks the global
  // unified set. When scoped to one profile it must compare that profile's own
  // loaded rows against that profile's total — otherwise a huge default profile
  // keeps "Load more" stuck on while you browse a small one (the aggregator's
  // total sums every profile). Per-profile totals come from the aggregator
  // (children excluded); fall back to the global total / loaded count.
  const loadedSessionCount = showAllProfiles ? sessions.length : visibleSessions.length
  const scopedProfileTotal = showAllProfiles ? undefined : sessionProfileTotals[profileScope]

  const knownSessionTotal = Math.max(
    showAllProfiles ? sessionsTotal : (scopedProfileTotal ?? loadedSessionCount),
    loadedSessionCount
  )

  const hasMoreSessions = knownSessionTotal > loadedSessionCount

  const recentsMeta = countLabel(displayAgentSessions.length, knownSessionTotal)
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

  const displayAgentGroups = showAllProfiles ? profileGroups : undefined

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

  const recentsVirtualizes =
    !displayAgentGroups?.length && !agentProjectTree?.length && displayAgentSessions.length >= VIRTUALIZE_THRESHOLD

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

  const showSessionSkeletons = sessionsLoading && sortedSessions.length === 0

  const showSessionSections = showSessionSkeletons || sortedSessions.length > 0 || projectModel.length > 0

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
        'relative h-full min-w-0 overflow-hidden border-t-0 border-b-0 text-foreground transition-none',
        panesFlipped ? 'border-l border-r-0' : 'border-r border-l-0',
        sidebarOpen
          ? 'border-(--sidebar-edge-border) bg-(--ui-sidebar-surface-background) opacity-100'
          : 'pointer-events-none border-transparent bg-transparent opacity-0',
        // While floated by PaneShell's hover-reveal, force visible + interactive
        // — on hover (group-hover/reveal) or when keyboard-pinned (data-forced).
        'in-data-[pane-hover-reveal=open]:pointer-events-auto in-data-[pane-hover-reveal=open]:border-(--sidebar-edge-border) in-data-[pane-hover-reveal=open]:bg-(--ui-sidebar-surface-background) in-data-[pane-hover-reveal=open]:opacity-100',
        'group-hover/reveal:pointer-events-auto group-hover/reveal:border-(--sidebar-edge-border) group-hover/reveal:bg-(--ui-sidebar-surface-background) group-hover/reveal:opacity-100'
      )}
      collapsible="none"
    >
      <SidebarContent className="gap-0 overflow-hidden bg-transparent px-2.5">
        <SidebarGroup className="shrink-0 p-0 pb-2 pt-[calc(var(--titlebar-height)+0.375rem)]">
          <SidebarGroupContent>
            <SidebarMenu className="gap-px">
              {SIDEBAR_NAV.map(item => {
                const isInteractive = Boolean(item.action) || Boolean(item.route)

                const active =
                  (item.id === 'skills' && currentView === 'skills') ||
                  (item.id === 'messaging' && currentView === 'messaging') ||
                  (item.id === 'artifacts' && currentView === 'artifacts')

                const isNewSession = item.id === 'new-session'

                return (
                  <SidebarMenuItem key={item.id}>
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
                      tooltip={s.nav[item.id] ?? item.label}
                      type="button"
                    >
                      <item.icon className="size-4 shrink-0 text-[color-mix(in_srgb,currentColor_72%,transparent)]" />
                      {contentVisible && (
                        <>
                          <span className="min-w-0 flex-1 truncate">{s.nav[item.id] ?? item.label}</span>
                          {isNewSession && (
                            <KbdGroup
                              className={cn('ml-auto opacity-55', newSessionKbdFlash && 'opacity-100!')}
                              keys={[...NEW_SESSION_KBD]}
                              size="sm"
                            />
                          )}
                        </>
                      )}
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                )
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {contentVisible && showSessionSections && (
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

        {contentVisible && showSessionSections && (
          <div className={cn('flex min-h-0 flex-1 flex-col pb-1.75', SCROLL_Y)}>
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
                labelMeta={String(searchResults.length)}
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
                workingSessionIdSet={workingSessionIdSet}
              />
            )}

            {!trimmedQuery && (
              <SidebarSessionsSection
                activeSessionId={activeSidebarSessionId}
                contentClassName={cn('flex max-h-44 flex-col gap-px rounded-lg pb-2 pt-1', GROUP_BODY)}
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
                sortable={pinnedSessions.length > 1}
                workingSessionIdSet={workingSessionIdSet}
              />
            )}

            {!trimmedQuery && (
              <SidebarSessionsSection
                activeProjectId={activeProjectId}
                activeSessionId={activeSidebarSessionId}
                collapsible={!inProject}
                contentClassName={cn(
                  'flex min-h-0 flex-1 flex-col pb-1.75',
                  SCROLL_Y,
                  // Separate profile sections clearly in the ALL view; rows inside
                  // each group keep their own tight gap-px rhythm.
                  showAllProfiles ? 'gap-3' : 'gap-px',
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
                      {inProject ? s.projectEmpty : pinnedSessions.length > 0 ? s.allPinned : s.noSessions}
                    </div>
                  )
                }
                footer={
                  // Hide "load more" only when workspace-grouped (those groups page
                  // themselves). ALL-profiles now pages per-profile from each profile
                  // header; the global footer only applies to non-ALL views.
                  !showAllProfiles && !agentsGrouped && !showSessionSkeletons && hasMoreSessions ? (
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
                groups={displayAgentGroups}
                headerAction={
                  inProject && enteredProject ? (
                    <div className="group/workspace flex shrink-0 items-center gap-0.5">
                      {enteredProject.path && (
                        <StartWorkButton onStarted={onNewSessionInWorkspace} repoPath={enteredProject.path} />
                      )}
                      <ProjectMenu
                        isActive={enteredProject.id === activeProjectId}
                        onExitScope={exitProjectScope}
                        project={enteredProject}
                        scoped
                      />
                      <div className="grid size-6 place-items-center">
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
                      </div>
                    </div>
                  ) : (
                    <div className="flex shrink-0 items-center gap-0.5">
                      {!showAllProfiles ? (
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
                      ) : null}
                      <div className="grid size-6 place-items-center">
                        {!showAllProfiles && agentSessions.length > 0 ? (
                          <Button
                            aria-label={agentsGrouped ? s.showSessions : s.showProjects}
                            className={cn(
                              HEADER_NAV_BTN,
                              agentsGrouped && 'bg-(--ui-control-active-background) text-foreground opacity-100'
                            )}
                            onClick={event => {
                              event.stopPropagation()
                              setSidebarRecentsOpen(true)
                              setSidebarAgentsGrouped(!agentsGrouped)
                            }}
                            size="icon-xs"
                            variant="ghost"
                          >
                            <Codicon name={agentsGrouped ? 'list-unordered' : 'root-folder'} size="0.75rem" />
                          </Button>
                        ) : null}
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
                  ) : (
                    recentsMeta
                  )
                }
                liveSessions={inProject ? agentSessions : undefined}
                onArchiveSession={onArchiveSession}
                onBranchSession={onBranchSession}
                onDeleteSession={onDeleteSession}
                onEnterProject={onEnterProject}
                onNewSessionInWorkspace={showAllProfiles ? undefined : onNewSessionInWorkspace}
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
                workingSessionIdSet={workingSessionIdSet}
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
                    labelMeta={countLabel(group.sessions.length, group.total)}
                    onArchiveSession={onArchiveSession}
                    onDeleteSession={onDeleteSession}
                    onResumeSession={onResumeSession}
                    onToggle={() => toggleSidebarMessagingOpen(group.sourceId)}
                    onTogglePin={pinSession}
                    open={messagingOpenIds.includes(group.sourceId)}
                    pinned={false}
                    rootClassName="shrink-0 p-0"
                    sessions={shownSessions}
                    workingSessionIdSet={workingSessionIdSet}
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

        {contentVisible && !showSessionSections && <SidebarBlankState onNewProject={openProjectCreate} />}

        {contentVisible && (
          <div className="shrink-0 px-0.5 pb-1 pt-0.5">
            <ProfileRail />
          </div>
        )}
      </SidebarContent>
      <ProjectDialog />
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
