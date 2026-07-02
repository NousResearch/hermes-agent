import {
  closestCenter,
  DndContext,
  type DragEndEvent,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors
} from '@dnd-kit/core'
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy
} from '@dnd-kit/sortable'
import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
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
import { Skeleton } from '@/components/ui/skeleton'
import { Tip } from '@/components/ui/tooltip'
import { searchSessions, type SessionInfo, type SessionSearchResult } from '@/hermes'
import { useI18n } from '@/i18n'
import { comboTokens } from '@/lib/keybinds/combo'
import { profileColor } from '@/lib/profile-color'
import { sessionMatchesSearch } from '@/lib/session-search'
import { cn } from '@/lib/utils'
import {
  $panesFlipped,
  $pinnedSessionIds,
  $sidebarOpen,
  $sidebarOverlayMounted,
  $sidebarPinsOpen,
  $sidebarProjectsOpen,
  $sidebarRecentsOpen,
  $sidebarSessionOrderIds,
  $sidebarSessionOrderManual,
  pinSession,
  SESSION_SEARCH_FOCUS_EVENT,
  setPinnedSessionOrder,
  setSidebarPinsOpen,
  setSidebarProjectsOpen,
  setSidebarRecentsOpen,
  setSidebarSessionOrderIds,
  setSidebarSessionOrderManual,
  SIDEBAR_SESSIONS_PAGE_SIZE,
  unpinSession
} from '@/store/layout'
import {
  $newChatProfile,
  $profiles,
  $profileScope,
  ALL_PROFILES,
  newSessionInProfile,
  normalizeProfileKey
} from '@/store/profile'
import {
  $cronSessions,
  $selectedStoredSessionId,
  $sessionProfileTotals,
  $sessions,
  $sessionsLoading,
  $sessionsTotal,
  $workingSessionIds,
  sessionPinId
} from '@/store/session'

import { type AppView, ARTIFACTS_ROUTE, CRON_ROUTE, SKILLS_ROUTE } from '../../routes'
import { SidebarPanelLabel } from '../../shell/sidebar-label'
import type { SidebarNavItem } from '../../types'

import { AccountPanel } from './account-panel'
import { SidebarLoadMoreRow } from './load-more-row'
import { resolveManualSessionOrderIds } from './order'
import { SidebarSessionRow } from './session-row'
import { VirtualSessionList } from './virtual-session-list'
import { isProjectCwd, type SidebarSessionGroup, workspaceGroupsFor } from './workspace-groups'

const VIRTUALIZE_THRESHOLD = 25

const NEW_SESSION_KBD = comboTokens('mod+n')

// Codex-style first screen: 新对话 / 搜索 / 已安排 / 插件 / 产物. Messaging
// moved off the first screen; artifacts moved back here from Settings (its
// settings nav entry is gone).
const SIDEBAR_NAV: SidebarNavItem[] = [
  {
    id: 'new-session',
    label: '',
    icon: props => <Codicon name="edit" {...props} />,
    action: 'new-session'
  },
  {
    id: 'search',
    label: '',
    icon: props => <Codicon name="search" {...props} />,
    action: 'search'
  },
  {
    id: 'cron',
    label: '',
    icon: props => <Codicon name="calendar" {...props} />,
    route: CRON_ROUTE
  },
  {
    id: 'skills',
    label: '',
    icon: props => <Codicon name="extensions" {...props} />,
    route: SKILLS_ROUTE
  },
  {
    id: 'artifacts',
    label: '',
    icon: props => <Codicon name="package" {...props} />,
    route: ARTIFACTS_ROUTE
  }
]

const WORKSPACE_PAGE = 5
// ALL-profiles view: show only the latest N per profile up front to keep the
// unified list scannable, then reveal/fetch more in N-sized steps on demand.
const PROFILE_INITIAL_PAGE = 5
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

// Sidebar reordering is a strictly vertical list. The dragged item's transform
// is rendered Y-only in useSortableBindings (no x, no scale); this just stops
// dnd-kit's auto-scroll from dragging the rail — or the window — sideways when
// the pointer nears an edge, killing the horizontal "drag to valhalla".
const reorderAutoScroll = { threshold: { x: 0, y: 0.2 } }

// One self-contained, nesting-safe reorderable list. It owns its DndContext, so a
// drag only ever collides with THIS list's own items (pinned rows vs. chat rows)
// and reordering "just works" without leaking into the lists around or inside it.
// Pair each item with useSortableBindings(id); the list reports the new id order
// and the caller persists it. This is the single generic primitive behind every
// reorderable surface in the sidebar.
function ReorderableList({
  children,
  ids,
  onReorder,
  sensors
}: {
  children: React.ReactNode
  ids: string[]
  onReorder: (ids: string[]) => void
  sensors?: ReturnType<typeof useSensors>
}) {
  const handleDragEnd = ({ active, over }: DragEndEvent) => {
    if (!over || active.id === over.id) {
      return
    }

    const from = ids.indexOf(String(active.id))
    const to = ids.indexOf(String(over.id))

    if (from >= 0 && to >= 0) {
      onReorder(arrayMove(ids, from, to))
    }
  }

  return (
    <DndContext autoScroll={reorderAutoScroll} collisionDetection={closestCenter} onDragEnd={handleDragEnd} sensors={sensors}>
      <SortableContext items={ids} strategy={verticalListSortingStrategy}>
        {children}
      </SortableContext>
    </DndContext>
  )
}

const countLabel = (loaded: number, total: number) => (total > loaded ? `${loaded}/${total}` : String(loaded))

function orderByIds<T>(items: T[], getId: (item: T) => string, orderIds: string[]): T[] {
  if (!orderIds.length) {
    return items
  }

  const byId = new Map(items.map(item => [getId(item), item]))
  const seen = new Set<string>()
  const ordered: T[] = []

  for (const id of orderIds) {
    const item = byId.get(id)

    if (item) {
      ordered.push(item)
      seen.add(id)
    }
  }

  // Items missing from the persisted order are new since it was last
  // reconciled. Callers pass recency-sorted lists (newest first), so surface
  // these at the TOP instead of burying them beneath the saved order —
  // otherwise a brand-new session sinks to the bottom of the sidebar and reads
  // as "my latest session never showed up".
  const fresh = items.filter(item => !seen.has(getId(item)))

  return fresh.length ? [...fresh, ...ordered] : ordered
}

function sameIds(left: string[], right: string[]) {
  return left.length === right.length && left.every((item, index) => item === right[index])
}

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

function useSortableBindings(id: string) {
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({ id })

  return {
    dragging: isDragging,
    dragHandleProps: { ...attributes, ...listeners },
    ref: setNodeRef,
    reorderable: true as const,
    style: {
      // Uniform vertical list: only ever translate on Y. Ignoring x and the
      // scaleX/scaleY that CSS.Transform.toString would emit keeps a dragged
      // group/row from drifting sideways or morphing its size mid-drag.
      transform: transform ? `translate3d(0px, ${transform.y}px, 0)` : undefined,
      transition: isDragging ? undefined : transition,
      willChange: isDragging ? 'transform' : undefined
    }
  }
}

interface ChatSidebarProps extends React.ComponentProps<typeof Sidebar> {
  currentView: AppView
  onNavigate: (item: SidebarNavItem) => void
  onLoadMoreSessions: () => void
  onLoadMoreProfileSessions?: (profile: string) => Promise<void> | void
  onResumeSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
  onArchiveSession: (sessionId: string) => void
  onNewSessionInWorkspace: (path: null | string) => void
}

export function ChatSidebar({
  currentView,
  onNavigate,
  onLoadMoreSessions,
  onLoadMoreProfileSessions,
  onResumeSession,
  onDeleteSession,
  onArchiveSession,
  onNewSessionInWorkspace
}: ChatSidebarProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const sidebarOpen = useStore($sidebarOpen)
  // Collapsed-but-overlay-mounted → render the full sidebar, not just the nav rail.
  const overlayMounted = useStore($sidebarOverlayMounted)
  const contentVisible = sidebarOpen || overlayMounted
  const panesFlipped = useStore($panesFlipped)
  const pinnedSessionIds = useStore($pinnedSessionIds)
  const pinsOpen = useStore($sidebarPinsOpen)
  const projectsOpen = useStore($sidebarProjectsOpen)
  const agentsOpen = useStore($sidebarRecentsOpen)
  const selectedSessionId = useStore($selectedStoredSessionId)
  const sessions = useStore($sessions)
  const cronSessions = useStore($cronSessions)
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
  const [searchQuery, setSearchQuery] = useState('')
  // Codex-style search: the field stays hidden until the 搜索 nav row (or the
  // focus-search hotkey) opens it; closing clears the query so the list snaps
  // back to the normal sections.
  const [searchOpen, setSearchOpen] = useState(false)
  const [serverMatches, setServerMatches] = useState<SessionSearchResult[]>([])
  const [newSessionKbdFlash, setNewSessionKbdFlash] = useState(false)
  const [profileLoadMorePending, setProfileLoadMorePending] = useState<Record<string, boolean>>({})
  const searchInputRef = useRef<HTMLInputElement>(null)
  const trimmedQuery = searchQuery.trim()

  const openSearch = useCallback(() => {
    setSearchOpen(true)
    // The field may not be mounted yet (it renders on searchOpen), so focus on
    // the next frame, after React has committed it.
    requestAnimationFrame(() => searchInputRef.current?.focus({ preventScroll: true }))
  }, [])

  const closeSearch = useCallback(() => {
    setSearchOpen(false)
    setSearchQuery('')
  }, [])

  // Hotkey (session.focusSearch) → reveal + focus the field.
  useEffect(() => {
    const onFocus = () => openSearch()

    window.addEventListener(SESSION_SEARCH_FOCUS_EVENT, onFocus)

    return () => window.removeEventListener(SESSION_SEARCH_FOCUS_EVENT, onFocus)
  }, [openSearch])

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

      return
    }

    let cancelled = false

    const id = window.setTimeout(() => {
      void searchSessions(trimmedQuery)
        .then(res => {
          if (!cancelled) {
            setServerMatches(res.results)
          }
        })
        .catch(() => undefined)
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

  const unpinnedSessions = useMemo(
    () => sortedSessions.filter(s => !pinnedRealIdSet.has(s.id)),
    [sortedSessions, pinnedRealIdSet]
  )

  // Codex-style split: sessions whose cwd is a real workspace folder (not the
  // home dir) live under 项目, grouped one row per distinct folder; everything
  // else is a plain 对话. The 项目 section disappears entirely when no session
  // has a project cwd.
  const projectSessions = useMemo(() => unpinnedSessions.filter(s => isProjectCwd(s.cwd)), [unpinnedSessions])

  const chatSessions = useMemo(() => unpinnedSessions.filter(s => !isProjectCwd(s.cwd)), [unpinnedSessions])

  const projectGroups = useMemo<SidebarSessionGroup[]>(
    () => (projectSessions.length ? workspaceGroupsFor(projectSessions, s.noWorkspace) : []),
    [projectSessions, s.noWorkspace]
  )

  useEffect(() => {
    const next = resolveManualSessionOrderIds(
      chatSessions.map(s => s.id),
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
  }, [agentOrderIds, agentOrderManual, chatSessions])

  const agentSessions = useMemo(
    () => (agentOrderManual ? orderByIds(chatSessions, s => s.id, agentOrderIds) : chatSessions),
    [chatSessions, agentOrderIds, agentOrderManual]
  )

  const loadMoreForProfileGroup = useCallback(
    (profile: string) => {
      if (!onLoadMoreProfileSessions) {
        return
      }

      setProfileLoadMorePending(prev => ({ ...prev, [profile]: true }))

      void Promise.resolve(onLoadMoreProfileSessions(profile))
        .catch(() => undefined)
        .finally(() => setProfileLoadMorePending(({ [profile]: _done, ...rest }) => rest))
    },
    [onLoadMoreProfileSessions]
  )

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
  const remainingSessionCount = Math.max(0, knownSessionTotal - loadedSessionCount)

  const recentsMeta = countLabel(agentSessions.length, knownSessionTotal)

  const displayAgentGroups = showAllProfiles ? profileGroups : undefined

  // The recents list owns its own (virtualized) scroll container only when it's a
  // long flat list. In that case it must keep its scroller even in short mode, so
  // we don't flatten it (flattening would defeat virtualization). Short flat lists
  // and grouped views (profile groups) flatten into the single outer scroll instead.
  const recentsVirtualizes = !displayAgentGroups?.length && displayAgentSessions.length >= VIRTUALIZE_THRESHOLD

  const showSessionSkeletons = sessionsLoading && sortedSessions.length === 0

  const showSessionSections = showSessionSkeletons || sortedSessions.length > 0

  // Sections gone (e.g. last session deleted) → the search field unmounts, so
  // drop the open state instead of leaving the 搜索 row stuck "active".
  useEffect(() => {
    if (searchOpen && !showSessionSections) {
      closeSearch()
    }
  }, [searchOpen, showSessionSections, closeSearch])

  // Each reorderable list reports its OWN new id order; persisting is a direct,
  // typed write — no id-prefix sniffing to figure out which level moved.
  const reorderSessions = (ids: string[]) => {
    setSidebarSessionOrderManual(true)
    setSidebarSessionOrderIds(ids)
  }

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
                  (item.id === 'cron' && currentView === 'cron') ||
                  (item.id === 'artifacts' && currentView === 'artifacts') ||
                  (item.id === 'search' && searchOpen)

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
                        'flex h-8 w-full justify-start gap-2.5 rounded-[0.625rem] border border-transparent px-2.5 text-left text-[0.8125rem] font-medium text-(--ui-text-secondary) transition-colors duration-100 ease-out [-webkit-app-region:no-drag] hover:bg-(--ui-control-hover-background) hover:text-foreground hover:transition-none',
                        active &&
                          'border-transparent bg-(--ui-row-active-background) text-foreground shadow-none hover:bg-(--ui-row-active-background)!',
                        !isInteractive &&
                          'cursor-default hover:border-transparent hover:bg-transparent hover:text-inherit'
                      )}
                      onClick={() => {
                        // 搜索 toggles the sidebar-local search field; it never
                        // routes anywhere, so don't hand it to onNavigate.
                        if (item.action === 'search') {
                          if (searchOpen) {
                            closeSearch()
                          } else {
                            openSearch()
                          }

                          return
                        }

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

        {contentVisible && showSessionSections && searchOpen && (
          <div
            className="shrink-0 px-2 pb-1 pt-1"
            onKeyDown={event => {
              // Esc backs out of search entirely (clears + hides the field).
              if (event.key === 'Escape') {
                event.stopPropagation()
                closeSearch()
              }
            }}
          >
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
                  <div className="grid min-h-24 place-items-center rounded-lg px-2 text-center text-xs text-(--ui-text-tertiary)">
                    {s.noMatch(trimmedQuery)}
                  </div>
                }
                label={s.results}
                labelMeta={String(searchResults.length)}
                onArchiveSession={onArchiveSession}
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

            {/* 项目 — Codex-style: one row per distinct workspace folder, that
                folder's chats nested beneath. Hidden entirely when no session
                runs in a project cwd. */}
            {!trimmedQuery && projectGroups.length > 0 && (
              <SidebarSessionsSection
                activeSessionId={activeSidebarSessionId}
                contentClassName={cn('flex max-h-72 flex-col gap-px rounded-lg pb-2 pt-1', GROUP_BODY)}
                emptyState={null}
                groups={projectGroups}
                label={s.projects}
                labelMeta={String(projectGroups.length)}
                onArchiveSession={onArchiveSession}
                onDeleteSession={onDeleteSession}
                onNewSessionInWorkspace={onNewSessionInWorkspace}
                onResumeSession={onResumeSession}
                onToggle={() => setSidebarProjectsOpen(!projectsOpen)}
                onTogglePin={pinSession}
                open={projectsOpen}
                pinned={false}
                rootClassName="shrink-0 p-0 pb-1"
                sessions={projectSessions}
                workingSessionIdSet={workingSessionIdSet}
              />
            )}

            {!trimmedQuery && (
              <SidebarSessionsSection
                activeSessionId={activeSidebarSessionId}
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
                emptyState={showSessionSkeletons ? <SidebarSessionSkeletons /> : <SidebarAllPinnedState />}
                footer={
                  // ALL-profiles pages per-profile from each profile header; the
                  // global footer only applies to non-ALL views.
                  !showAllProfiles && !showSessionSkeletons && hasMoreSessions ? (
                    <SidebarLoadMoreRow
                      loading={sessionsLoading}
                      onClick={onLoadMoreSessions}
                      step={Math.min(SIDEBAR_SESSIONS_PAGE_SIZE, remainingSessionCount)}
                    />
                  ) : null
                }
                forceEmptyState={showSessionSkeletons}
                groups={displayAgentGroups}
                label={s.sessions}
                labelMeta={recentsMeta}
                onArchiveSession={onArchiveSession}
                onDeleteSession={onDeleteSession}
                onNewSessionInWorkspace={showAllProfiles ? undefined : onNewSessionInWorkspace}
                onReorderSessions={showAllProfiles ? undefined : reorderSessions}
                onResumeSession={onResumeSession}
                onToggle={() => setSidebarRecentsOpen(!agentsOpen)}
                onTogglePin={pinSession}
                open={agentsOpen}
                pinned={false}
                rootClassName={cn(
                  'min-h-32 flex-1 overflow-hidden p-0',
                  !recentsVirtualizes && 'compact:min-h-0 compact:flex-none compact:overflow-visible'
                )}
                sessions={displayAgentSessions}
                sortable={!showAllProfiles && agentSessions.length > 1}
                workingSessionIdSet={workingSessionIdSet}
              />
            )}
          </div>
        )}

        {contentVisible && !showSessionSections && <div className="min-h-0 flex-1" />}

        {contentVisible && (
          <div className="shrink-0 px-0.5 pb-1 pt-0.5">
            {/* Codex-style bottom-left account row (avatar + name + email →
                popover menu). Renders only on managed builds when signed in; the
                auth gate covers the signed-out case. Profile management lives in
                the account menu (个人资料), so no separate profile rail. */}
            <AccountPanel />
          </div>
        )}
      </SidebarContent>
    </Sidebar>
  )
}

interface SidebarSectionHeaderProps {
  label: string
  open: boolean
  onToggle: () => void
  action?: React.ReactNode
  meta?: React.ReactNode
  icon?: React.ReactNode
}

function SidebarSectionHeader({ label, open, onToggle, action, meta, icon }: SidebarSectionHeaderProps) {
  return (
    <div className="group/section flex shrink-0 items-center justify-between pb-1 pt-1.5">
      <button
        className="group/section-label flex w-fit items-center gap-1 bg-transparent text-left leading-none"
        onClick={onToggle}
        type="button"
      >
        {icon}
        <SidebarPanelLabel>{label}</SidebarPanelLabel>
        {meta && <SidebarCount>{meta}</SidebarCount>}
        <DisclosureCaret
          className="text-(--ui-text-tertiary) opacity-0 transition group-hover/section-label:opacity-100"
          open={open}
        />
      </button>
      {action}
    </div>
  )
}

function SidebarSessionSkeletons() {
  return (
    <div aria-hidden="true" className="grid gap-px">
      {['w-32', 'w-40', 'w-28', 'w-36', 'w-24'].map((width, i) => (
        <div className="grid min-h-7 grid-cols-[minmax(0,1fr)_1.5rem] items-center rounded-lg" key={`${width}-${i}`}>
          <Skeleton className={cn('h-3.5 rounded-full', width)} />
          <Skeleton className="mx-auto size-4 rounded-md opacity-60" />
        </div>
      ))}
    </div>
  )
}

function SidebarAllPinnedState() {
  const { t } = useI18n()

  return (
    <div className="grid min-h-24 place-items-center rounded-lg text-center text-xs text-(--ui-text-tertiary)">
      {t.sidebar.allPinned}
    </div>
  )
}

function SidebarPinnedEmptyState() {
  const { t } = useI18n()

  return (
    <div className="flex min-h-7 items-center gap-1.5 rounded-lg pl-2 text-[0.75rem] text-(--ui-text-tertiary)">
      <span className="grid w-3.5 shrink-0 place-items-center text-(--ui-text-quaternary)">
        <Codicon name="pin" size="0.75rem" />
      </span>
      <span>{t.sidebar.shiftClickHint}</span>
    </div>
  )
}

interface SidebarSessionsSectionProps {
  label: string
  open: boolean
  onToggle: () => void
  sessions: SessionInfo[]
  activeSessionId: null | string
  workingSessionIdSet: Set<string>
  onResumeSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
  onArchiveSession: (sessionId: string) => void
  onTogglePin: (sessionId: string) => void
  onNewSessionInWorkspace?: (path: null | string) => void
  pinned: boolean
  rootClassName?: string
  contentClassName?: string
  emptyState: React.ReactNode
  forceEmptyState?: boolean
  headerAction?: React.ReactNode
  footer?: React.ReactNode
  groups?: SidebarSessionGroup[]
  labelMeta?: React.ReactNode
  labelIcon?: React.ReactNode
  sortable?: boolean
  // Optional; the flat session list is draggable iff its callback is supplied.
  onReorderSessions?: (ids: string[]) => void
  dndSensors?: ReturnType<typeof useSensors>
}

function SidebarSessionsSection({
  label,
  open,
  onToggle,
  sessions,
  activeSessionId,
  workingSessionIdSet,
  onResumeSession,
  onDeleteSession,
  onArchiveSession,
  onTogglePin,
  onNewSessionInWorkspace,
  pinned,
  rootClassName,
  contentClassName,
  emptyState,
  forceEmptyState = false,
  headerAction,
  footer,
  groups,
  labelMeta,
  labelIcon,
  sortable = false,
  onReorderSessions,
  dndSensors
}: SidebarSessionsSectionProps) {
  const hasGroupedSessions = Boolean(groups?.some(group => group.sessions.length > 0))
  const showEmptyState = forceEmptyState || (!hasGroupedSessions && sessions.length === 0)
  // The flat recents/pinned list is the only place sessions reorder by hand;
  // grouped views always sort by creation date and never drag.
  const sessionsDraggable = sortable && !!onReorderSessions

  const renderRow = (session: SessionInfo, draggable: boolean) => {
    const rowProps = {
      isPinned: pinned,
      isSelected: session.id === activeSessionId,
      isWorking: workingSessionIdSet.has(session.id),
      onArchive: () => onArchiveSession(session.id),
      onDelete: () => onDeleteSession(session.id),
      onPin: () => onTogglePin(sessionPinId(session)),
      onResume: () => onResumeSession(session.id),
      session
    }

    return draggable ? (
      <SortableSidebarSessionRow key={session.id} {...rowProps} />
    ) : (
      <SidebarSessionRow key={session.id} {...rowProps} />
    )
  }

  // Sessions inside project/profile groups are date-ordered and static.
  const renderRows = (items: SessionInfo[]) => items.map(session => renderRow(session, false))

  const flatVirtualized = !showEmptyState && !groups?.length && sessions.length >= VIRTUALIZE_THRESHOLD

  let inner: React.ReactNode

  if (showEmptyState) {
    inner = emptyState
  } else if (groups?.length) {
    // Project/profile groups never reorder; render them flat with static rows.
    inner = groups.map(group => (
      <SidebarWorkspaceGroup group={group} key={group.id} onNewSession={onNewSessionInWorkspace} renderRows={renderRows} />
    ))
  } else if (flatVirtualized) {
    const virtual = (
      <VirtualSessionList
        activeSessionId={activeSessionId}
        className={contentClassName}
        onArchiveSession={onArchiveSession}
        onDeleteSession={onDeleteSession}
        onResumeSession={onResumeSession}
        onTogglePin={onTogglePin}
        pinned={pinned}
        sessions={sessions}
        sortable={sessionsDraggable}
        workingSessionIdSet={workingSessionIdSet}
      />
    )

    inner =
      sessionsDraggable && onReorderSessions ? (
        <ReorderableList ids={sessions.map(s => s.id)} onReorder={onReorderSessions} sensors={dndSensors}>
          {virtual}
        </ReorderableList>
      ) : (
        virtual
      )
  } else if (sessionsDraggable && onReorderSessions) {
    inner = (
      <ReorderableList ids={sessions.map(s => s.id)} onReorder={onReorderSessions} sensors={dndSensors}>
        {sessions.map(session => renderRow(session, true))}
      </ReorderableList>
    )
  } else {
    inner = renderRows(sessions)
  }

  // The virtualizer owns its own scroller, so suppress the wrapper's overflow
  // to avoid a double scroll container.
  const resolvedContentClassName = cn(contentClassName, flatVirtualized && 'overflow-y-visible')

  return (
    <SidebarGroup className={rootClassName}>
      <SidebarSectionHeader
        action={headerAction}
        icon={labelIcon}
        label={label}
        meta={labelMeta}
        onToggle={onToggle}
        open={open}
      />
      {open && (
        <SidebarGroupContent className={resolvedContentClassName}>
          {inner}
          {footer}
        </SidebarGroupContent>
      )}
    </SidebarGroup>
  )
}

interface SidebarWorkspaceGroupProps extends React.ComponentProps<'div'> {
  group: SidebarSessionGroup
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  onNewSession?: (path: null | string) => void
  reorderable?: boolean
  dragging?: boolean
  dragHandleProps?: React.HTMLAttributes<HTMLElement>
}

function SidebarWorkspaceGroup({
  group,
  renderRows,
  onNewSession,
  reorderable = false,
  dragging = false,
  dragHandleProps,
  className,
  style,
  ref,
  ...rest
}: SidebarWorkspaceGroupProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const isProfileGroup = group.mode === 'profile'
  const pageStep = isProfileGroup ? PROFILE_INITIAL_PAGE : WORKSPACE_PAGE
  const [open, setOpen] = useState(true)
  const [visibleCount, setVisibleCount] = useState(pageStep)

  const loadedCount = group.sessions.length
  // Profile groups know their on-disk total (children excluded); workspace
  // groups only ever page within what's already loaded.
  const totalCount = isProfileGroup ? Math.max(group.totalCount ?? loadedCount, loadedCount) : loadedCount
  const visibleSessions = group.sessions.slice(0, visibleCount)
  const hiddenCount = Math.max(0, totalCount - visibleSessions.length)
  const nextCount = Math.min(pageStep, hiddenCount)

  // Leading glyph: profile color dot, or a folder mark for a project (项目)
  // row. When reorderable it doubles as the drag handle (icon ↔ grabber).
  const leadingIcon = group.color ? (
    <span aria-hidden="true" className="size-2 shrink-0 rounded-full" style={{ backgroundColor: group.color }} />
  ) : (
    <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="folder" size="0.75rem" />
  )

  // Reveal already-loaded rows first; only hit the backend when the next page
  // crosses what's been fetched for this profile.
  const handleProfileLoadMore = () => {
    const target = visibleCount + pageStep

    setVisibleCount(target)

    if (target > loadedCount && loadedCount < totalCount) {
      group.onLoadMore?.()
    }
  }

  return (
    <div
      className={cn(
        // While lifted, paint the opaque sidebar surface so the dragged group
        // erases the rows it floats over instead of ghosting them through a
        // translucent body.
        // minmax(0,1fr): pin the single column to the rail width. A bare `grid`
        // auto column sizes to the widest child's MAX-content (the full,
        // untruncated label), overflowing the rail so overflow-x-hidden clips the
        // +/grabber off-screen — the inner truncate never gets a bounded width.
        'grid grid-cols-[minmax(0,1fr)] gap-px data-[dragging=true]:z-10 data-[dragging=true]:rounded-md data-[dragging=true]:bg-(--ui-sidebar-surface-background) data-[dragging=true]:will-change-transform',
        className
      )}
      data-dragging={dragging ? 'true' : undefined}
      ref={ref}
      style={style}
      {...rest}
    >
      <WorkspaceHeader
        action={
          (onNewSession || isProfileGroup) && (
            <WorkspaceAddButton
              label={s.newSessionIn(group.label)}
              // Profile groups start a fresh session in that profile but keep the
              // all-profiles browse view (newSessionInProfile leaves the scope
              // alone); workspace groups seed the new session's cwd from the path.
              onClick={() => (isProfileGroup ? newSessionInProfile(group.id) : onNewSession?.(group.path))}
            />
          )
        }
        count={isProfileGroup ? countLabel(visibleSessions.length, totalCount) : group.sessions.length}
        dragging={dragging}
        dragHandleProps={dragHandleProps}
        icon={leadingIcon}
        label={group.label}
        onToggle={() => setOpen(value => !value)}
        open={open}
        reorderable={reorderable}
      />
      {open && (
        <>
          {renderRows(visibleSessions)}
          {hiddenCount > 0 &&
            (isProfileGroup ? (
              <SidebarLoadMoreRow
                loading={Boolean(group.loadingMore)}
                onClick={handleProfileLoadMore}
                step={nextCount}
              />
            ) : (
              <WorkspaceShowMoreButton
                count={nextCount}
                label={group.label}
                onClick={() => setVisibleCount(count => count + WORKSPACE_PAGE)}
              />
            ))}
        </>
      )}
    </div>
  )
}

function SidebarCount({ children }: { children: React.ReactNode }) {
  return <span className="text-[0.6875rem] font-medium text-(--ui-text-quaternary)">{children}</span>
}

// Reveals the next page of already-loaded rows within a project group.
function WorkspaceShowMoreButton({ count, label, onClick }: { count: number; label: string; onClick: () => void }) {
  const { t } = useI18n()
  const text = t.sidebar.showMoreIn(count, label)

  return (
    <Tip label={text}>
      <button
        aria-label={text}
        className="ml-auto grid size-5 place-items-center rounded-sm bg-transparent text-(--ui-text-tertiary) transition-colors hover:bg-(--ui-control-hover-background) hover:text-foreground"
        onClick={onClick}
        type="button"
      >
        <Codicon name="ellipsis" size="0.75rem" />
      </button>
    </Tip>
  )
}

// Reorder handle that lives in the header's leading-icon slot: the resting icon
// fades out and a grabber fades in on hover/drag (same swap as the session row),
// so the drag affordance never eats header width on the right.
function WorkspaceReorderHandle({
  dragHandleProps,
  dragging,
  icon,
  label
}: {
  dragHandleProps?: React.HTMLAttributes<HTMLElement>
  dragging: boolean
  icon: React.ReactNode
  label: string
}) {
  return (
    <span
      {...dragHandleProps}
      aria-label={label}
      className="group/handle relative -my-0.5 grid size-4 shrink-0 cursor-grab touch-none place-items-center self-stretch overflow-hidden active:cursor-grabbing"
      data-reorder-handle
      onClick={event => event.stopPropagation()}
    >
      <span
        className={cn(
          'grid place-items-center transition-opacity group-hover/handle:opacity-0 group-focus-within/handle:opacity-0',
          dragging && 'opacity-0'
        )}
      >
        {icon}
      </span>
      <Codicon
        className={cn(
          'absolute text-(--ui-text-quaternary) opacity-0 transition-opacity group-hover/handle:opacity-80 group-focus-within/handle:opacity-80 hover:text-(--ui-text-secondary)',
          dragging && 'text-(--ui-text-secondary) opacity-100'
        )}
        name="grabber"
        size="0.75rem"
      />
    </span>
  )
}

// "+" affordance on project/profile group headers — reveals on header hover.
function WorkspaceAddButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <Tip label={label}>
      <button
        aria-label={label}
        className="grid size-4 shrink-0 place-items-center rounded-sm bg-transparent text-(--ui-text-quaternary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/workspace:opacity-100"
        onClick={onClick}
        type="button"
      >
        <Codicon name="add" size="0.75rem" />
      </button>
    </Tip>
  )
}

// Collapsible group header (project folders, profile groups): a toggle button
// whose leading glyph doubles as the reorder handle, plus an optional trailing
// action (the +).
function WorkspaceHeader({
  action,
  count,
  dragHandleProps,
  dragging = false,
  emphasis = false,
  icon,
  label,
  onToggle,
  open,
  reorderable = false
}: {
  action?: React.ReactNode
  count: React.ReactNode
  dragHandleProps?: React.HTMLAttributes<HTMLElement>
  dragging?: boolean
  emphasis?: boolean
  icon: React.ReactNode
  label: string
  onToggle: () => void
  open: boolean
  reorderable?: boolean
}) {
  const { t } = useI18n()

  return (
    <div
      className={cn(
        'group/workspace flex min-h-6 items-center gap-1 px-2 pt-1 text-[0.6875rem]',
        emphasis ? 'font-semibold text-(--ui-text-secondary)' : 'font-medium text-(--ui-text-tertiary)'
      )}
    >
      <button
        className={cn(
          'flex min-w-0 flex-1 items-center gap-1.5 bg-transparent text-left',
          emphasis ? 'hover:text-foreground' : 'hover:text-(--ui-text-secondary)'
        )}
        onClick={onToggle}
        type="button"
      >
        {reorderable ? (
          <WorkspaceReorderHandle
            dragging={dragging}
            dragHandleProps={dragHandleProps}
            icon={icon}
            label={t.sidebar.reorderWorkspace(label)}
          />
        ) : (
          icon
        )}
        <span className="min-w-0 truncate">{label}</span>
        <span className="shrink-0">
          <SidebarCount>{count}</SidebarCount>
        </span>
        <DisclosureCaret
          className="shrink-0 text-(--ui-text-tertiary) opacity-0 transition group-hover/workspace:opacity-100"
          open={open}
        />
      </button>
      {action}
    </div>
  )
}

interface SortableSessionRowProps {
  session: SessionInfo
  isPinned: boolean
  isSelected: boolean
  isWorking: boolean
  onArchive: () => void
  onDelete: () => void
  onPin: () => void
  onResume: () => void
}

function SortableSidebarSessionRow(props: SortableSessionRowProps) {
  return <SidebarSessionRow {...props} {...useSortableBindings(props.session.id)} />
}
