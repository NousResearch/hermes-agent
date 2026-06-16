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
import { CSS } from '@dnd-kit/utilities'
import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { SessionDragPayload } from '@/app/chat/composer/inline-refs'
import { PlatformAvatar } from '@/app/messaging/platform-icon'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
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
import { normalizeSessionSource, sessionSourceLabel } from '@/lib/session-source'
import { cn } from '@/lib/utils'
import { $cronJobs } from '@/store/cron'
import {
  $panesFlipped,
  $pinnedSessionIds,
  $sidebarAgentsGrouped,
  $sidebarArchivedOpen,
  $sidebarCronOpen,
  $sidebarMessagingOpenIds,
  $sidebarOpen,
  $sidebarOverlayMounted,
  $sidebarPinsOpen,
  $sidebarRecentsOpen,
  $sidebarSessionOrderIds,
  $sidebarWorkspaceOrderIds,
  pinSession,
  reorderPinnedSession,
  SESSION_SEARCH_FOCUS_EVENT,
  setSidebarAgentsGrouped,
  setSidebarArchivedOpen,
  setSidebarCronOpen,
  setSidebarPinsOpen,
  setSidebarRecentsOpen,
  setSidebarSessionOrderIds,
  setSidebarWorkspaceOrderIds,
  SIDEBAR_SESSIONS_PAGE_SIZE,
  toggleSidebarMessagingOpen,
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
  $archivedSessions,
  $archivedSessionsLoading,
  $archivedSessionsTotal,
  $cronSessions,
  $messagingPlatformTotals,
  $messagingSessions,
  $messagingTruncated,
  $selectedStoredSessionId,
  $sessionProfileTotals,
  $sessions,
  $sessionsLoading,
  $sessionsTotal,
  $workingSessionIds,
  sessionPinId
} from '@/store/session'
import {
  $sidebarSelection,
  pruneSidebarSelection,
  rangeSelectSessions,
  type SidebarSectionKey,
  toggleSessionSelected
} from '@/store/sidebar-selection'

import { type AppView, ARTIFACTS_ROUTE, MESSAGING_ROUTE, SKILLS_ROUTE } from '../../routes'
import { SidebarPanelLabel } from '../../shell/sidebar-label'
import type { SidebarNavItem } from '../../types'

import { CloudChannelsDialog } from './cloud-channels-dialog'
import { SidebarCronJobsSection } from './cron-jobs-section'
import { SidebarLoadMoreRow } from './load-more-row'
import { ProfileRail } from './profile-switcher'
import { SelectionActionBar } from './selection-action-bar'
import { resolveSidebarSessionReleaseDrop } from './session-release-drop'
import { SidebarSessionRow } from './session-row'
import { type SessionDropAnchor, useSessionDropZone } from './use-session-drop-zone'
import { VirtualSessionList } from './virtual-session-list'

const VIRTUALIZE_THRESHOLD = 25

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
  {
    id: 'cloud-channels',
    label: '',
    icon: props => <Codicon name="cloud" {...props} />,
    action: 'cloud-channels'
  },
  { id: 'messaging', label: '', icon: props => <Codicon name="comment" {...props} />, route: MESSAGING_ROUTE },
  { id: 'artifacts', label: '', icon: props => <Codicon name="files" {...props} />, route: ARTIFACTS_ROUTE }
]

const WORKSPACE_PAGE = 5
// ALL-profiles view: show only the latest N per profile up front to keep the
// unified list scannable, then reveal/fetch more in N-sized steps on demand.
const PROFILE_INITIAL_PAGE = 5
const GROUP_DND_ID_PREFIX = 'group:'

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

const groupDndId = (id: string) => `${GROUP_DND_ID_PREFIX}${id}`

const parseGroupDndId = (id: string) =>
  id.startsWith(GROUP_DND_ID_PREFIX) ? id.slice(GROUP_DND_ID_PREFIX.length) : null

const countLabel = (loaded: number, total: number) => (total > loaded ? `${loaded}/${total}` : String(loaded))
const sessionTime = (s: SessionInfo) => s.last_active || s.started_at || 0

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

function reconcileOrderIds(currentIds: string[], orderIds: string[]): string[] {
  if (!currentIds.length) {
    return []
  }

  if (!orderIds.length) {
    return currentIds
  }

  const current = new Set(currentIds)
  const retained = orderIds.filter(id => current.has(id))
  const retainedSet = new Set(retained)

  // New ids (absent from the saved order) are the newest sessions/groups; keep
  // them ahead of the persisted order so fresh activity surfaces at the top of
  // the sidebar rather than being appended to the bottom.
  const fresh = currentIds.filter(id => !retainedSet.has(id))

  return [...fresh, ...retained]
}

function sameIds(left: string[], right: string[]) {
  return left.length === right.length && left.every((item, index) => item === right[index])
}

const baseName = (path: string) =>
  path
    .replace(/[/\\]+$/, '')
    .split(/[/\\]/)
    .filter(Boolean)
    .pop()

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

function workspaceGroupsFor(
  sessions: SessionInfo[],
  noWorkspaceLabel: string,
  options: { preserveSessionOrder?: boolean } = {}
): SidebarSessionGroup[] {
  const groups = new Map<string, SidebarSessionGroup>()

  for (const session of sessions) {
    const path = session.cwd?.trim() || ''
    const id = path || '__no_workspace__'
    const label = baseName(path) || path || noWorkspaceLabel

    const group = groups.get(id) ?? { id, label, path: path || null, sessions: [] }
    group.sessions.push(session)
    groups.set(id, group)
  }

  if (!options.preserveSessionOrder) {
    // Groups keep recency order (Map insertion = first-seen in the recency-sorted
    // input, so an active project floats up), but rows *within* a group sort by
    // creation time so they don't reshuffle every time a message lands — keeps
    // muscle memory intact.
    for (const group of groups.values()) {
      group.sessions.sort((a, b) => b.started_at - a.started_at)
    }
  }

  return [...groups.values()]
}

function useSortableBindings(id: string) {
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({ id })

  return {
    dragging: isDragging,
    dragHandleProps: { ...attributes, ...listeners },
    nativeDraggable: true as const,
    ref: setNodeRef,
    reorderable: true as const,
    style: {
      transform: CSS.Transform.toString(transform),
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
  onLoadMoreMessaging?: (platform: string) => Promise<void> | void
  onResumeSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
  onArchiveSession: (sessionId: string) => void
  onArchiveAllSessions: () => Promise<unknown> | void
  onNewSessionInWorkspace: (path: null | string) => void
  onManageCronJob: (jobId: string) => void
  onTriggerCronJob: (jobId: string) => void
  onEnsureArchivedLoaded?: () => void
  onLoadMoreArchived?: () => void
  onRestoreSession?: (sessionId: string) => void
  onArchiveSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onRestoreSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onDeleteSessions?: (sessionIds: string[]) => Promise<unknown> | void
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
  onArchiveAllSessions,
  onNewSessionInWorkspace,
  onManageCronJob,
  onTriggerCronJob,
  onEnsureArchivedLoaded,
  onLoadMoreArchived,
  onRestoreSession,
  onArchiveSessions,
  onRestoreSessions,
  onDeleteSessions
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
  const [cloudChannelsOpen, setCloudChannelsOpen] = useState(false)
  const [archiveAllOpen, setArchiveAllOpen] = useState(false)
  const [archiveAllSubmitting, setArchiveAllSubmitting] = useState(false)
  const archivedOpen = useStore($sidebarArchivedOpen)
  const archivedSessions = useStore($archivedSessions)
  const archivedTotal = useStore($archivedSessionsTotal)
  const archivedLoading = useStore($archivedSessionsLoading)
  const selection = useStore($sidebarSelection)
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
  const workspaceOrderIds = useStore($sidebarWorkspaceOrderIds)
  const [searchQuery, setSearchQuery] = useState('')
  const [serverMatches, setServerMatches] = useState<SessionSearchResult[]>([])
  const [newSessionKbdFlash, setNewSessionKbdFlash] = useState(false)
  const [profileLoadMorePending, setProfileLoadMorePending] = useState<Record<string, boolean>>({})
  const [messagingLoadMorePending, setMessagingLoadMorePending] = useState<Record<string, boolean>>({})
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

  const sortedSessions = useMemo(
    () => [...visibleSessions].sort((a, b) => sessionTime(b) - sessionTime(a)),
    [visibleSessions]
  )

  const workingSessionIdSet = useMemo(() => new Set(workingSessionIds), [workingSessionIds])

  // Index sessions by both their live id and their lineage-root id so a pin
  // stored as the pre-compression root resolves to the live continuation tip.
  const sessionByAnyId = useMemo(() => {
    const map = new Map<string, SessionInfo>()

    // Cron and messaging sessions are listed separately but can still be
    // pinned, so index them too — otherwise a pinned non-recents row can't
    // resolve into the Pinned section. Recents take precedence on collisions.
    for (const s of [...cronSessions, ...messagingSessions, ...visibleSessions]) {
      map.set(s.id, s)

      if (s._lineage_root_id && !map.has(s._lineage_root_id)) {
        map.set(s._lineage_root_id, s)
      }
    }

    return map
  }, [visibleSessions, cronSessions, messagingSessions])

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

  const anchorPinIdForSessionId = useCallback(
    (sessionId: string) => {
      const session = sessionByAnyId.get(sessionId)

      return session ? sessionPinId(session) : sessionId
    },
    [sessionByAnyId]
  )

  const resolveSessionRelease = useCallback(
    (sectionKey: 'archived' | 'pinned' | 'sessions', payload: SessionDragPayload, anchor: null | SessionDropAnchor) => {
      const decision = resolveSidebarSessionReleaseDrop({
        anchor,
        anchorPinIdForSessionId,
        payload,
        pinnedSessionIds,
        sectionKey,
        sessionOrderIds: agentOrderIds,
        showAllProfiles: showAllProfiles || agentsGrouped
      })

      switch (decision.type) {
        case 'archive':
          onArchiveSession(decision.sessionId)
          setSidebarArchivedOpen(true)
          onEnsureArchivedLoaded?.()

          break

        case 'open':
          if (decision.section === 'pinned') {
            setSidebarPinsOpen(true)
          } else {
            setSidebarRecentsOpen(true)
          }

          break

        case 'pin':
          pinSession(decision.pinId, decision.index)
          setSidebarPinsOpen(true)

          break

        case 'sessions':
          if (decision.restoreSessionId) {
            onRestoreSession?.(decision.restoreSessionId)
          }

          if (decision.unpinPinId) {
            unpinSession(decision.unpinPinId)
          }

          if (decision.nextOrder) {
            setSidebarSessionOrderIds(decision.nextOrder)
          }

          setSidebarRecentsOpen(true)

          break

        case 'none':
          break
      }
    },
    [
      agentOrderIds,
      agentsGrouped,
      anchorPinIdForSessionId,
      onArchiveSession,
      onEnsureArchivedLoaded,
      onRestoreSession,
      pinnedSessionIds,
      showAllProfiles
    ]
  )

  const pinnedDropZone = useSessionDropZone({
    accepts: flags => !flags.pinned && !flags.archived,
    onDropSession: (payload, _event, anchor) => resolveSessionRelease('pinned', payload, anchor)
  })

  const sessionsDropZone = useSessionDropZone({
    accepts: flags => flags.pinned || flags.archived,
    onDropSession: (payload, _event, anchor) => resolveSessionRelease('sessions', payload, anchor)
  })

  const archivedDropZone = useSessionDropZone({
    accepts: flags => !flags.archived,
    onDropSession: (payload, _event, anchor) => resolveSessionRelease('archived', payload, anchor)
  })

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

  const unpinnedAgentSessions = useMemo(
    () => sortedSessions.filter(s => !pinnedRealIdSet.has(s.id)),
    [sortedSessions, pinnedRealIdSet]
  )

  useEffect(() => {
    const next = reconcileOrderIds(
      unpinnedAgentSessions.map(s => s.id),
      agentOrderIds
    )

    if (!sameIds(next, agentOrderIds)) {
      setSidebarSessionOrderIds(next)
    }
  }, [agentOrderIds, unpinnedAgentSessions])

  const agentSessions = useMemo(
    () => orderByIds(unpinnedAgentSessions, s => s.id, agentOrderIds),
    [unpinnedAgentSessions, agentOrderIds]
  )

  // Recents are local-only: messaging-platform sessions are fetched as their
  // own slice ($messagingSessions) and rendered in self-managed per-platform
  // sections below, so there is no source-grouping magic to untangle here.
  const agentGroups = useMemo(
    () => orderByIds(workspaceGroupsFor(agentSessions, s.noWorkspace), g => g.id, workspaceOrderIds),
    [agentSessions, s.noWorkspace, workspaceOrderIds]
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

  const loadMoreForMessaging = useCallback(
    (platform: string) => {
      if (!onLoadMoreMessaging) {
        return
      }

      setMessagingLoadMorePending(prev => ({ ...prev, [platform]: true }))

      void Promise.resolve(onLoadMoreMessaging(platform))
        .catch(() => undefined)
        .finally(() => setMessagingLoadMorePending(({ [platform]: _done, ...rest }) => rest))
    },
    [onLoadMoreMessaging]
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

  const displayAgentSessions = agentSessions

  const selectionSessionsById = useMemo(() => {
    const map = new Map<string, SessionInfo>()

    for (const session of [
      ...archivedSessions,
      ...searchResults,
      ...cronSessions,
      ...messagingSessions,
      ...visibleSessions
    ]) {
      map.set(session.id, session)
    }

    return map
  }, [archivedSessions, cronSessions, messagingSessions, searchResults, visibleSessions])

  const selectionSessions = useMemo(
    () =>
      selection.ids
        .map(id => selectionSessionsById.get(id))
        .filter((session): session is SessionInfo => Boolean(session)),
    [selection.ids, selectionSessionsById]
  )

  useEffect(() => {
    const section = selection.section

    if (!section || !selection.ids.length) {
      return
    }

    let rows: SessionInfo[]

    switch (section) {
      case 'archived':
        rows = archivedSessions

        break

      case 'pinned':
        rows = pinnedSessions

        break

      case 'results':
        rows = searchResults

        break

      case 'sessions':
        rows = agentSessions

        break
      default: {
        const sourceId = section.slice('messaging:'.length)
        rows = messagingGroups.find(group => group.sourceId === sourceId)?.sessions ?? []
      }
    }

    pruneSidebarSelection(section, rows)
  }, [selection, agentSessions, archivedSessions, messagingGroups, pinnedSessions, searchResults])

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
  const archiveAllDisabled = sessionsLoading || agentSessions.length === 0 || archiveAllSubmitting

  const handleArchiveAll = async () => {
    if (archiveAllSubmitting) {
      return
    }

    setArchiveAllSubmitting(true)

    try {
      await onArchiveAllSessions()
      setArchiveAllOpen(false)
      onEnsureArchivedLoaded?.()
    } catch {
      // The caller owns the error toast/rollback; keep the dialog open.
    } finally {
      setArchiveAllSubmitting(false)
    }
  }

  const displayAgentGroups = showAllProfiles ? profileGroups : agentsGrouped ? agentGroups : undefined

  // The recents list owns its own (virtualized) scroll container only when it's a
  // long flat list. In that case it must keep its scroller even in short mode, so
  // we don't flatten it (flattening would defeat virtualization). Short flat lists
  // and grouped views flatten into the single outer scroll instead.
  const recentsVirtualizes = !displayAgentGroups?.length && displayAgentSessions.length >= VIRTUALIZE_THRESHOLD

  useEffect(() => {
    if (!displayAgentGroups?.length || showAllProfiles) {
      return
    }

    const next = reconcileOrderIds(
      displayAgentGroups.map(g => g.id),
      workspaceOrderIds
    )

    if (!sameIds(next, workspaceOrderIds)) {
      setSidebarWorkspaceOrderIds(next)
    }
  }, [displayAgentGroups, showAllProfiles, workspaceOrderIds])

  const showSessionSkeletons = sessionsLoading && sortedSessions.length === 0

  const showSessionSections = showSessionSkeletons || sortedSessions.length > 0

  const handlePinnedDragEnd = ({ active, over }: DragEndEvent) => {
    if (!over || active.id === over.id) {
      return
    }

    const newIndex = pinnedSessions.findIndex(s => s.id === String(over.id))

    if (newIndex < 0) {
      return
    }

    // Sortable ids are live session ids; the pinned store is keyed by durable
    // (lineage-root) ids, so translate before reordering.
    const dragged = sessionByAnyId.get(String(active.id))
    reorderPinnedSession(dragged ? sessionPinId(dragged) : String(active.id), newIndex)
  }

  const handleAgentDragEnd = ({ active, over }: DragEndEvent) => {
    if (!over || active.id === over.id) {
      return
    }

    const activeId = String(active.id)
    const overId = String(over.id)
    const activeGroup = parseGroupDndId(activeId)
    const overGroup = parseGroupDndId(overId)

    if (activeGroup && overGroup) {
      const groups = displayAgentGroups ?? []
      const oldIdx = groups.findIndex(g => g.id === activeGroup)
      const newIdx = groups.findIndex(g => g.id === overGroup)

      if (oldIdx < 0 || newIdx < 0) {
        return
      }

      setSidebarWorkspaceOrderIds(arrayMove(groups, oldIdx, newIdx).map(g => g.id))

      return
    }

    if (activeGroup || overGroup) {
      return
    }

    const oldIdx = agentSessions.findIndex(s => s.id === activeId)
    const newIdx = agentSessions.findIndex(s => s.id === overId)

    if (oldIdx < 0 || newIdx < 0) {
      return
    }

    setSidebarSessionOrderIds(arrayMove(agentSessions, oldIdx, newIdx).map(s => s.id))
  }

  return (
    <>
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
                  const isCloudChannels = item.id === 'cloud-channels'

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

                          if (isCloudChannels) {
                            setCloudChannelsOpen(true)

                            return
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
                    <div className="grid min-h-24 place-items-center rounded-lg px-2 text-center text-xs text-(--ui-text-tertiary)">
                      {s.noMatch(trimmedQuery)}
                    </div>
                  }
                  label={s.results}
                  labelMeta={String(searchResults.length)}
                  onArchiveSession={onArchiveSession}
                  onArchiveSessions={onArchiveSessions}
                  onDeleteSession={onDeleteSession}
                  onDeleteSessions={onDeleteSessions}
                  onRestoreSessions={onRestoreSessions}
                  onResumeSession={onResumeSession}
                  onToggle={() => undefined}
                  onTogglePin={pinSession}
                  open
                  pinned={false}
                  rootClassName="min-h-32 flex-1 overflow-hidden p-0"
                  sectionKey="results"
                  sessions={searchResults}
                  workingSessionIdSet={workingSessionIdSet}
                />
              )}

              {!trimmedQuery && (
                <SidebarSessionsSection
                  activeSessionId={activeSidebarSessionId}
                  contentClassName={cn('flex max-h-44 flex-col gap-px rounded-lg pb-2 pt-1', GROUP_BODY)}
                  dndSensors={dndSensors}
                  dropActive={pinnedDropZone.active}
                  dropHandlers={pinnedDropZone.dropHandlers}
                  emptyState={<SidebarPinnedEmptyState />}
                  label={s.pinned}
                  onArchiveSession={onArchiveSession}
                  onArchiveSessions={onArchiveSessions}
                  onDeleteSession={onDeleteSession}
                  onDeleteSessions={onDeleteSessions}
                  onReorder={handlePinnedDragEnd}
                  onRestoreSessions={onRestoreSessions}
                  onResumeSession={onResumeSession}
                  onToggle={() => setSidebarPinsOpen(!pinsOpen)}
                  onTogglePin={unpinSession}
                  open={pinsOpen}
                  pinned
                  rootClassName="shrink-0 p-0 pb-1"
                  sectionKey="pinned"
                  sessions={pinnedSessions}
                  sortable={pinnedSessions.length > 1}
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
                  dropActive={sessionsDropZone.active}
                  dropHandlers={sessionsDropZone.dropHandlers}
                  emptyState={showSessionSkeletons ? <SidebarSessionSkeletons /> : <SidebarAllPinnedState />}
                  footer={
                    // Hide "load more" only when workspace-grouped (those groups page
                    // themselves). ALL-profiles now pages per-profile from each profile
                    // header; the global footer only applies to non-ALL views.
                    !showAllProfiles && !agentsGrouped && !showSessionSkeletons && hasMoreSessions ? (
                      <SidebarLoadMoreRow
                        loading={sessionsLoading}
                        onClick={onLoadMoreSessions}
                        step={Math.min(SIDEBAR_SESSIONS_PAGE_SIZE, remainingSessionCount)}
                      />
                    ) : null
                  }
                  forceEmptyState={showSessionSkeletons}
                  groups={displayAgentGroups}
                  headerAction={
                    <div className="flex items-center gap-0.5">
                      <div className="grid size-6 shrink-0 place-items-center">
                        {!showAllProfiles && agentSessions.length > 0 ? (
                          <Tip label={s.archiveAllTitle}>
                            <Button
                              aria-label={s.archiveAllAria}
                              className="text-(--ui-text-tertiary) opacity-70 hover:bg-(--ui-control-hover-background) hover:text-foreground hover:opacity-100 focus-visible:opacity-100"
                              disabled={archiveAllDisabled}
                              onClick={event => {
                                event.stopPropagation()
                                setSidebarRecentsOpen(true)
                                setArchiveAllOpen(true)
                              }}
                              size="icon-xs"
                              variant="ghost"
                            >
                              <Codicon
                                name={archiveAllSubmitting ? 'loading' : 'archive'}
                                size="0.75rem"
                                spinning={archiveAllSubmitting}
                              />
                            </Button>
                          </Tip>
                        ) : null}
                      </div>
                      <div className="grid size-6 shrink-0 place-items-center">
                        {!showAllProfiles && agentSessions.length > 0 ? (
                          <Tip label={agentsGrouped ? s.groupTitleGrouped : s.groupTitleUngrouped}>
                            <Button
                              aria-label={agentsGrouped ? s.groupAriaGrouped : s.groupAriaUngrouped}
                              className={cn(
                                'text-(--ui-text-tertiary) opacity-70 hover:bg-(--ui-control-hover-background) hover:text-foreground hover:opacity-100 focus-visible:opacity-100',
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
                          </Tip>
                        ) : null}
                      </div>
                    </div>
                  }
                  label={s.sessions}
                  labelMeta={recentsMeta}
                  onArchiveSession={onArchiveSession}
                  onArchiveSessions={onArchiveSessions}
                  onDeleteSession={onDeleteSession}
                  onDeleteSessions={onDeleteSessions}
                  onNewSessionInWorkspace={showAllProfiles ? undefined : onNewSessionInWorkspace}
                  onReorder={showAllProfiles ? undefined : handleAgentDragEnd}
                  onRestoreSessions={onRestoreSessions}
                  onResumeSession={onResumeSession}
                  onToggle={() => setSidebarRecentsOpen(!agentsOpen)}
                  onTogglePin={pinSession}
                  open={agentsOpen}
                  pinned={false}
                  rootClassName={cn(
                    'min-h-32 flex-1 overflow-hidden p-0',
                    !recentsVirtualizes && 'compact:min-h-0 compact:flex-none compact:overflow-visible'
                  )}
                  sectionKey="sessions"
                  sessions={displayAgentSessions}
                  sortable={!showAllProfiles && agentSessions.length > 1}
                  workingSessionIdSet={workingSessionIdSet}
                />
              )}

              {!trimmedQuery &&
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
                      onArchiveSessions={onArchiveSessions}
                      onDeleteSession={onDeleteSession}
                      onDeleteSessions={onDeleteSessions}
                      onRestoreSessions={onRestoreSessions}
                      onResumeSession={onResumeSession}
                      onToggle={() => toggleSidebarMessagingOpen(group.sourceId)}
                      onTogglePin={pinSession}
                      open={messagingOpenIds.includes(group.sourceId)}
                      pinned={false}
                      rootClassName="shrink-0 p-0"
                      sectionKey={`messaging:${group.sourceId}`}
                      sessions={shownSessions}
                      workingSessionIdSet={workingSessionIdSet}
                    />
                  )
                })}

              {!trimmedQuery && cronJobs.length > 0 && (
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

              {!trimmedQuery && (archivedTotal > 0 || archivedSessions.length > 0) && (
                <SidebarSessionsSection
                  activeSessionId={activeSidebarSessionId}
                  archivedRows
                  contentClassName={cn('flex max-h-56 shrink-0 flex-col gap-px pb-1.75', GROUP_BODY)}
                  dropActive={archivedDropZone.active}
                  dropHandlers={archivedDropZone.dropHandlers}
                  emptyState={
                    archivedLoading ? (
                      <SidebarSessionSkeletons />
                    ) : (
                      <div className="flex min-h-7 items-center rounded-lg pl-2 text-[0.75rem] text-(--ui-text-tertiary)">
                        {s.archivedEmpty}
                      </div>
                    )
                  }
                  footer={
                    archivedTotal > archivedSessions.length ? (
                      <SidebarLoadMoreRow
                        loading={archivedLoading}
                        onClick={() => onLoadMoreArchived?.()}
                        step={Math.min(SIDEBAR_SESSIONS_PAGE_SIZE, archivedTotal - archivedSessions.length)}
                      />
                    ) : null
                  }
                  label={s.archived}
                  labelIcon={
                    <span className="grid w-4 shrink-0 place-items-center text-(--ui-text-quaternary)">
                      <Codicon name="archive" size="0.75rem" />
                    </span>
                  }
                  labelMeta={
                    archivedOpen
                      ? countLabel(archivedSessions.length, Math.max(archivedTotal, archivedSessions.length))
                      : String(Math.max(archivedTotal, archivedSessions.length))
                  }
                  onArchiveSession={onArchiveSession}
                  onArchiveSessions={onArchiveSessions}
                  onDeleteSession={onDeleteSession}
                  onDeleteSessions={onDeleteSessions}
                  onRestoreSession={onRestoreSession}
                  onRestoreSessions={onRestoreSessions}
                  onResumeSession={onResumeSession}
                  onToggle={() => {
                    const next = !archivedOpen
                    setSidebarArchivedOpen(next)

                    if (next) {
                      onEnsureArchivedLoaded?.()
                    }
                  }}
                  onTogglePin={pinSession}
                  open={archivedOpen}
                  pinned={false}
                  rootClassName="shrink-0 p-0"
                  sectionKey="archived"
                  sessions={archivedSessions}
                  workingSessionIdSet={workingSessionIdSet}
                />
              )}
            </div>
          )}

          {contentVisible && !showSessionSections && <div className="min-h-0 flex-1" />}

          {contentVisible && selection.ids.length > 0 && (
            <SelectionActionBar
              onArchiveSessions={ids => onArchiveSessions?.(ids)}
              onDeleteSessions={ids => onDeleteSessions?.(ids)}
              onRestoreSessions={ids => onRestoreSessions?.(ids)}
              sessions={selectionSessions}
            />
          )}

          {contentVisible && (
            <div className="shrink-0 px-0.5 pb-1 pt-0.5">
              <ProfileRail />
            </div>
          )}
        </SidebarContent>
      </Sidebar>
      <ArchiveAllSessionsDialog
        count={knownSessionTotal}
        onConfirm={handleArchiveAll}
        onOpenChange={setArchiveAllOpen}
        open={archiveAllOpen}
        submitting={archiveAllSubmitting}
      />
      <CloudChannelsDialog onOpenChange={setCloudChannelsOpen} open={cloudChannelsOpen} />
    </>
  )
}

interface ArchiveAllSessionsDialogProps {
  count: number
  open: boolean
  onConfirm: () => void | Promise<void>
  onOpenChange: (open: boolean) => void
  submitting: boolean
}

function ArchiveAllSessionsDialog({ count, open, onConfirm, onOpenChange, submitting }: ArchiveAllSessionsDialogProps) {
  const { t } = useI18n()
  const s = t.sidebar

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{s.archiveAllDialogTitle}</DialogTitle>
          <DialogDescription>{s.archiveAllDialogDesc}</DialogDescription>
        </DialogHeader>
        <div className="rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-3 py-2 text-xs text-(--ui-text-secondary)">
          {count > 0 ? s.archiveAllChecked(count) : s.archiveAllNone}
        </div>
        <DialogFooter>
          <Button disabled={submitting} onClick={() => onOpenChange(false)} type="button" variant="ghost">
            {s.archiveAllCancel}
          </Button>
          <Button disabled={submitting} onClick={() => void onConfirm()} type="button" variant="destructive">
            {submitting ? s.archiveAllSubmitting : s.archiveAllConfirm}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
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

interface SidebarSessionGroup {
  id: string
  label: string
  path: null | string
  sessions: SessionInfo[]
  // Profile color for the ALL-profiles view; absent for workspace groups.
  color?: null | string
  loadingMore?: boolean
  mode?: 'profile' | 'source' | 'workspace'
  onLoadMore?: () => void
  sourceId?: string
  totalCount?: number
}

interface MessagingSection {
  sourceId: string
  label: string
  sessions: SessionInfo[]
  total: number
  hasMore: boolean
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
  sectionKey?: SidebarSectionKey
  archivedRows?: boolean
  onRestoreSession?: (sessionId: string) => void
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
  onReorder?: (event: DragEndEvent) => void
  dndSensors?: ReturnType<typeof useSensors>
  dropActive?: boolean
  dropHandlers?: ReturnType<typeof useSessionDropZone>['dropHandlers']
  onArchiveSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onRestoreSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onDeleteSessions?: (sessionIds: string[]) => Promise<unknown> | void
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
  sectionKey,
  archivedRows = false,
  onRestoreSession,
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
  onReorder,
  dndSensors,
  dropActive = false,
  dropHandlers,
  onArchiveSessions,
  onRestoreSessions,
  onDeleteSessions
}: SidebarSessionsSectionProps) {
  const hasGroupedSessions = Boolean(groups?.some(group => group.sessions.length > 0))
  const showEmptyState = forceEmptyState || (!hasGroupedSessions && sessions.length === 0)
  const dndActive = sortable && !!onReorder
  const selection = useStore($sidebarSelection)
  const selectable = Boolean(sectionKey)
  const selectionActive = selectable && selection.section === sectionKey && selection.ids.length > 0

  const selectedIdSet = useMemo(
    () => (selectionActive ? new Set(selection.ids) : undefined),
    [selectionActive, selection.ids]
  )

  const orderedIds = useMemo(() => sessions.map(s => s.id), [sessions])

  const handleToggleSelect = useCallback(
    (sessionId: string, mode: 'range' | 'single') => {
      if (!sectionKey) {
        return
      }

      if (mode === 'range') {
        rangeSelectSessions(sectionKey, sessionId, orderedIds, activeSessionId)
      } else {
        toggleSessionSelected(sectionKey, sessionId)
      }
    },
    [activeSessionId, orderedIds, sectionKey]
  )

  const renderRow = (session: SessionInfo) => {
    const checked = selectedIdSet?.has(session.id) ?? false

    const rowProps = {
      archived: archivedRows,
      bulkSelectedSessionIds: checked && selection.ids.length > 1 ? selection.ids : undefined,
      checked,
      isPinned: pinned,
      isSelected: session.id === activeSessionId,
      isWorking: workingSessionIdSet.has(session.id),
      onArchive: () => onArchiveSession(session.id),
      onArchiveSelectedSessions: onArchiveSessions,
      onDelete: () => onDeleteSession(session.id),
      onDeleteSelectedSessions: onDeleteSessions,
      onPin: () => onTogglePin(sessionPinId(session)),
      onRestore: onRestoreSession ? () => onRestoreSession(session.id) : undefined,
      onRestoreSelectedSessions: onRestoreSessions,
      onResume: () => onResumeSession(session.id),
      onToggleSelect: sectionKey ? (mode: 'range' | 'single') => handleToggleSelect(session.id, mode) : undefined,
      selectable,
      selectionActive,
      session
    }

    return sortable ? (
      <SortableSidebarSessionRow key={session.id} {...rowProps} />
    ) : (
      <SidebarSessionRow key={session.id} {...rowProps} />
    )
  }

  const renderRows = (items: SessionInfo[]) => items.map(renderRow)

  const renderSessionList = (items: SessionInfo[]) =>
    dndActive ? (
      <SortableContext items={items.map(s => s.id)} strategy={verticalListSortingStrategy}>
        {renderRows(items)}
      </SortableContext>
    ) : (
      renderRows(items)
    )

  const renderNestedSessionList = (items: SessionInfo[]) =>
    dndActive ? (
      <DndContext collisionDetection={closestCenter} onDragEnd={onReorder} sensors={dndSensors}>
        <SortableContext items={items.map(s => s.id)} strategy={verticalListSortingStrategy}>
          {renderRows(items)}
        </SortableContext>
      </DndContext>
    ) : (
      renderRows(items)
    )

  const flatVirtualized = !showEmptyState && !groups?.length && sessions.length >= VIRTUALIZE_THRESHOLD

  let inner: React.ReactNode
  let bodyOwnsDndContext = dndActive && !showEmptyState

  if (showEmptyState) {
    inner = emptyState
    bodyOwnsDndContext = false
  } else if (groups?.length) {
    const groupNodes = groups.map(group =>
      dndActive ? (
        <SortableSidebarWorkspaceGroup
          group={group}
          key={group.id}
          onNewSession={onNewSessionInWorkspace}
          renderRows={renderNestedSessionList}
        />
      ) : (
        <SidebarWorkspaceGroup
          group={group}
          key={group.id}
          onNewSession={onNewSessionInWorkspace}
          renderRows={renderSessionList}
        />
      )
    )

    inner = dndActive ? (
      <DndContext collisionDetection={closestCenter} onDragEnd={onReorder} sensors={dndSensors}>
        <SortableContext items={groups.map(g => groupDndId(g.id))} strategy={verticalListSortingStrategy}>
          {groupNodes}
        </SortableContext>
      </DndContext>
    ) : (
      groupNodes
    )
    bodyOwnsDndContext = false
  } else if (flatVirtualized) {
    inner = (
      <VirtualSessionList
        activeSessionId={activeSessionId}
        archived={archivedRows}
        className={contentClassName}
        dropActive={dropActive}
        onArchiveSession={onArchiveSession}
        onArchiveSessions={onArchiveSessions}
        onDeleteSession={onDeleteSession}
        onDeleteSessions={onDeleteSessions}
        onRestoreSession={onRestoreSession}
        onRestoreSessions={onRestoreSessions}
        onResumeSession={onResumeSession}
        onTogglePin={onTogglePin}
        onToggleSelect={sectionKey ? handleToggleSelect : undefined}
        pinned={pinned}
        sectionKey={sectionKey}
        selectable={selectable}
        selectedIds={selectedIdSet}
        selectedSessionIds={selection.ids}
        selectionActive={selectionActive}
        sessions={sessions}
        sortable={sortable}
        workingSessionIdSet={workingSessionIdSet}
      />
    )
  } else {
    inner = renderSessionList(sessions)
  }

  const body = bodyOwnsDndContext ? (
    <DndContext collisionDetection={closestCenter} onDragEnd={onReorder} sensors={dndSensors}>
      {inner}
    </DndContext>
  ) : (
    inner
  )

  // The virtualizer owns its own scroller, so suppress the wrapper's overflow
  // to avoid a double scroll container.
  const resolvedContentClassName = cn(contentClassName, flatVirtualized && 'overflow-y-visible')

  return (
    <SidebarGroup
      className={cn(
        rootClassName,
        dropActive && 'rounded-lg bg-(--ui-control-hover-background) ring-1 ring-inset ring-(--ui-stroke-tertiary)'
      )}
      {...dropHandlers}
    >
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
          {body}
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
  const isSourceGroup = group.mode === 'source'
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
        'grid gap-px data-[dragging=true]:z-10 data-[dragging=true]:opacity-70 data-[dragging=true]:will-change-transform',
        className
      )}
      data-dragging={dragging ? 'true' : undefined}
      ref={ref}
      style={style}
      {...rest}
    >
      <div className="group/workspace flex min-h-6 items-center gap-1 px-2 pt-1 text-[0.6875rem] font-medium text-(--ui-text-tertiary)">
        <button
          className="flex min-w-0 items-center gap-1.5 bg-transparent text-left hover:text-(--ui-text-secondary)"
          onClick={() => setOpen(value => !value)}
          type="button"
        >
          {group.color ? (
            <span
              aria-hidden="true"
              className="size-2 shrink-0 rounded-full"
              style={{ backgroundColor: group.color }}
            />
          ) : null}
          {isSourceGroup && group.sourceId ? (
            <PlatformAvatar
              className="size-4 rounded-[4px] text-[0.5625rem] [&_svg]:size-3"
              platformId={group.sourceId}
              platformName={group.label}
            />
          ) : null}
          <span className="truncate">{group.label}</span>
          <SidebarCount>
            {isProfileGroup ? countLabel(visibleSessions.length, totalCount) : group.sessions.length}
          </SidebarCount>
          <DisclosureCaret
            className="text-(--ui-text-tertiary) opacity-0 transition group-hover/workspace:opacity-100"
            open={open}
          />
        </button>
        {(onNewSession || isProfileGroup) && (
          <Tip label={s.newSessionIn(group.label)}>
            <button
              aria-label={s.newSessionIn(group.label)}
              className="grid size-4 shrink-0 place-items-center rounded-sm bg-transparent text-(--ui-text-quaternary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground group-hover/workspace:opacity-100"
              // Profile groups start a fresh session in that profile but keep the
              // all-profiles browse view (newSessionInProfile leaves the scope
              // alone); workspace groups seed the new session's cwd from the path.
              onClick={() => (isProfileGroup ? newSessionInProfile(group.id) : onNewSession?.(group.path))}
              type="button"
            >
              <Codicon name="add" size="0.75rem" />
            </button>
          </Tip>
        )}
        {reorderable && (
          <span
            {...dragHandleProps}
            aria-label={s.reorderWorkspace(group.label)}
            className="ml-auto -my-0.5 grid w-4 shrink-0 cursor-grab touch-none place-items-center self-stretch overflow-hidden active:cursor-grabbing"
            onClick={event => event.stopPropagation()}
          >
            <Codicon
              className={cn(
                'text-(--ui-text-quaternary) opacity-0 transition-opacity group-hover/workspace:opacity-80 hover:text-(--ui-text-secondary)',
                dragging && 'text-(--ui-text-secondary) opacity-100'
              )}
              name="grabber"
              size="0.75rem"
            />
          </span>
        )}
      </div>
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
              <Tip label={s.showMoreIn(nextCount, group.label)}>
                <button
                  aria-label={s.showMoreIn(nextCount, group.label)}
                  className="ml-auto grid size-5 place-items-center rounded-sm bg-transparent text-(--ui-text-tertiary) transition-colors hover:bg-(--ui-control-hover-background) hover:text-foreground"
                  onClick={() => setVisibleCount(count => count + WORKSPACE_PAGE)}
                  type="button"
                >
                  <Codicon name="ellipsis" size="0.75rem" />
                </button>
              </Tip>
            ))}
        </>
      )}
    </div>
  )
}

interface SortableWorkspaceProps {
  group: SidebarSessionGroup
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  onNewSession?: (path: null | string) => void
}

function SortableSidebarWorkspaceGroup(props: SortableWorkspaceProps) {
  return <SidebarWorkspaceGroup {...props} {...useSortableBindings(groupDndId(props.group.id))} />
}

function SidebarCount({ children }: { children: React.ReactNode }) {
  return <span className="text-[0.6875rem] font-medium text-(--ui-text-quaternary)">{children}</span>
}

type SortableSessionRowProps = React.ComponentProps<typeof SidebarSessionRow>

function SortableSidebarSessionRow(props: SortableSessionRowProps) {
  return <SidebarSessionRow {...props} {...useSortableBindings(props.session.id)} />
}
