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
import { profileColor } from '@/lib/profile-color'
import { sessionMatchesSearch } from '@/lib/session-search'
import { normalizeSessionSource, sessionSourceLabel } from '@/lib/session-source'
import { cn } from '@/lib/utils'
import { $cronJobs } from '@/store/cron'
import {
  $panesFlipped,
  $pinnedSessionIds,
  $sidebarArchivedOpen,
  $sidebarCronOpen,
  $sidebarOpen,
  $sidebarOverlayMounted,
  $sidebarPinsOpen,
  $sidebarRecentsOpen,
  $sidebarSessionOrderIds,
  pinSession,
  SESSION_SEARCH_FOCUS_EVENT,
  setSidebarArchivedOpen,
  setSidebarCronOpen,
  setSidebarPinsOpen,
  setSidebarRecentsOpen,
  setSidebarSessionOrderIds,
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
import type { SidebarNavItem } from '../../types'

import { CloudChannelsDialog } from './cloud-channels-dialog'
import { SidebarCronJobsSection } from './cron-jobs-section'
import { SidebarLoadMoreRow } from './load-more-row'
import { ProfileRail } from './profile-switcher'
import { SidebarRemoteSessionsSection } from './remote-sessions-section'
import { SidebarCount, SidebarSectionHeader } from './section-header'
import { SelectionActionBar } from './selection-action-bar'
import { SidebarSessionRow } from './session-row'
import {
  placeSessionIdAtAnchor,
  previewItemsAtAnchor,
  type SessionDropAnchor,
  useSessionDropZone
} from './use-session-drop-zone'
import { VirtualSessionList } from './virtual-session-list'

const VIRTUALIZE_THRESHOLD = 25

// Render the modifier key the user actually presses on this platform. The
// global accelerator is bound to both Cmd+N (macOS) and Ctrl+N (everywhere
// else) in desktop-controller.tsx, but the hint should match muscle memory.
const NEW_SESSION_KBD: readonly string[] =
  typeof navigator !== 'undefined' && navigator.platform.toLowerCase().includes('mac') ? ['⌘', 'N'] : ['Ctrl', 'N']

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

// ALL-profiles view: show only the latest N per profile up front to keep the
// unified list scannable, then reveal/fetch more in N-sized steps on demand.
const PROFILE_INITIAL_PAGE = 5

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
  onLoadMoreSessions: () => void
  onLoadMoreProfileSessions?: (profile: string) => Promise<void> | void
  onLoadMoreMessaging?: (platform: string) => Promise<void> | void
  onResumeSession: (sessionId: string) => void
  onCreateOnDevice: (endpoint: string) => void
  onDeleteSession: (sessionId: string) => void
  onArchiveSession: (sessionId: string) => void
  onArchiveAllSessions: () => Promise<void> | void
  onNewSessionInWorkspace: (path: null | string) => void
  onManageCronJob: (jobId: string) => void
  onTriggerCronJob: (jobId: string) => void
  /** Archived section: fetch the row page on first expand / page further. */
  onEnsureArchivedLoaded?: () => void
  onLoadMoreArchived?: () => void
  onRestoreSession?: (sessionId: string) => void
  /** Bulk verbs for the multi-select action bar. */
  onArchiveSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onRestoreSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onDeleteSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onPromptSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
  onSteerSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
  onHaltSessions?: (sessionIds: string[]) => Promise<unknown> | void
}

export function ChatSidebar({
  currentView,
  onNavigate,
  onLoadMoreSessions,
  onLoadMoreProfileSessions,
  onLoadMoreMessaging,
  onResumeSession,
  onCreateOnDevice,
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
  onDeleteSessions,
  onPromptSessions,
  onSteerSessions,
  onHaltSessions
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
  const agentsOpen = useStore($sidebarRecentsOpen)
  const cronOpen = useStore($sidebarCronOpen)
  const selectedSessionId = useStore($selectedStoredSessionId)
  const sessions = useStore($sessions)
  const cronSessions = useStore($cronSessions)
  const cronJobs = useStore($cronJobs)
  const [remoteOpen, setRemoteOpen] = useState(true)
  const [archiveAllOpen, setArchiveAllOpen] = useState(false)
  const [archiveAllSubmitting, setArchiveAllSubmitting] = useState(false)
  const [cloudChannelsOpen, setCloudChannelsOpen] = useState(false)
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
  const [searchQuery, setSearchQuery] = useState('')
  const [serverMatches, setServerMatches] = useState<SessionSearchResult[]>([])
  const [newSessionKbdFlash, setNewSessionKbdFlash] = useState(false)
  const [profileLoadMorePending, setProfileLoadMorePending] = useState<Record<string, boolean>>({})
  const [messagingLoadMorePending, setMessagingLoadMorePending] = useState<Record<string, boolean>>({})
  const [messagingOpen, setMessagingOpen] = useState<Record<string, boolean>>({})
  const [draggingSession, setDraggingSession] = useState<null | SessionDragPayload>(null)
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

    // Cron and messaging sessions are listed in their own sections but can
    // still be pinned (menu, shift-click, or drag), so index them too —
    // otherwise their pins resolve to nothing and silently never render in
    // the Pinned section. Recents take precedence on id collisions (set last).
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

  const dragPreviewSessionById = useMemo(() => {
    const map = new Map(sessionByAnyId)

    for (const session of archivedSessions) {
      map.set(session.id, session)
    }

    return map
  }, [archivedSessions, sessionByAnyId])

  const dragPreviewSession = draggingSession ? (dragPreviewSessionById.get(draggingSession.id) ?? null) : null
  const handleSessionDragStart = useCallback((payload: SessionDragPayload) => setDraggingSession(payload), [])
  const handleSessionDragEnd = useCallback(() => setDraggingSession(null), [])

  // Whole session rows are native-draggable (the same drag that drops a
  // session into the composer), so Pinned/Sessions accept that drag directly:
  // same-section drops reorder, cross-section drops pin/unpin/restore, and
  // row-edge anchors choose the exact insertion position. Header/empty-space
  // drops keep the old fallback behavior.
  const pinnedDropZone = useSessionDropZone({
    // Archived rows can't pin — the Pinned section resolves pins against the
    // live (unarchived) slices, so accepting one would pin into the void.
    accepts: flags => !flags.archived,
    draggingSessionId: draggingSession?.id,
    onDropSession: (payload, _event, anchor) => {
      // Translate the anchor row into an index in the RAW pinned-id store:
      // rendered rows can be a subset of stored ids (a pin whose session
      // isn't loaded renders nothing), so indexOf the anchor's durable pin id
      // rather than trusting the rendered position.
      const payloadPinId = payload.pinId ?? payload.id
      let index: number | undefined

      if (anchor) {
        const anchorSession = sessionByAnyId.get(anchor.sessionId)
        const anchorPinId = anchorSession ? sessionPinId(anchorSession) : anchor.sessionId

        const nextPinnedIds = placeSessionIdAtAnchor(pinnedSessionIds, payloadPinId, {
          before: anchor.before,
          sessionId: anchorPinId
        })

        if (nextPinnedIds) {
          index = nextPinnedIds.indexOf(payloadPinId)
        } else if (anchorPinId === payloadPinId) {
          setSidebarPinsOpen(true)

          return
        }
      }

      pinSession(payloadPinId, index)
      setSidebarPinsOpen(true)
    }
  })

  const sessionsDropZone = useSessionDropZone({
    // Any session row can land in Sessions: pinned rows unpin, archived rows
    // restore, and already-live rows reorder by the row edge under the pointer.
    accepts: () => true,
    draggingSessionId: draggingSession?.id,
    onDropSession: (payload, _event, anchor) => {
      if (payload.archived) {
        onRestoreSession?.(payload.id)
      } else if (payload.pinned) {
        unpinSession(payload.pinId ?? payload.id)
      }

      // Positional drop applies only to the flat list — grouped and
      // ALL-profiles views derive row order themselves. With no usable anchor
      // (header drop, or the anchor is a fresh row the saved order hasn't
      // reconciled yet) the id is simply left out of the saved order and the
      // reconcile effect surfaces it at the top, the pre-positional behavior.
      if (!showAllProfiles) {
        const nextOrder = placeSessionIdAtAnchor(
          reconcileOrderIds(
            agentSessions.map(s => s.id),
            agentOrderIds
          ),
          payload.id,
          anchor
        )

        if (nextOrder) {
          setSidebarSessionOrderIds(nextOrder)
        } else if (anchor?.sessionId === payload.id) {
          setSidebarRecentsOpen(true)

          return
        }
      }

      setSidebarRecentsOpen(true)
    }
  })

  // The Archived section is itself a drop target: dragging any live row onto
  // it archives the row — the spatial mirror of dragging an archived row onto
  // Sessions to restore it.
  const archivedDropZone = useSessionDropZone({
    accepts: flags => !flags.archived,
    draggingSessionId: draggingSession?.id,
    onDropSession: payload => {
      onArchiveSession(payload.id)
      setSidebarArchivedOpen(true)
      onEnsureArchivedLoaded?.()
    }
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

    return (
      [...bySource.entries()]
        .map(([sourceId, list]) => {
          const ordered = [...list].sort((a, b) => sessionTime(b) - sessionTime(a))
          // Pinning MOVES a row to the Pinned section (matching how recents
          // disappear from Sessions when pinned) — don't render it here too.
          // Pinned rows still count as LOADED for the load-more math: they're
          // platform conversations already in memory, just housed elsewhere, so
          // hiding them must not make the pager think more remain on disk.
          const rows = ordered.filter(s => !pinnedRealIdSet.has(s.id))
          const known = messagingPlatformTotals[sourceId]
          const total = Math.max(ordered.length, known ?? 0)

          return {
            // Known exact total → more exist iff total exceeds loaded; otherwise
            // the seed fetch was capped, so assume more until a per-platform load
            // resolves the count.
            hasMore: known != null ? known > ordered.length : messagingTruncated,
            label: sessionSourceLabel(sourceId) ?? sourceId,
            // Section recency comes from every loaded row (pinned included) so a
            // platform doesn't reshuffle when its newest thread gets pinned.
            latestActivity: sessionTime(ordered[0]),
            loadedCount: ordered.length,
            sessions: rows,
            sourceId,
            total
          }
        })
        // A platform whose every loaded row is pinned (and with nothing more on
        // disk) has nothing left to show — drop the empty shell. Kept when more
        // rows exist so its pager stays reachable.
        .filter(group => group.sessions.length > 0 || group.hasMore)
        .sort((a, b) => b.latestActivity - a.latestActivity)
    )
  }, [messagingSessions, messagingPlatformTotals, messagingTruncated, pinnedRealIdSet])

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

  // Drop selected ids whose rows left their section (archived elsewhere,
  // deleted on another device, paged out) so the bar's count stays honest.
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

    // Rows (not bare ids) so the prune can remap compression-rotated ids and
    // ignore transient empty lists instead of resetting the selection.
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

  // The server total counts every listable local conversation — including ones
  // currently pinned, which this section deliberately doesn't render. Pinned
  // rows are always loaded (the refresh keep-set preserves them), so drop them
  // from the label's BOTH sides; otherwise pinning one row leaves the count
  // stuck at "17/18" forever, advertising a page that can never arrive.
  const pinnedFromRecentsCount = useMemo(
    () => visibleSessions.reduce((count, s) => (pinnedRealIdSet.has(s.id) ? count + 1 : count), 0),
    [visibleSessions, pinnedRealIdSet]
  )

  const agentKnownTotal = Math.max(knownSessionTotal - pinnedFromRecentsCount, agentSessions.length)

  const recentsMeta = countLabel(agentSessions.length, agentKnownTotal)
  const archiveAllDisabled = sessionsLoading || agentSessions.length === 0 || archiveAllSubmitting

  const handleArchiveAll = async () => {
    if (archiveAllSubmitting) {
      return
    }

    setArchiveAllSubmitting(true)

    try {
      await onArchiveAllSessions()
      setArchiveAllOpen(false)
    } catch {
      // The caller owns the error toast/rollback; keep the dialog open.
    } finally {
      setArchiveAllSubmitting(false)
    }
  }

  const displayAgentGroups = showAllProfiles ? profileGroups : undefined

  const showSessionSkeletons = sessionsLoading && sortedSessions.length === 0

  const showSessionSections = showSessionSkeletons || sortedSessions.length > 0

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
                          // Inset ring (not border) for the active state: a border
                          // eats 1px of padding and pushes the icon column off the
                          // grid shared with the section dots / row dots below.
                          'flex h-7 w-full justify-start gap-1.5 rounded-md px-2 text-left text-[0.8125rem] font-medium text-(--ui-text-secondary) transition-colors duration-100 ease-out hover:bg-(--ui-control-hover-background) hover:text-foreground hover:transition-none',
                          active &&
                            'bg-(--ui-control-active-background) text-foreground shadow-none ring-1 ring-inset ring-(--ui-stroke-tertiary)',
                          !isInteractive && 'cursor-default hover:bg-transparent hover:text-inherit'
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
                        {/* Same w-3.5 leading slot as the section headers and
                          session rows — one glyph column for the whole rail. */}
                        <span className="grid w-3.5 shrink-0 place-items-center">
                          <item.icon className="size-4 shrink-0 text-[color-mix(in_srgb,currentColor_72%,transparent)]" />
                        </span>
                        {contentVisible && (
                          <>
                            <span className="min-w-0 flex-1 truncate">{s.nav[item.id] ?? item.label}</span>
                            {isNewSession && (
                              <KbdGroup
                                className={cn('ml-auto', newSessionKbdFlash && 'opacity-100!')}
                                keys={[...NEW_SESSION_KBD]}
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
            // px-1.5 + the field's own px-0.5 put the search glyph on the shared
            // leading column and the typed text on the rows' title edge.
            <div className="shrink-0 px-1.5 pb-1 pt-1">
              <SearchField
                aria-label={s.searchAria}
                inputRef={searchInputRef}
                onChange={setSearchQuery}
                placeholder={s.searchPlaceholder}
                value={searchQuery}
              />
            </div>
          )}

          {contentVisible && showSessionSections && trimmedQuery && (
            <SidebarSessionsSection
              activeSessionId={activeSidebarSessionId}
              contentClassName="flex min-h-0 flex-1 flex-col gap-px overflow-y-auto overscroll-contain pb-1.75"
              draggingSession={draggingSession}
              dragPreviewSession={dragPreviewSession}
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
              onHaltSessions={onHaltSessions}
              onPromptSessions={onPromptSessions}
              onRestoreSessions={onRestoreSessions}
              onResumeSession={onResumeSession}
              onSessionDragEnd={handleSessionDragEnd}
              onSessionDragStart={handleSessionDragStart}
              onSteerSessions={onSteerSessions}
              onToggle={() => undefined}
              onTogglePin={pinSession}
              open
              pinned={false}
              rootClassName="min-h-0 flex-1 p-0"
              sectionKey="results"
              sessions={searchResults}
              workingSessionIdSet={workingSessionIdSet}
            />
          )}

          {contentVisible && showSessionSections && !trimmedQuery && (
            <SidebarSessionsSection
              activeSessionId={activeSidebarSessionId}
              contentClassName="flex min-h-10 shrink-0 flex-col gap-px rounded-lg pb-2 pt-1"
              draggingSession={draggingSession}
              dragPreviewSession={dragPreviewSession}
              dropActive={pinnedDropZone.active}
              dropAnchor={pinnedDropZone.anchor}
              dropHandlers={pinnedDropZone.dropHandlers}
              emptyState={<SidebarPinnedEmptyState />}
              label={s.pinned}
              onArchiveSession={onArchiveSession}
              onArchiveSessions={onArchiveSessions}
              onDeleteSession={onDeleteSession}
              onDeleteSessions={onDeleteSessions}
              onHaltSessions={onHaltSessions}
              onPromptSessions={onPromptSessions}
              onRestoreSessions={onRestoreSessions}
              onResumeSession={onResumeSession}
              onSessionDragEnd={handleSessionDragEnd}
              onSessionDragStart={handleSessionDragStart}
              onSteerSessions={onSteerSessions}
              onToggle={() => setSidebarPinsOpen(!pinsOpen)}
              onTogglePin={unpinSession}
              open={pinsOpen}
              pinned
              rootClassName="shrink-0 p-0 pb-1"
              sectionKey="pinned"
              sessions={pinnedSessions}
              workingSessionIdSet={workingSessionIdSet}
            />
          )}

          {contentVisible && showSessionSections && !trimmedQuery && (
            <SidebarSessionsSection
              activeSessionId={activeSidebarSessionId}
              contentClassName={cn(
                'flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain pb-1.75',
                // Separate profile sections clearly in the ALL view; rows inside
                // each group keep their own tight gap-px rhythm.
                showAllProfiles ? 'gap-3' : 'gap-px'
              )}
              draggingSession={draggingSession}
              dragPreviewSession={dragPreviewSession}
              dropActive={sessionsDropZone.active}
              dropAnchor={sessionsDropZone.anchor}
              dropHandlers={sessionsDropZone.dropHandlers}
              emptyState={showSessionSkeletons ? <SidebarSessionSkeletons /> : <SidebarAllPinnedState />}
              footer={
                // ALL-profiles pages per-profile from each profile header; the
                // global footer only applies to the flat recents view.
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
              headerAction={
                <div className="grid size-5 shrink-0 place-items-center pr-0.5">
                  {!showAllProfiles && agentSessions.length > 0 ? (
                    <Tip label={s.archiveAllTitle}>
                      <Button
                        aria-label={s.archiveAllAria}
                        className="size-5 text-(--ui-text-tertiary) opacity-70 hover:bg-(--ui-control-hover-background) hover:text-foreground hover:opacity-100 focus-visible:opacity-100"
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
              }
              label={s.sessions}
              labelMeta={recentsMeta}
              onArchiveSession={onArchiveSession}
              onArchiveSessions={onArchiveSessions}
              onDeleteSession={onDeleteSession}
              onDeleteSessions={onDeleteSessions}
              onHaltSessions={onHaltSessions}
              onNewSessionInWorkspace={showAllProfiles ? undefined : onNewSessionInWorkspace}
              onPromptSessions={onPromptSessions}
              onRestoreSessions={onRestoreSessions}
              onResumeSession={onResumeSession}
              onSessionDragEnd={handleSessionDragEnd}
              onSessionDragStart={handleSessionDragStart}
              onSteerSessions={onSteerSessions}
              onToggle={() => setSidebarRecentsOpen(!agentsOpen)}
              onTogglePin={pinSession}
              open={agentsOpen}
              pinned={false}
              rootClassName="min-h-0 flex-1 p-0"
              sectionKey="sessions"
              sessions={displayAgentSessions}
              sortable={!showAllProfiles && agentSessions.length > 1}
              workingSessionIdSet={workingSessionIdSet}
            />
          )}

          {contentVisible &&
            showSessionSections &&
            !trimmedQuery &&
            messagingGroups.map(group => (
              <SidebarSessionsSection
                activeSessionId={activeSidebarSessionId}
                contentClassName="flex max-h-56 shrink-0 flex-col gap-px overflow-y-auto overscroll-contain pb-1.75"
                draggingSession={draggingSession}
                dragPreviewSession={dragPreviewSession}
                emptyState={null}
                footer={
                  group.hasMore ? (
                    <SidebarLoadMoreRow
                      loading={Boolean(messagingLoadMorePending[group.sourceId])}
                      onClick={() => loadMoreForMessaging(group.sourceId)}
                      step={Math.max(0, group.total - group.loadedCount)}
                    />
                  ) : null
                }
                key={group.sourceId}
                label={group.label}
                labelIcon={
                  // size-3.5 fills the header's shared leading slot exactly.
                  <PlatformAvatar
                    className="size-3.5 rounded-[4px] text-[0.5625rem] [&_svg]:size-3"
                    platformId={group.sourceId}
                    platformName={group.label}
                  />
                }
                labelMeta={countLabel(group.loadedCount, group.total)}
                onArchiveSession={onArchiveSession}
                onArchiveSessions={onArchiveSessions}
                onDeleteSession={onDeleteSession}
                onDeleteSessions={onDeleteSessions}
                onHaltSessions={onHaltSessions}
                onPromptSessions={onPromptSessions}
                onRestoreSessions={onRestoreSessions}
                onResumeSession={onResumeSession}
                onSessionDragEnd={handleSessionDragEnd}
                onSessionDragStart={handleSessionDragStart}
                onSteerSessions={onSteerSessions}
                onToggle={() =>
                  setMessagingOpen(prev => ({ ...prev, [group.sourceId]: prev[group.sourceId] === false }))
                }
                onTogglePin={pinSession}
                open={messagingOpen[group.sourceId] !== false}
                pinned={false}
                rootClassName="shrink-0 p-0"
                sectionKey={`messaging:${group.sourceId}`}
                sessions={group.sessions}
                workingSessionIdSet={workingSessionIdSet}
              />
            ))}

          {contentVisible && !trimmedQuery && cronJobs.length > 0 && (
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

          {contentVisible && !trimmedQuery && (
            <SidebarRemoteSessionsSection
              label={s.liveElsewhere}
              newSessionLabel={s.newSessionIn}
              onCreateOnDevice={onCreateOnDevice}
              onResumeSession={onResumeSession}
              onToggle={() => setRemoteOpen(prev => !prev)}
              open={remoteOpen}
            />
          )}

          {/* Archived sits LAST in the section stack: it's cold storage, below
            every live surface (pins, recents, platforms, cron, presence), but
            always one scroll away — no settings detour. Collapsed by default;
            hidden entirely while nothing is archived. Rows are fetched lazily
            on first expand (the boot refresh only resolves the count). */}
          {contentVisible && !trimmedQuery && (archivedTotal > 0 || archivedSessions.length > 0) && (
            <SidebarSessionsSection
              activeSessionId={activeSidebarSessionId}
              archivedRows
              contentClassName="flex max-h-56 shrink-0 flex-col gap-px overflow-y-auto overscroll-contain pb-1.75"
              draggingSession={draggingSession}
              dragPreviewSession={dragPreviewSession}
              dropActive={archivedDropZone.active}
              dropHandlers={archivedDropZone.dropHandlers}
              emptyState={
                archivedLoading ? (
                  <SidebarSessionSkeletons />
                ) : (
                  <div className="flex min-h-7 items-center rounded-lg pl-7 text-[0.75rem] text-(--ui-text-tertiary)">
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
              labelIcon={<Codicon className="text-(--ui-text-quaternary)" name="archive" size="0.75rem" />}
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
              onSessionDragEnd={handleSessionDragEnd}
              onSessionDragStart={handleSessionDragStart}
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

          {contentVisible && !showSessionSections && <div className="min-h-0 flex-1" />}

          {contentVisible && (
            <div className="shrink-0 px-0.5 pb-1 pt-0.5">
              <ProfileRail />
            </div>
          )}
        </SidebarContent>
        <ArchiveAllSessionsDialog
          count={knownSessionTotal}
          onConfirm={handleArchiveAll}
          onOpenChange={setArchiveAllOpen}
          open={archiveAllOpen}
          submitting={archiveAllSubmitting}
        />
      </Sidebar>
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
  /** Rows this section renders (pinned rows excluded — they live in Pinned). */
  sessions: SessionInfo[]
  /** Loaded conversations for this platform, pinned included — the honest
   * "loaded" side of the count label and load-more math. */
  loadedCount: number
  /** Latest activity across every loaded row (pinned included). */
  latestActivity: number
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
  /** Identity for multi-select; sections without a key don't select. */
  sectionKey?: SidebarSectionKey
  /** Rows live in the Archived section (restore semantics, no pinning). */
  archivedRows?: boolean
  onRestoreSession?: (sessionId: string) => void
  /** Bulk verbs for this section's selection-mode header. */
  onArchiveSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onRestoreSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onDeleteSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onPromptSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
  onSteerSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
  onHaltSessions?: (sessionIds: string[]) => Promise<unknown> | void
  rootClassName?: string
  contentClassName?: string
  emptyState: React.ReactNode
  forceEmptyState?: boolean
  headerAction?: React.ReactNode
  footer?: React.ReactNode
  groups?: SidebarSessionGroup[]
  labelMeta?: React.ReactNode
  labelIcon?: React.ReactNode
  draggingSession?: null | SessionDragPayload
  dragPreviewSession?: null | SessionInfo
  onSessionDragEnd?: () => void
  onSessionDragStart?: (payload: SessionDragPayload) => void
  sortable?: boolean
  /** Native session-drag drop target (drag-to-pin/unpin): true while an
   * acceptable row drag hovers the section. */
  dropActive?: boolean
  dropAnchor?: null | SessionDropAnchor
  dropHandlers?: ReturnType<typeof useSessionDropZone>['dropHandlers']
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
  onArchiveSessions,
  onRestoreSessions,
  onDeleteSessions,
  onPromptSessions,
  onSteerSessions,
  onHaltSessions,
  rootClassName,
  contentClassName,
  emptyState,
  forceEmptyState = false,
  headerAction,
  footer,
  groups,
  labelMeta,
  labelIcon,
  draggingSession,
  dragPreviewSession,
  onSessionDragEnd,
  onSessionDragStart,
  sortable = false,
  dropActive = false,
  dropAnchor,
  dropHandlers
}: SidebarSessionsSectionProps) {
  const hasGroupedSessions = Boolean(groups?.some(group => group.sessions.length > 0))
  const showEmptyState = forceEmptyState || (!hasGroupedSessions && sessions.length === 0)

  // One subscription per SECTION (not per row): rows receive plain props, so
  // only sections re-render as the selection changes.
  const selection = useStore($sidebarSelection)
  const selectable = Boolean(sectionKey)
  const selectionActive = selectable && selection.section === sectionKey && selection.ids.length > 0

  const selectedIdSet = useMemo(
    () => (selectionActive ? new Set(selection.ids) : undefined),
    [selectionActive, selection.ids]
  )

  const previewSessions = useMemo(
    () => (dropActive ? previewItemsAtAnchor(sessions, dragPreviewSession, dropAnchor ?? null) : sessions),
    [dragPreviewSession, dropActive, dropAnchor, sessions]
  )

  // Range anchors resolve against this flat order. In grouped views it can
  // differ from the rendered order — the range is still a contiguous run of
  // the section's recency list, which is the least surprising interpretation.
  const orderedIds = useMemo(() => sessions.map(s => s.id), [sessions])

  const handleToggleSelect = useCallback(
    (sessionId: string, mode: 'range' | 'single') => {
      if (!sectionKey) {
        return
      }

      if (mode === 'range') {
        // Cold shift-clicks anchor from the OPEN session's row so the range
        // includes where the user started (Finder semantics).
        rangeSelectSessions(sectionKey, sessionId, orderedIds, activeSessionId)
      } else {
        toggleSessionSelected(sectionKey, sessionId)
      }
    },
    [sectionKey, orderedIds, activeSessionId]
  )

  const renderRow = (session: SessionInfo) => {
    const rowIsChecked = selectedIdSet?.has(session.id) ?? false

    const rowProps = {
      archived: archivedRows,
      bulkSelectedSessionIds: rowIsChecked && selection.ids.length > 1 ? selection.ids : undefined,
      checked: rowIsChecked,
      dragging: draggingSession?.id === session.id,
      isPinned: pinned,
      isSelected: session.id === activeSessionId,
      isWorking: workingSessionIdSet.has(session.id),
      onArchive: () => onArchiveSession(session.id),
      onArchiveSelectedSessions: onArchiveSessions,
      onDelete: () => onDeleteSession(session.id),
      onDeleteSelectedSessions: onDeleteSessions,
      onHaltSelectedSessions: archivedRows ? undefined : onHaltSessions,
      onPin: () => onTogglePin(sessionPinId(session)),
      onPromptSelectedSessions: archivedRows ? undefined : onPromptSessions,
      onRestore: onRestoreSession ? () => onRestoreSession(session.id) : undefined,
      onRestoreSelectedSessions: onRestoreSessions,
      onResume: () => onResumeSession(session.id),
      onSteerSelectedSessions: archivedRows ? undefined : onSteerSessions,
      onSessionDragEnd,
      onSessionDragStart,
      onToggleSelect: sectionKey ? (mode: 'range' | 'single') => handleToggleSelect(session.id, mode) : undefined,
      selectable,
      selectionActive,
      session
    }

    return <SidebarSessionRow key={session.id} {...rowProps} reorderable={sortable} />
  }

  const renderRows = (items: SessionInfo[]) => items.map(renderRow)

  const flatVirtualized = !showEmptyState && !groups?.length && previewSessions.length >= VIRTUALIZE_THRESHOLD

  let inner: React.ReactNode

  if (showEmptyState) {
    inner = emptyState
  } else if (groups?.length) {
    inner = groups.map(group =>
      <SidebarWorkspaceGroup
        group={group}
        key={group.id}
        onNewSession={onNewSessionInWorkspace}
        renderRows={renderRows}
      />
    )
  } else if (flatVirtualized) {
    inner = (
      <VirtualSessionList
        activeSessionId={activeSessionId}
        archived={archivedRows}
        draggingSessionId={draggingSession?.id}
        onArchiveSession={onArchiveSession}
        onArchiveSessions={onArchiveSessions}
        onDeleteSession={onDeleteSession}
        onDeleteSessions={onDeleteSessions}
        onHaltSessions={archivedRows ? undefined : onHaltSessions}
        onPromptSessions={archivedRows ? undefined : onPromptSessions}
        onRestoreSession={onRestoreSession}
        onRestoreSessions={onRestoreSessions}
        onResumeSession={onResumeSession}
        onSessionDragEnd={onSessionDragEnd}
        onSessionDragStart={onSessionDragStart}
        onSteerSessions={archivedRows ? undefined : onSteerSessions}
        onTogglePin={onTogglePin}
        onToggleSelect={sectionKey ? handleToggleSelect : undefined}
        pinned={pinned}
        reorderable={sortable}
        selectable={selectable}
        selectedIds={selectedIdSet}
        selectedSessionIds={selection.ids}
        selectionActive={selectionActive}
        sessions={previewSessions}
        workingSessionIdSet={workingSessionIdSet}
      />
    )
  } else {
    inner = renderRows(previewSessions)
  }

  // The virtualizer owns its own scroller, so suppress the wrapper's overflow
  // to avoid a double scroll container.
  const resolvedContentClassName = cn(contentClassName, flatVirtualized && 'overflow-y-visible')

  return (
    <SidebarGroup
      className={cn(
        rootClassName,
        // Light the whole section (header included — drops there count too, even
        // collapsed) while an acceptable row drag hovers it.
        dropActive && 'rounded-lg bg-(--ui-control-hover-background)'
      )}
      {...dropHandlers}
    >
      {selectionActive ? (
        // Selection mode: the section header becomes the bulk action bar —
        // count + verbs sit directly above the checked rows instead of a
        // detached bar at the bottom of the sidebar nobody finds.
        <SelectionActionBar
          onArchiveSessions={onArchiveSessions}
          onDeleteSessions={onDeleteSessions}
          onHaltSessions={archivedRows ? undefined : onHaltSessions}
          onPromptSessions={archivedRows ? undefined : onPromptSessions}
          onRestoreSessions={onRestoreSessions}
          onSteerSessions={archivedRows ? undefined : onSteerSessions}
          sessions={sessions}
        />
      ) : (
        <SidebarSectionHeader
          action={headerAction}
          icon={labelIcon}
          label={label}
          meta={labelMeta}
          onToggle={onToggle}
          open={open}
        />
      )}
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
}

function SidebarWorkspaceGroup({
  group,
  renderRows,
  onNewSession,
  className,
  style,
  ref,
  ...rest
}: SidebarWorkspaceGroupProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const isProfileGroup = group.mode === 'profile'
  const isSourceGroup = group.mode === 'source'
  const pageStep = PROFILE_INITIAL_PAGE
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
      className={cn('grid gap-px', className)}
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
          {/* Leading w-3.5 slot mirrors the section headers and session rows,
              so group labels share their text edge — even when the group has
              no color dot or platform avatar to show. */}
          <span aria-hidden="true" className="grid w-3.5 shrink-0 place-items-center">
            {group.color ? (
              <span className="size-2 shrink-0 rounded-full" style={{ backgroundColor: group.color }} />
            ) : isSourceGroup && group.sourceId ? (
              <PlatformAvatar
                className="size-3.5 rounded-[4px] text-[0.5625rem] [&_svg]:size-3"
                platformId={group.sourceId}
                platformName={group.label}
              />
            ) : null}
          </span>
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
                  onClick={() => setVisibleCount(count => count + pageStep)}
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
