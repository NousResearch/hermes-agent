import { useStore } from '@nanostores/react'
import { useQueryClient } from '@tanstack/react-query'
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Navigate, Route, Routes, useLocation, useNavigate, useParams } from 'react-router-dom'

import { BootFailureOverlay } from '@/components/boot-failure-overlay'
import { DesktopInstallOverlay } from '@/components/desktop-install-overlay'
import { DesktopOnboardingOverlay } from '@/components/desktop-onboarding-overlay'
import { GatewayConnectingOverlay } from '@/components/gateway-connecting-overlay'
import { Pane, PaneMain } from '@/components/pane-shell'
import { RemoteDisplayBanner } from '@/components/remote-display-banner'
import { useMediaQuery } from '@/hooks/use-media-query'
import { useSkinCommand } from '@/themes/use-skin-command'

import { formatRefValue } from '../components/assistant-ui/directive-text'
import { getCronJobs, getSessionMessages, listAllProfileSessions, type SessionInfo, triggerCronJob } from '../hermes'
import { type ChatMessage, chatMessageText, preserveLocalAssistantErrors, toChatMessages } from '../lib/chat-messages'
import { storedSessionIdForNotification } from '../lib/session-ids'
import {
  isMessagingSource,
  LOCAL_SESSION_SOURCE_IDS,
  MESSAGING_SESSION_SOURCE_IDS,
  normalizeSessionSource
} from '../lib/session-source'
import { latestSessionTodos } from '../lib/todos'
import { setCronFocusJobId, setCronJobs } from '../store/cron'
import {
  $panesFlipped,
  $pinnedSessionIds,
  $sessionsLimit,
  bumpSessionsLimit,
  FILE_BROWSER_DEFAULT_WIDTH,
  FILE_BROWSER_MAX_WIDTH,
  FILE_BROWSER_MIN_WIDTH,
  pinSession,
  setSidebarOverlayMounted,
  SIDEBAR_DEFAULT_WIDTH,
  SIDEBAR_MAX_WIDTH,
  SIDEBAR_SESSIONS_PAGE_SIZE,
  unpinSession
} from '../store/layout'
import { respondToApprovalAction } from '../store/native-notifications'
import { $filePreviewTarget, $previewTarget, closeActiveRightRailTab } from '../store/preview'
import {
  $activeGatewayProfile,
  $freshSessionRequest,
  $profileScope,
  ALL_PROFILES,
  normalizeProfileKey,
  refreshActiveProfile
} from '../store/profile'
import { getProject } from '../store/projects'
import {
  $activeSessionId,
  $currentCwd,
  $freshDraftReady,
  $gatewayState,
  $messages,
  $messagingSessions,
  $resumeExhaustedSessionId,
  $resumeFailedSessionId,
  $selectedStoredSessionId,
  $sessions,
  $workingSessionIds,
  CRON_SECTION_LIMIT,
  getRecentlySettledSessionIds,
  mergeSessionPage,
  MESSAGING_SECTION_LIMIT,
  sessionPinId,
  setAwaitingResponse,
  setBusy,
  setCronSessions,
  setCurrentBranch,
  setCurrentCwd,
  setCurrentModel,
  setCurrentProvider,
  setMessages,
  setMessagingPlatformTotals,
  setMessagingSessions,
  setMessagingTruncated,
  setSessionProfileTotals,
  setSessions,
  setSessionsLoading,
  setSessionsTotal
} from '../store/session'
import { onSessionsChanged } from '../store/session-sync'
import { clearSessionTodos, setSessionTodos, todoListActive } from '../store/todos'
import { openUpdatesWindow, startUpdatePoller, stopUpdatePoller } from '../store/updates'
import { isSecondaryWindow } from '../store/windows'

import { ChatView } from './chat'
import { requestComposerFocus, requestComposerInsert } from './chat/composer/focus'
import { useComposerActions } from './chat/hooks/use-composer-actions'
import {
  ChatPreviewRail,
  PREVIEW_RAIL_MAX_WIDTH,
  PREVIEW_RAIL_MIN_WIDTH,
  PREVIEW_RAIL_PANE_WIDTH
} from './chat/right-rail'
import { ChatSidebar } from './chat/sidebar'
import { CommandPalette } from './command-palette'
import { useGatewayBoot } from './gateway/hooks/use-gateway-boot'
import { useGatewayRequest } from './gateway/hooks/use-gateway-request'
import { useKeybinds } from './hooks/use-keybinds'
import { SIDEBAR_COLLAPSE_MEDIA_QUERY } from './layout-constants'
import { ModelPickerOverlay } from './model-picker-overlay'
import { ModelVisibilityOverlay } from './model-visibility-overlay'
import { CreateProjectDialog } from './projects/create-project-dialog'
import { ProjectPageView } from './projects/project-page'
import { RightSidebarPane } from './right-sidebar'
import { $terminalTakeover } from './right-sidebar/store'
import { PersistentTerminal, TerminalSlot } from './right-sidebar/terminal/persistent'
import { CRON_ROUTE, NEW_CHAT_ROUTE, projectRoute, routeSessionId, sessionRoute, SETTINGS_ROUTE } from './routes'
import { SessionPickerOverlay } from './session-picker-overlay'
import { SessionSwitcher } from './session-switcher'
import { useContextSuggestions } from './session/hooks/use-context-suggestions'
import { useCwdActions } from './session/hooks/use-cwd-actions'
import { useHermesConfig } from './session/hooks/use-hermes-config'
import { useMessageStream } from './session/hooks/use-message-stream'
import { useModelControls } from './session/hooks/use-model-controls'
import { usePreviewRouting } from './session/hooks/use-preview-routing'
import { usePromptActions } from './session/hooks/use-prompt-actions'
import { useRouteResume } from './session/hooks/use-route-resume'
import { useSessionActions } from './session/hooks/use-session-actions'
import { useSessionStateCache } from './session/hooks/use-session-state-cache'
import { AppShell } from './shell/app-shell'
import { useOverlayRouting } from './shell/hooks/use-overlay-routing'
import { useStatusSnapshot } from './shell/hooks/use-status-snapshot'
import { useStatusbarItems } from './shell/hooks/use-statusbar-items'
import { ModelMenuPanel } from './shell/model-menu-panel'
import type { StatusbarItem } from './shell/statusbar-controls'
import type { TitlebarTool } from './shell/titlebar-controls'
import { useGroupRegistry } from './shell/use-group-registry'
import { UpdatesOverlay } from './updates-overlay'

const AgentsView = lazy(async () => ({ default: (await import('./agents')).AgentsView }))
const ArtifactsView = lazy(async () => ({ default: (await import('./artifacts')).ArtifactsView }))
const CommandCenterView = lazy(async () => ({ default: (await import('./command-center')).CommandCenterView }))
const CronView = lazy(async () => ({ default: (await import('./cron')).CronView }))
const MessagingView = lazy(async () => ({ default: (await import('./messaging')).MessagingView }))
const ProfilesView = lazy(async () => ({ default: (await import('./profiles')).ProfilesView }))
const SettingsView = lazy(async () => ({ default: (await import('./settings')).SettingsView }))
const SkillsView = lazy(async () => ({ default: (await import('./skills')).SkillsView }))

// Latest cron-job sessions surfaced in the collapsed "Cron jobs" section. The
// Cron sessions are written by a background scheduler tick (the desktop
// backend), so no user action signals the UI. Poll the bounded cron list on
// this cadence while the app is open + visible so new runs surface promptly
// instead of waiting for the next user-triggered refreshSessions().
const CRON_POLL_INTERVAL_MS = 30_000
// The recents list is local-only: cron rows have their own section, and each
// messaging platform (telegram, discord, …) is fetched separately into its own
// self-managed sidebar section (refreshMessagingSessions). Excluding both here
// keeps "Load more" paging through interactive local chats instead of
// interleaving gateway threads that bury them.
const SIDEBAR_EXCLUDED_SOURCES = ['cron', 'subagent', 'tool', ...MESSAGING_SESSION_SOURCE_IDS]
// The messaging slice is the inverse: drop cron + every local source so only
// external-platform conversations remain, then split per platform in the UI.
const MESSAGING_EXCLUDED_SOURCES = ['cron', ...LOCAL_SESSION_SOURCE_IDS]

// Cheap signature compare so the poll only swaps the atom (and re-renders the
// sidebar) when the visible cron rows actually changed.
function sameCronSignature(a: SessionInfo[], b: SessionInfo[]): boolean {
  if (a.length !== b.length) {
    return false
  }

  return a.every((session, i) => session.id === b[i]?.id && session.title === b[i]?.title)
}

// Rows a session refresh must preserve even if the aggregator omits them:
// in-flight first turns (message_count 0), pinned rows aged off the page, the
// actively-viewed chat (its "working" flag clears a beat before the aggregator
// sees the persisted row), and sessions whose turn just settled (same race, but
// for a chat the user has already navigated away from). Pass `scope` to only
// keep the active row when it belongs to the profile being paged.
function sessionsToKeep(scope?: string): Set<string> {
  const keep = new Set<string>([
    ...$workingSessionIds.get(),
    ...$pinnedSessionIds.get(),
    ...getRecentlySettledSessionIds()
  ])

  const active = $selectedStoredSessionId.get()

  if (active) {
    const session = scope ? $sessions.get().find(s => s.id === active) : null

    if (!scope || !session || normalizeProfileKey(session.profile) === scope) {
      keep.add(active)
    }
  }

  return keep
}

export function DesktopController() {
  const queryClient = useQueryClient()
  const location = useLocation()
  const navigate = useNavigate()

  const [createProjectOpen, setCreateProjectOpen] = useState(false)

  // Derive the active project directly from the URL so it clears automatically
  // when the user navigates to a session, new chat, or any other view.
  const selectedProjectId = location.pathname.startsWith('/project/')
    ? decodeURIComponent(location.pathname.slice('/project/'.length))
    : null

  const busyRef = useRef(false)
  const creatingSessionRef = useRef(false)
  const refreshSessionsRequestRef = useRef(0)

  const gatewayState = useStore($gatewayState)
  const activeSessionId = useStore($activeSessionId)
  const currentCwd = useStore($currentCwd)
  const freshDraftReady = useStore($freshDraftReady)
  const resumeFailedSessionId = useStore($resumeFailedSessionId)
  const resumeExhaustedSessionId = useStore($resumeExhaustedSessionId)
  const filePreviewTarget = useStore($filePreviewTarget)
  const previewTarget = useStore($previewTarget)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const terminalTakeover = useStore($terminalTakeover)
  const panesFlipped = useStore($panesFlipped)
  const profileScope = useStore($profileScope)
  // Below SIDEBAR_COLLAPSE_BREAKPOINT_PX there's no room for a docked rail —
  // collapse both sidebars (without touching their stored open state) so the
  // hover-reveal overlay becomes the way in. Restores once it's wide again.
  const narrowViewport = useMediaQuery(SIDEBAR_COLLAPSE_MEDIA_QUERY)

  const routedSessionId = routeSessionId(location.pathname)
  const routeToken = `${location.pathname}:${location.search}:${location.hash}`
  const routeTokenRef = useRef(routeToken)
  routeTokenRef.current = routeToken
  const getRouteToken = useCallback(() => routeTokenRef.current, [])

  const {
    agentsOpen,
    chatOpen,
    closeOverlayToPreviousRoute,
    commandCenterInitialSection,
    commandCenterOpen,
    cronOpen,
    currentView,
    openAgents,
    openCommandCenterSection,
    profilesOpen,
    settingsOpen,
    toggleCommandCenter
  } = useOverlayRouting()

  const terminalSidebarOpen = chatOpen && terminalTakeover

  const titlebarToolGroups = useGroupRegistry<TitlebarTool>()
  const statusbarItemGroups = useGroupRegistry<StatusbarItem>()
  const setTitlebarToolGroup = titlebarToolGroups.set
  const setStatusbarItemGroup = statusbarItemGroups.set

  const {
    activeSessionIdRef,
    ensureSessionState,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    syncSessionStateToView,
    updateSessionState
  } = useSessionStateCache({
    activeSessionId,
    busyRef,
    selectedStoredSessionId,
    setAwaitingResponse,
    setBusy,
    setMessages
  })

  const { connectionRef, gatewayRef, requestGateway } = useGatewayRequest()

  useEffect(() => {
    window.hermesDesktop?.setPreviewShortcutActive?.(Boolean(chatOpen && (filePreviewTarget || previewTarget)))
  }, [chatOpen, filePreviewTarget, previewTarget])

  useEffect(() => {
    startUpdatePoller()
    const unsubscribe = window.hermesDesktop?.onOpenUpdatesRequested?.(() => openUpdatesWindow())

    return () => {
      unsubscribe?.()
      stopUpdatePoller()
    }
  }, [])

  // Notification click: the main process already focused the window; jump to its
  // session. Notifications are tagged with the gateway *runtime* session id, but
  // the chat route is keyed by the *stored* id — navigating with the runtime id
  // resumes a non-existent stored session ("session not found") and strands the
  // user. Translate runtime -> stored before navigating.
  useEffect(() => {
    const unsubscribe = window.hermesDesktop?.onFocusSession?.(sessionId => {
      if (sessionId) {
        navigate(sessionRoute(storedSessionIdForNotification(sessionId, runtimeIdByStoredSessionIdRef.current)))
      }
    })

    return () => unsubscribe?.()
  }, [navigate, runtimeIdByStoredSessionIdRef])

  // Notification action button (Approve/Reject) — resolve in place, no navigation.
  useEffect(() => {
    const unsubscribe = window.hermesDesktop?.onNotificationAction?.(({ actionId, sessionId }) => {
      void respondToApprovalAction(sessionId ?? null, actionId)
    })

    return () => unsubscribe?.()
  }, [])

  // hermes:// deep links (e.g. a docs "Send to App" button for an automation blueprint).
  // Build the equivalent /blueprint slash command from the payload and drop
  // it into the composer — the user reviews/edits, then sends; the agent (or
  // the shared command handler) creates the job. Signal readiness so a link
  // that arrived during boot is flushed exactly once.
  useEffect(() => {
    const unsubscribe = window.hermesDesktop?.onDeepLink?.(payload => {
      if (!payload || payload.kind !== 'blueprint' || !payload.name) {
        return
      }

      const slots = Object.entries(payload.params || {})
        .map(([k, v]) => {
          const sval = /\s/.test(v) ? `"${v.replace(/"/g, '\\"')}"` : v

          return `${k}=${sval}`
        })
        .join(' ')

      const command = `/blueprint ${payload.name}${slots ? ' ' + slots : ''}`
      requestComposerInsert(command, { mode: 'block', target: 'main' })
      requestComposerFocus('main')
    })

    // Tell the main process the renderer is ready to receive deep links.
    void window.hermesDesktop?.signalDeepLinkReady?.()

    return () => unsubscribe?.()
  }, [])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!$filePreviewTarget.get() && !$previewTarget.get()) {
        return
      }

      if ((event.metaKey || event.ctrlKey) && !event.altKey && !event.shiftKey && event.key.toLowerCase() === 'w') {
        event.preventDefault()
        event.stopPropagation()
        closeActiveRightRailTab()
      }
    }

    const unsubscribe = window.hermesDesktop?.onClosePreviewRequested?.(closeActiveRightRailTab)

    window.addEventListener('keydown', onKeyDown, { capture: true })

    return () => {
      unsubscribe?.()
      window.removeEventListener('keydown', onKeyDown, { capture: true })
    }
  }, [])

  // Cron-job sessions as their own list (latest N). Independent of the recents
  // page so the two never compete for slots. Cheap + bounded. Kept (even though
  // the sidebar now lists cron *jobs*, not run sessions) so a pinned cron run
  // still resolves into the Pinned section via sessionByAnyId.
  const refreshCronSessions = useCallback(async () => {
    try {
      const { sessions } = await listAllProfileSessions(CRON_SECTION_LIMIT, 1, 'exclude', 'recent', 'all', {
        source: 'cron'
      })

      setCronSessions(prev => (sameCronSignature(prev, sessions) ? prev : sessions))
    } catch {
      // Non-fatal: the cron section just stays empty/stale.
    }
  }, [])

  // Messaging-platform sessions as their own slice, fetched separately from
  // local recents so each platform renders a self-managed section and never
  // competes with local chats for the recents page budget. One combined fetch
  // seeds every platform; the sidebar splits the rows per source.
  const refreshMessagingSessions = useCallback(async () => {
    try {
      const result = await listAllProfileSessions(MESSAGING_SECTION_LIMIT, 1, 'exclude', 'recent', 'all', {
        excludeSources: MESSAGING_EXCLUDED_SOURCES
      })

      // Drop any non-messaging source the broad exclude didn't catch (custom
      // sources) — those stay in local recents, not a platform section.
      const rows = result.sessions.filter(s => isMessagingSource(s.source))

      setMessagingSessions(prev => (sameCronSignature(prev, rows) ? prev : rows))
      // Hit the cap → at least one platform may have more on disk than loaded,
      // so platform sections offer their own per-platform "load more".
      setMessagingTruncated(result.sessions.length >= MESSAGING_SECTION_LIMIT)
    } catch {
      // Non-fatal: the messaging sections just stay empty/stale.
    }
  }, [])

  // Page a single platform's section independently (mirrors the per-profile
  // pager): fetch that source's next window and merge it back in place, leaving
  // every other platform's rows untouched. Resolves the platform's exact total.
  const loadMoreMessagingForPlatform = useCallback(async (platform: string) => {
    const inPlatform = (s: SessionInfo) => normalizeSessionSource(s.source) === platform
    const loaded = $messagingSessions.get().filter(inPlatform).length

    const result = await listAllProfileSessions(loaded + SIDEBAR_SESSIONS_PAGE_SIZE, 1, 'exclude', 'recent', 'all', {
      source: platform
    })

    const incoming = result.sessions.filter(s => normalizeSessionSource(s.source) === platform)

    setMessagingSessions(prev => [
      ...prev.filter(s => !inPlatform(s)),
      ...mergeSessionPage(prev.filter(inPlatform), incoming, sessionsToKeep())
    ])

    const total = result.total ?? incoming.length
    setMessagingPlatformTotals(prev => ({ ...prev, [platform]: Math.max(total, incoming.length) }))
  }, [])

  // Cron *jobs* drive the sidebar "Cron jobs" section. Jobs are created
  // synchronously (agent tool call or the cron UI), so refreshing here right
  // after an agent turn surfaces a new job immediately; the interval poll keeps
  // next-run/state fresh as the scheduler advances them.
  const refreshCronJobs = useCallback(async () => {
    try {
      const jobs = await getCronJobs()

      setCronJobs(jobs)
    } catch {
      // Non-fatal: the cron section just keeps its last-known jobs.
    }
  }, [])

  const refreshSessions = useCallback(async () => {
    const requestId = refreshSessionsRequestRef.current + 1
    refreshSessionsRequestRef.current = requestId
    setSessionsLoading(true)

    try {
      const limit = $sessionsLimit.get()

      // Require at least one message so abandoned/empty "Untitled" drafts (one
      // was created per TUI/desktop launch before the lazy-create fix) don't
      // clutter the sidebar.
      // Unified cross-profile list (served read-only off each profile's
      // state.db; no per-profile backend is spawned). Single-profile users get
      // the same rows tagged profile="default". Cron sessions are excluded here
      // and fetched separately (refreshCronSessions) so the scheduler's
      // always-newest rows can't consume the recents page budget.
      // Scope the fetch to the active profile (not always 'all') so a profile
      // with few recent sessions isn't windowed out of the cross-profile
      // recency page — the empty-history-on-profile-switch bug.
      const sessionProfile = profileScope === ALL_PROFILES ? 'all' : profileScope

      const result = await listAllProfileSessions(limit, 1, 'exclude', 'recent', sessionProfile, {
        excludeSources: SIDEBAR_EXCLUDED_SOURCES
      })

      if (refreshSessionsRequestRef.current === requestId) {
        setSessions(prev => mergeSessionPage(prev, result.sessions, sessionsToKeep()))
        setSessionsTotal(typeof result.total === 'number' ? result.total : result.sessions.length)
        setSessionProfileTotals(result.profile_totals ?? {})
      }
    } finally {
      if (refreshSessionsRequestRef.current === requestId) {
        setSessionsLoading(false)
      }
    }

    void refreshCronSessions()
    void refreshCronJobs()
    void refreshMessagingSessions()
  }, [profileScope, refreshCronSessions, refreshCronJobs, refreshMessagingSessions])

  const loadMoreSessions = useCallback(() => {
    bumpSessionsLimit()
    void refreshSessions()
  }, [refreshSessions])

  // Another window mutated the shared session list (e.g. a chat started in the
  // pop-out). Re-pull so the sidebar reflects it. Pop-outs have no sidebar, so
  // only real windows bother.
  useEffect(() => {
    if (isSecondaryWindow()) {
      return
    }

    return onSessionsChanged(() => void refreshSessions().catch(() => undefined))
  }, [refreshSessions])

  // ALL-profiles view pages one profile at a time: fetch that profile's next
  // page and merge it in place, leaving every other profile's rows untouched.
  const loadMoreSessionsForProfile = useCallback(async (profile: string) => {
    const key = normalizeProfileKey(profile)
    const inKey = (s: SessionInfo) => normalizeProfileKey(s.profile) === key
    const loaded = $sessions.get().filter(inKey).length

    const result = await listAllProfileSessions(loaded + SIDEBAR_SESSIONS_PAGE_SIZE, 1, 'exclude', 'recent', key, {
      excludeSources: SIDEBAR_EXCLUDED_SOURCES
    })

    const keep = sessionsToKeep(key)

    setSessions(prev => [
      ...prev.filter(s => !inKey(s)),
      ...mergeSessionPage(prev.filter(inKey), result.sessions, keep)
    ])

    const total = result.profile_totals?.[key] ?? result.total ?? result.sessions.length
    setSessionProfileTotals(prev => ({ ...prev, [key]: Math.max(total, result.sessions.length) }))
  }, [])

  const toggleSelectedPin = useCallback(() => {
    const sessionId = $selectedStoredSessionId.get()

    if (!sessionId) {
      return
    }

    // Pin on the durable lineage-root id so the pin survives auto-compression.
    const session = $sessions.get().find(s => s.id === sessionId || s._lineage_root_id === sessionId)
    const pinId = session ? sessionPinId(session) : sessionId

    if ($pinnedSessionIds.get().includes(pinId)) {
      unpinSession(pinId)
    } else {
      pinSession(pinId)
    }
  }, [])

  const { gatewayLogLines, inferenceStatus, statusSnapshot } = useStatusSnapshot(gatewayState, requestGateway)

  const updateActiveSessionRuntimeInfo = useCallback(
    (info: { branch?: string; cwd?: string }) => {
      const sessionId = activeSessionIdRef.current

      if (!sessionId) {
        return
      }

      updateSessionState(sessionId, state => ({
        ...state,
        branch: info.branch ?? state.branch,
        cwd: info.cwd ?? state.cwd
      }))
    },
    [activeSessionIdRef, updateSessionState]
  )

  const { changeSessionCwd, refreshProjectBranch } = useCwdActions({
    activeSessionId,
    activeSessionIdRef,
    onSessionRuntimeInfo: updateActiveSessionRuntimeInfo,
    requestGateway
  })

  const { refreshHermesConfig, sttEnabled, voiceMaxRecordingSeconds } = useHermesConfig({
    activeSessionIdRef,
    refreshProjectBranch
  })

  const { refreshCurrentModel, selectModel, updateModelOptionsCache } = useModelControls({
    activeSessionId,
    queryClient,
    requestGateway
  })

  const openProviderSettings = useCallback(() => {
    navigate(`${SETTINGS_ROUTE}?tab=providers`)
  }, [navigate])

  const modelMenuContent = useMemo(
    () =>
      gatewayState === 'open' ? (
        <ModelMenuPanel
          gateway={gatewayRef.current || undefined}
          onSelectModel={selectModel}
          requestGateway={requestGateway}
        />
      ) : null,
    [gatewayRef, gatewayState, requestGateway, selectModel]
  )

  useContextSuggestions({
    activeSessionId,
    activeSessionIdRef,
    currentCwd,
    gatewayState,
    requestGateway
  })

  const hydrateFromStoredSession = useCallback(
    async (
      attempts = 1,
      storedSessionId = selectedStoredSessionIdRef.current,
      runtimeSessionId = activeSessionIdRef.current
    ) => {
      if (!storedSessionId || !runtimeSessionId) {
        return
      }

      const storedProfile = $sessions
        .get()
        .find(session => session.id === storedSessionId || session._lineage_root_id === storedSessionId)?.profile

      for (let index = 0; index < Math.max(1, attempts); index += 1) {
        try {
          const latest = await getSessionMessages(storedSessionId, storedProfile)
          const messages = toChatMessages(latest.messages)
          updateSessionState(
            runtimeSessionId,
            state => ({
              ...state,
              messages: preserveLocalAssistantErrors(messages, state.messages)
            }),
            storedSessionId
          )

          // Seed the status stack's todo group from history — but only while
          // the plan is still in flight, so reopening an old chat doesn't pin
          // its finished todo list above the composer forever.
          const todos = latestSessionTodos(messages)

          if (todos && todoListActive(todos)) {
            setSessionTodos(runtimeSessionId, todos)
          } else {
            clearSessionTodos(runtimeSessionId)
          }

          return
        } catch {
          // Best-effort fallback when live stream payloads are empty.
        }

        if (index < attempts - 1) {
          await new Promise(resolve => window.setTimeout(resolve, 250))
        }
      }
    },
    [activeSessionIdRef, selectedStoredSessionIdRef, updateSessionState]
  )

  const { handleGatewayEvent } = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession,
    queryClient,
    refreshHermesConfig,
    refreshSessions,
    sessionStateByRuntimeIdRef,
    updateSessionState
  })

  const { handleDesktopGatewayEvent, restartPreviewServer } = usePreviewRouting({
    activeSessionIdRef,
    baseHandleGatewayEvent: handleGatewayEvent,
    currentCwd,
    currentView,
    requestGateway,
    routedSessionId,
    selectedStoredSessionId
  })

  const {
    archiveSession,
    branchCurrentSession,
    createBackendSessionForSend,
    openSettings,
    removeSession,
    resumeSession,
    selectSidebarItem,
    startFreshSessionDraft
  } = useSessionActions({
    activeSessionId,
    activeSessionIdRef,
    busyRef,
    creatingSessionRef,
    ensureSessionState,
    getRouteToken,
    navigate,
    requestGateway,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    syncSessionStateToView,
    updateSessionState
  })

  // Single global listener for every rebindable hotkey (incl. profile switching)
  // plus the on-screen keybind editor's capture mode.
  useKeybinds({
    startFreshSession: startFreshSessionDraft,
    toggleCommandCenter,
    toggleSelectedPin
  })

  // A profile switch/create drops to a fresh new-session draft so the previously
  // open session doesn't bleed across contexts. Skip the initial value.
  const freshSessionRequest = useStore($freshSessionRequest)
  const lastFreshRef = useRef(freshSessionRequest)

  useEffect(() => {
    if (freshSessionRequest === lastFreshRef.current) {
      return
    }

    lastFreshRef.current = freshSessionRequest
    startFreshSessionDraft()
  }, [freshSessionRequest, startFreshSessionDraft])

  // Swapping the live gateway to another profile must re-pull that profile's
  // global model + active-profile pill. Both are nanostores, so the blanket
  // invalidateQueries() the profile store fires on swap doesn't touch them —
  // without this the statusbar keeps showing the previous profile's model
  // (the "forgets the LLM setting" report). gatewayState stays 'open' across a
  // swap (background sockets persist), so the open→open effect won't re-run.
  const activeGatewayProfile = useStore($activeGatewayProfile)
  const lastGatewayProfileRef = useRef(activeGatewayProfile)

  useEffect(() => {
    if (activeGatewayProfile === lastGatewayProfileRef.current) {
      return
    }

    lastGatewayProfileRef.current = activeGatewayProfile
    // Force: the new profile has its own default, so reseed even if the composer
    // already shows the previous profile's model.
    void refreshCurrentModel(true)
    void refreshActiveProfile()
  }, [activeGatewayProfile, refreshCurrentModel])

  const composer = useComposerActions({
    activeSessionId,
    currentCwd,
    requestGateway
  })

  const branchInNewChat = useCallback(
    async (messageId?: string) => {
      const branched = await branchCurrentSession(messageId)

      if (branched) {
        await refreshSessions().catch(() => undefined)
      }

      return branched
    },
    [branchCurrentSession, refreshSessions]
  )

  // Clear a failed turn's red error banner from the transcript. Errors are
  // renderer-local state (never persisted), so dismissing is purely a view +
  // session-cache edit. A message that errored before emitting any visible
  // text is a bare error placeholder → drop it entirely; one that streamed
  // partial output then failed keeps its content and just sheds the error.
  // Both the per-runtime cache AND the live $messages view must be updated:
  // `preserveLocalAssistantErrors` re-grafts any still-errored message it
  // finds in the view onto the next session.info flush, so clearing only the
  // cache would let the heartbeat resurrect the banner.
  const dismissError = useCallback(
    (messageId: string) => {
      const runtimeSessionId = activeSessionIdRef.current

      if (!runtimeSessionId) {
        return
      }

      const clearErrorIn = (messages: ChatMessage[]): ChatMessage[] =>
        messages.flatMap(message => {
          if (message.id !== messageId || !message.error) {
            return [message]
          }

          if (!chatMessageText(message).trim() && !message.parts.some(part => part.type !== 'text')) {
            return []
          }

          return [{ ...message, error: undefined, pending: false }]
        })

      // View first: the flush below reads $messages as the "current" baseline
      // for error preservation, so the banner must be gone from it before the
      // cache update triggers a re-sync.
      setMessages(clearErrorIn($messages.get()))

      updateSessionState(runtimeSessionId, state => ({
        ...state,
        messages: clearErrorIn(state.messages)
      }))
    },
    [activeSessionIdRef, updateSessionState]
  )

  const startSessionInWorkspace = useCallback(
    (path: null | string) => {
      startFreshSessionDraft()

      const target = path?.trim()

      if (!target) {
        return
      }

      // The next message creates the backend session in $currentCwd, so seed
      // it (and the branch) from the workspace the user clicked the + on.
      setCurrentCwd(target)
      void requestGateway<{ branch?: string; cwd?: string }>('config.get', { key: 'project', cwd: target })
        .then(info => {
          setCurrentCwd(info.cwd || target)
          setCurrentBranch(info.branch || '')
        })
        .catch(() => undefined)
    },
    [requestGateway, startFreshSessionDraft]
  )

  const handleSkinCommand = useSkinCommand()

  const {
    cancelRun,
    editMessage,
    handleThreadMessagesChange,
    reloadFromMessage,
    restoreToMessage,
    steerPrompt,
    submitText,
    transcribeVoiceAudio
  } = usePromptActions({
    activeSessionId,
    activeSessionIdRef,
    branchCurrentSession: branchInNewChat,
    busyRef,
    createBackendSessionForSend,
    handleSkinCommand,
    refreshSessions,
    requestGateway,
    resumeStoredSession: resumeSession,
    selectedStoredSessionIdRef,
    startFreshSessionDraft,
    sttEnabled,
    updateSessionState
  })

  useGatewayBoot({
    handleGatewayEvent: handleDesktopGatewayEvent,
    onConnectionReady: c => {
      connectionRef.current = c
    },
    onGatewayReady: g => {
      gatewayRef.current = g
    },
    refreshHermesConfig,
    refreshSessions
  })

  useEffect(() => {
    if (gatewayState === 'open') {
      void refreshCurrentModel()
      void refreshActiveProfile()
      void refreshSessions().catch(() => undefined)
    }
  }, [gatewayState, refreshCurrentModel, refreshSessions])

  // Keep the cron jobs section live without a user action: the scheduler ticks
  // in the background (advancing next-run/state and creating runs), so poll the
  // job list on an interval (and on tab re-focus) while connected.
  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    const tick = () => {
      if (document.visibilityState === 'visible') {
        void refreshCronJobs()
      }
    }

    const intervalId = window.setInterval(tick, CRON_POLL_INTERVAL_MS)
    document.addEventListener('visibilitychange', tick)

    return () => {
      window.clearInterval(intervalId)
      document.removeEventListener('visibilitychange', tick)
    }
  }, [gatewayState, refreshCronJobs])

  useEffect(() => {
    if (gatewayState === 'open' && !activeSessionId && freshDraftReady) {
      void refreshCurrentModel()
      void refreshHermesConfig()
    }
  }, [activeSessionId, freshDraftReady, gatewayState, refreshCurrentModel, refreshHermesConfig])

  useRouteResume({
    activeSessionId,
    activeSessionIdRef,
    creatingSessionRef,
    currentView,
    freshDraftReady,
    gatewayState,
    locationPathname: location.pathname,
    resumeSession,
    resumeFailedSessionId,
    resumeExhaustedSessionId,
    routedSessionId,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId,
    selectedStoredSessionIdRef,
    startFreshSessionDraft
  })

  const { leftStatusbarItems, statusbarItems } = useStatusbarItems({
    agentsOpen,
    chatOpen,
    commandCenterOpen,
    extraLeftItems: statusbarItemGroups.flat.left,
    extraRightItems: statusbarItemGroups.flat.right,
    gatewayLogLines,
    gatewayState,
    inferenceStatus,
    openAgents,
    freshDraftReady,
    openCommandCenterSection,
    requestGateway,
    statusSnapshot,
    toggleCommandCenter
  })

  const sidebar = (
    <ChatSidebar
      currentView={currentView}
      onArchiveSession={sessionId => void archiveSession(sessionId)}
      onDeleteSession={sessionId => void removeSession(sessionId)}
      onLoadMoreMessaging={loadMoreMessagingForPlatform}
      onLoadMoreProfileSessions={loadMoreSessionsForProfile}
      onLoadMoreSessions={loadMoreSessions}
      onManageCronJob={jobId => {
        setCronFocusJobId(jobId)
        navigate(CRON_ROUTE)
      }}
      onNavigate={selectSidebarItem}
      onNewProject={() => setCreateProjectOpen(true)}
      onNewSessionInWorkspace={startSessionInWorkspace}
      onResumeSession={sessionId => navigate(sessionRoute(sessionId))}
      onSelectProject={id => {
        const project = getProject(id)

        navigate(projectRoute(id))

        // Seed cwd so the next new session starts in the project folder,
        // but do NOT call startSessionInWorkspace — that starts a fresh
        // draft and navigates away, overwriting the project route we just set.
        if (project) { setCurrentCwd(project.path) }
      }}
      onTriggerCronJob={jobId => {
        void triggerCronJob(jobId)
          .then(() => refreshCronJobs())
          .catch(() => undefined)
      }}
      selectedProjectId={selectedProjectId}
    />
  )

  // One PTY-backed terminal mounted forever; <TerminalSlot /> placeholders decide
  // where it shows. Lives in main's stacking context (not the root overlay layer)
  // so pane resize handles still paint above it. Toggling never rebuilds the shell.
  const mainOverlays = (
    <PersistentTerminal cwd={currentCwd} onAddSelectionToChat={composer.addTerminalSelectionAttachment} />
  )

  const overlays = (
    <>
      <RemoteDisplayBanner />
      {!isSecondaryWindow() && <DesktopInstallOverlay />}
      {!isSecondaryWindow() && (
        <DesktopOnboardingOverlay
          enabled={gatewayState === 'open'}
          onCompleted={() => {
            void refreshHermesConfig()
            void refreshCurrentModel()
            void queryClient.invalidateQueries({ queryKey: ['model-options'] })
          }}
          requestGateway={requestGateway}
        />
      )}
      <ModelPickerOverlay gateway={gatewayRef.current || undefined} onSelect={selectModel} />
      <SessionPickerOverlay onResume={resumeSession} />
      <ModelVisibilityOverlay gateway={gatewayRef.current || undefined} onOpenProviders={openProviderSettings} />
      <UpdatesOverlay />
      <GatewayConnectingOverlay />
      <BootFailureOverlay />
      <CommandPalette />
      <SessionSwitcher />
      <CreateProjectDialog
        onClose={() => setCreateProjectOpen(false)}
        onCreated={project => {
          navigate(projectRoute(project.id))
        }}
        open={createProjectOpen}
      />

      {settingsOpen && (
        <Suspense fallback={null}>
          <SettingsView
            gateway={gatewayRef.current}
            onClose={closeOverlayToPreviousRoute}
            onConfigSaved={() => {
              void refreshHermesConfig()
              void refreshCurrentModel()
              void queryClient.invalidateQueries({ queryKey: ['model-options'] })
            }}
            onMainModelChanged={(provider, model) => {
              setCurrentProvider(provider)
              setCurrentModel(model)
              updateModelOptionsCache(provider, model, true)
              void refreshCurrentModel()
              void queryClient.invalidateQueries({ queryKey: ['model-options'] })
            }}
          />
        </Suspense>
      )}

      {commandCenterOpen && (
        <Suspense fallback={null}>
          <CommandCenterView
            initialSection={commandCenterInitialSection}
            onClose={closeOverlayToPreviousRoute}
            onDeleteSession={removeSession}
            onNavigateRoute={path => navigate(path)}
            onOpenSession={sessionId => navigate(sessionRoute(sessionId))}
          />
        </Suspense>
      )}

      {agentsOpen && (
        <Suspense fallback={null}>
          <AgentsView onClose={closeOverlayToPreviousRoute} />
        </Suspense>
      )}

      {cronOpen && (
        <Suspense fallback={null}>
          <CronView
            onClose={closeOverlayToPreviousRoute}
            onOpenSession={sessionId => navigate(sessionRoute(sessionId))}
          />
        </Suspense>
      )}

      {profilesOpen && (
        <Suspense fallback={null}>
          <ProfilesView onClose={closeOverlayToPreviousRoute} />
        </Suspense>
      )}
    </>
  )

  const chatView = (
    <ChatView
      gateway={gatewayRef.current}
      maxVoiceRecordingSeconds={voiceMaxRecordingSeconds}
      modelMenuContent={modelMenuContent}
      onAddContextRef={composer.addContextRefAttachment}
      onAddUrl={url => composer.addContextRefAttachment(`@url:${formatRefValue(url)}`, url)}
      onAttachDroppedItems={composer.attachDroppedItems}
      onAttachImageBlob={composer.attachImageBlob}
      onBranchInNewChat={branchInNewChat}
      onCancel={cancelRun}
      onDeleteSelectedSession={() => {
        if (selectedStoredSessionId) {
          void removeSession(selectedStoredSessionId)
        }
      }}
      onDismissError={dismissError}
      onEdit={editMessage}
      onPasteClipboardImage={() => void composer.pasteClipboardImage()}
      onPickFiles={() => void composer.pickContextPaths('file')}
      onPickFolders={() => void composer.pickContextPaths('folder')}
      onPickImages={() => void composer.pickImages()}
      onReload={reloadFromMessage}
      onRemoveAttachment={id => void composer.removeAttachment(id)}
      onRestoreToMessage={restoreToMessage}
      onRetryResume={sessionId => void resumeSession(sessionId, true)}
      onSteer={steerPrompt}
      onSubmit={submitText}
      onThreadMessagesChange={handleThreadMessagesChange}
      onToggleSelectedPin={toggleSelectedPin}
      onTranscribeAudio={transcribeVoiceAudio}
    />
  )

  // Flipped layout mirrors the default: sessions sidebar → right, file
  // browser + preview rail → left. Same panes, swapped sides.
  const sidebarSide = panesFlipped ? 'right' : 'left'
  const railSide = panesFlipped ? 'left' : 'right'

  const previewPane = (
    <Pane
      disabled={!chatOpen || (!previewTarget && !filePreviewTarget)}
      id="preview"
      key="preview"
      maxWidth={PREVIEW_RAIL_MAX_WIDTH}
      minWidth={PREVIEW_RAIL_MIN_WIDTH}
      resizable
      side={railSide}
      width={PREVIEW_RAIL_PANE_WIDTH}
    >
      {chatOpen ? (
        <ChatPreviewRail onRestartServer={restartPreviewServer} setTitlebarToolGroup={setTitlebarToolGroup} />
      ) : null}
    </Pane>
  )

  const fileBrowserPane = (
    <Pane
      defaultOpen={false}
      // disabled={!chatOpen || currentView === 'project'}
      disabled={!chatOpen && currentView !== 'project'}
      forceCollapsed={narrowViewport}
      hoverReveal
      id="file-browser"
      key="file-browser"
      maxWidth={FILE_BROWSER_MAX_WIDTH}
      minWidth={FILE_BROWSER_MIN_WIDTH}
      resizable
      side={railSide}
      width={FILE_BROWSER_DEFAULT_WIDTH}
    >
      <RightSidebarPane
        onActivateFile={path => composer.insertContextPathInlineRef(path)}
        onActivateFolder={path => composer.insertContextPathInlineRef(path, true)}
        onChangeCwd={changeSessionCwd}
      />
    </Pane>
  )

  const terminalPane = (
    <Pane
      defaultOpen
      disabled={!terminalSidebarOpen}
      divider
      id="terminal-sidebar"
      key="terminal-sidebar"
      maxWidth="80vw"
      minWidth="22vw"
      resizable
      side={railSide}
      width="42vw"
    >
      <div className="relative flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-(--ui-editor-surface-background) pt-(--titlebar-height)">
        <TerminalSlot />
      </div>
    </Pane>
  )

  return (
    <AppShell
      leftStatusbarItems={leftStatusbarItems}
      leftTitlebarTools={titlebarToolGroups.flat.left}
      mainOverlays={mainOverlays}
      onOpenSettings={openSettings}
      overlays={overlays}
      previewPaneOpen={chatOpen && Boolean(previewTarget || filePreviewTarget)}
      statusbarItems={statusbarItems}
      terminalPaneOpen={terminalSidebarOpen}
      titlebarTools={titlebarToolGroups.flat.right}
    >
      {!isSecondaryWindow() && (
        <Pane
          forceCollapsed={narrowViewport}
          hoverReveal
          id="chat-sidebar"
          maxWidth={SIDEBAR_MAX_WIDTH}
          minWidth={SIDEBAR_DEFAULT_WIDTH}
          onOverlayActiveChange={setSidebarOverlayMounted}
          resizable
          side={sidebarSide}
          width={`${SIDEBAR_DEFAULT_WIDTH}px`}
        >
          {sidebar}
        </Pane>
      )}
      <PaneMain>
        <Routes>
          <Route element={chatView} index />
          <Route element={chatView} path=":sessionId" />
          <Route
            element={
              <Suspense fallback={null}>
                <SkillsView setStatusbarItemGroup={setStatusbarItemGroup} />
              </Suspense>
            }
            path="skills"
          />
          <Route
            element={
              <Suspense fallback={null}>
                <MessagingView setStatusbarItemGroup={setStatusbarItemGroup} />
              </Suspense>
            }
            path="messaging"
          />
          <Route
            element={
              <Suspense fallback={null}>
                <ArtifactsView setStatusbarItemGroup={setStatusbarItemGroup} />
              </Suspense>
            }
            path="artifacts"
          />
          <Route element={null} path="cron" />
          <Route element={null} path="profiles" />
          <Route element={null} path="settings" />
          <Route element={null} path="command-center" />
          <Route element={null} path="agents" />
          <Route element={<Navigate replace to={NEW_CHAT_ROUTE} />} path="new" />
          <Route element={<LegacySessionRedirect />} path="sessions/:sessionId" />
          <Route
            element={<ProjectPageViewRoute onStartSession={startSessionInWorkspace} />}
            path="project/:id"
          />
          <Route element={<Navigate replace to={NEW_CHAT_ROUTE} />} path="*" />
        </Routes>
      </PaneMain>
      {/*
        Order within a side maps to column order. Default (rail on the right):
        main | terminal | preview | file-browser. Flipped (rail on the left):
        mirror to file-browser | preview | terminal | main so terminal stays
        adjacent to the chat.
      */}
      {panesFlipped ? fileBrowserPane : terminalPane}
      {previewPane}
      {panesFlipped ? terminalPane : fileBrowserPane}
    </AppShell>
  )
}

function LegacySessionRedirect() {
  const { sessionId } = useParams()

  return <Navigate replace to={sessionId ? sessionRoute(sessionId) : NEW_CHAT_ROUTE} />
}

function ProjectPageViewRoute({ onStartSession }: { onStartSession: (path: string) => void }) {
  const { id } = useParams()

  if (!id) {
    return <Navigate replace to={NEW_CHAT_ROUTE} />
  }

  return <ProjectPageView onStartSession={onStartSession} projectId={decodeURIComponent(id)} />
}
