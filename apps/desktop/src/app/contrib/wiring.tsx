/**
 * Real-featureset wiring for the contrib (layout tree) root — the minimal
 * subset of DesktopController's hook chain that makes the REAL surfaces work:
 * gateway boot -> sessions list -> click-to-resume -> live transcript ->
 * composer send, plus the real terminal.
 *
 * The wired nodes (sidebar / chat routes / terminal) are exposed through
 * context; registered panes render `<WiredPane part="…"/>` to consume them.
 */

import { useStore } from '@nanostores/react'
import { useQueryClient } from '@tanstack/react-query'
import { type CSSProperties, lazy, type ReactNode, Suspense, useCallback, useEffect, useMemo, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { formatRefValue } from '@/components/assistant-ui/directive-text'
import { BootFailureOverlay } from '@/components/boot-failure-overlay'
import { DesktopInstallOverlay } from '@/components/desktop-install-overlay'
import { FindBar } from '@/components/find-bar'
import { GatewayConnectingOverlay } from '@/components/gateway-connecting-overlay'
import { NotificationStack } from '@/components/notifications'
import { DesktopOnboardingOverlay } from '@/components/onboarding'
import { $newSessionTabAction, registerPaneCloser } from '@/components/pane-shell/tree/store'
import { FloatingPet } from '@/components/pet/floating-pet'
import { RemoteDisplayBanner } from '@/components/remote-display-banner'
import { emitGatewayEvent } from '@/contrib/events'
import { getLatestSessionMessages } from '@/hermes'
import { type ChatMessage, chatMessageText, preserveLocalAssistantErrors, toChatMessages } from '@/lib/chat-messages'
import { isMessagingSource } from '@/lib/session-source'
import { latestSessionTodos } from '@/lib/todos'
import { activateWakeIndicator } from '@/lib/wake-indicator'
import { playWakeSound } from '@/lib/wake-sound'
import { $billingSettingsRequest } from '@/store/billing-block'
import { $desktopBoot } from '@/store/boot'
import { requestVoiceConversationStart } from '@/store/composer'
import { $cronReviewRequest, setCronFocusJobId } from '@/store/cron'
import { $pinnedSessionIds, pinSession, restoreWorktree, unpinSession } from '@/store/layout'
import { $previewTarget } from '@/store/preview'
import {
  $activeGatewayProfile,
  $freshSessionRequest,
  $profileScope,
  ALL_PROFILES,
  ensureGatewayProfile,
  newSessionInProfile,
  normalizeProfileKey,
  refreshActiveProfile
} from '@/store/profile'
import { $startWorkSessionRequest, followActiveSessionCwd } from '@/store/projects'
import {
  $activeSessionId,
  $connection,
  $currentCwd,
  $freshDraftReady,
  $gatewayState,
  $messages,
  $messagingSessions,
  $resumeExhaustedSessionId,
  $resumeFailedSessionId,
  $selectedStoredSessionId,
  $sessions,
  sessionMatchesStoredId,
  sessionPinId,
  setAwaitingResponse,
  setBusy,
  setMessages
} from '@/store/session'
import { clearSessionTodos, setSessionTodos, todosForHydration } from '@/store/todos'
import { armWakeWord, stopClientCapture } from '@/store/wake-word'
import { isAuxiliaryWindow, isHudWindow } from '@/store/windows'
import { useSkinCommand } from '@/themes/use-skin-command'

import { closeWorkspaceTab } from '../chat/close-tab'
import { requestComposerInsert } from '../chat/composer/focus'
import { useComposerActions } from '../chat/hooks/use-composer-actions'
import { CommandPalette } from '../command-palette'
import { triggerAndRefreshCronJobs } from '../cron/cron-actions'
import { useGatewayBoot } from '../gateway/hooks/use-gateway-boot'
import { useGatewayRequest } from '../gateway/hooks/use-gateway-request'
import { useKeybinds } from '../hooks/use-keybinds'
import { useHudHandoff } from '../hud/handoff'
import { ModelPickerOverlay } from '../model-picker-overlay'
import { ModelVisibilityOverlay } from '../model-visibility-overlay'
import { mainChatOccupied, openSession } from '../open-session'
import { PetGenerateOverlay } from '../pet-generate/pet-generate-overlay'
import { FileActionDialogs } from '../right-sidebar/file-actions'
import { RemoteFolderPicker } from '../right-sidebar/files/remote-picker'
import { resetProjectTreeState } from '../right-sidebar/files/use-project-tree'
import { PersistentTerminal } from '../right-sidebar/terminal/persistent'
import { closeAllTerminals } from '../right-sidebar/terminal/terminals'
import {
  CRON_ROUTE,
  navigateToWorkspacePage,
  routeSessionId,
  sessionRoute,
  SETTINGS_ROUTE,
  syncWorkspaceRoute
} from '../routes'
import { SessionPickerOverlay } from '../session-picker-overlay'
import { SessionSwitcher } from '../session-switcher'
import { useBackgroundQueueDrain } from '../session/hooks/use-background-queue-drain'
import { useContextSuggestions } from '../session/hooks/use-context-suggestions'
import { useCwdActions } from '../session/hooks/use-cwd-actions'
import { useHermesConfig } from '../session/hooks/use-hermes-config'
import { useMessageStream } from '../session/hooks/use-message-stream'
import { useModelControls } from '../session/hooks/use-model-controls'
import { usePreviewRouting } from '../session/hooks/use-preview-routing'
import { usePromptActions } from '../session/hooks/use-prompt-actions'
import { useRouteResume } from '../session/hooks/use-route-resume'
import { useSessionActions } from '../session/hooks/use-session-actions'
import { useSessionListActions } from '../session/hooks/use-session-list-actions'
import { useSessionStateCache } from '../session/hooks/use-session-state-cache'
import { startWorkspaceSession } from '../session/workspace-session-target'
import { useOverlayRouting } from '../shell/hooks/use-overlay-routing'
import { useWindowControlsOverlayWidth } from '../shell/hooks/use-window-controls-overlay-width'
import {
  titlebarControlsPosition,
  titlebarControlsYNudge,
  titlebarToolsRightCss,
  titlebarToolsWidthCss
} from '../shell/titlebar'
import { TitlebarControls } from '../shell/titlebar-controls'
import { UpdatesOverlay } from '../updates-overlay'

import { ContribWiringContext } from './context'
import {
  reconcileActiveTranscript,
  resolveActiveTranscriptSession,
  useBackgroundSync
} from './hooks/use-background-sync'
import { useDesktopIntegrations } from './hooks/use-desktop-integrations'
import { usePetBridge } from './hooks/use-pet-bridge'
import { useQuickEntryBridge } from './hooks/use-quick-entry-bridge'
import { useSessionTileDelegate } from './hooks/use-session-tile-delegate'
import { $restartPreviewServer, useTitlebarToolContributions } from './panes'
import { ChatRoutesSurface, SidebarSurface, StatusbarSurface, TerminalSurface } from './surfaces'
import type { WiringActions, WiringApi } from './types'

// Overlay views the controller mounts over the shell — lazy, load on demand.
// The workspace-route full-page views (skills/messaging/artifacts) are the
// ChatRoutesSurface's and live in ./surfaces.
const AgentsView = lazy(async () => ({ default: (await import('../agents')).AgentsView }))
const CommandCenterView = lazy(async () => ({ default: (await import('../command-center')).CommandCenterView }))
const CronView = lazy(async () => ({ default: (await import('../cron')).CronView }))
const WebhooksView = lazy(async () => ({ default: (await import('../webhooks')).WebhooksView }))
const ProfilesView = lazy(async () => ({ default: (await import('../profiles')).ProfilesView }))
const SettingsView = lazy(async () => ({ default: (await import('../settings')).SettingsView }))
const StarmapView = lazy(async () => ({ default: (await import('../starmap')).StarmapView }))

// Surfaces (the four wired panes), the render context + WiredPane, and the
// WiringActions/WiringApi contracts all live in sibling modules — this file is
// the controller that assembles them.
export { WiredPane } from './context'

export function ContribWiring({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient()
  const location = useLocation()
  const navigate = useNavigate()

  const busyRef = useRef(false)
  const creatingSessionRef = useRef(false)
  // Billing recovery routes to Settings → Billing from surfaces without router
  // context (the sticky toast). The shell owns `navigate`, so it consumes the
  // intent counter here; the ref skips the initial mount value.
  const billingSettingsSeenRef = useRef(0)
  const cronReviewSeenRef = useRef(0)
  const activeTranscriptSignatureRef = useRef(new Map<string, string>())
  const activeTranscriptRequestSequenceRef = useRef(0)
  // Stable identity for the whole callback surface (see WiringActions). Mutated
  // in place each render so memoized surfaces never re-render on churn.
  const actionsRef = useRef<WiringActions | null>(null)

  const gatewayState = useStore($gatewayState)
  const activeSessionId = useStore($activeSessionId)
  const billingSettingsRequest = useStore($billingSettingsRequest)
  const cronReviewRequest = useStore($cronReviewRequest)
  const currentCwd = useStore($currentCwd)

  // eslint-disable-next-line no-restricted-syntax -- one-shot request-seen sentinel, not an atom mirror
  useEffect(() => {
    if (billingSettingsRequest === billingSettingsSeenRef.current) {
      return
    }

    billingSettingsSeenRef.current = billingSettingsRequest

    if (billingSettingsRequest > 0) {
      navigate(`${SETTINGS_ROUTE}?tab=billing`)
    }
  }, [billingSettingsRequest, navigate])

  // eslint-disable-next-line no-restricted-syntax -- one-shot request-seen sentinel, not an atom mirror
  useEffect(() => {
    if (cronReviewRequest === cronReviewSeenRef.current) {
      return
    }

    cronReviewSeenRef.current = cronReviewRequest

    if (cronReviewRequest > 0) {
      navigate(CRON_ROUTE)
    }
  }, [cronReviewRequest, navigate])
  const freshDraftReady = useStore($freshDraftReady)
  const resumeFailedSessionId = useStore($resumeFailedSessionId)
  const resumeExhaustedSessionId = useStore($resumeExhaustedSessionId)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const messagingSessions = useStore($messagingSessions)
  const sessions = useStore($sessions)
  const activeGatewayProfile = useStore($activeGatewayProfile)
  const profileScope = useStore($profileScope)
  const boot = useStore($desktopBoot)

  const routedSessionId = routeSessionId(location.pathname)
  const routedSessionIdRef = useRef(routedSessionId)

  routedSessionIdRef.current = routedSessionId
  const routeToken = `${location.pathname}:${location.search}:${location.hash}`
  const routeTokenRef = useRef(routeToken)
  routeTokenRef.current = routeToken
  const getRouteToken = useCallback(() => routeTokenRef.current, [])

  const getRoutedStoredSessionId = useCallback(() => routedSessionIdRef.current, [])

  const clearRoutedSessionIntent = useCallback(() => {
    routedSessionIdRef.current = null
  }, [])

  // Point the workspace at the route: the pane contribution re-registers
  // headerVeto from $workspaceIsPage (so the main zone's tab bar stands down
  // on pages), and a page route fronts the pane so it can't stay stuck behind
  // a focused session tile.
  useEffect(() => {
    syncWorkspaceRoute(location.pathname)
  }, [location.pathname])

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
    openStarmap,
    profilesOpen,
    resetOverlayReturnRoute,
    settingsOpen,
    starmapOpen,
    toggleCommandCenter,
    webhooksOpen
  } = useOverlayRouting()

  const {
    activeSessionIdRef,
    ensureSessionState,
    getRuntimeIdForStoredSession,
    resetViewSync,
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

  const { connectionRef, gateway, gatewayRef, requestGateway } = useGatewayRequest()

  const { loadMoreMessagingForPlatform, loadMoreSessions, refreshCronJobs, refreshMessagingSessions, refreshSessions } =
    useSessionListActions({ profileScope })

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

  const { refreshProjectBranch } = useCwdActions({
    activeSessionIdRef,
    onSessionRuntimeInfo: updateActiveSessionRuntimeInfo,
    requestGateway
  })

  const { refreshHermesConfig, sttEnabled, voiceMaxRecordingSeconds } = useHermesConfig({ activeSessionIdRef })

  const { applySavedMainModel, refreshCurrentModel, selectModel } = useModelControls({
    queryClient,
    requestGateway
  })

  const openProviderSettings = useCallback(() => navigate(`${SETTINGS_ROUTE}?tab=providers`), [navigate])

  // Palette "Keyboard shortcuts" entry dispatches a custom event (contributions
  // don't have router access); listen and navigate to the settings keybinds tab.
  useEffect(() => {
    const onOpenKeybinds = () => navigate(`${SETTINGS_ROUTE}?tab=keybinds`)
    window.addEventListener('hermes:open-keybinds', onOpenKeybinds)

    return () => window.removeEventListener('hermes:open-keybinds', onOpenKeybinds)
  }, [navigate])

  // Dev-only: install the credit-notice demo trigger (Ctrl+Shift+C / ⌘K palette
  // / window.__creditsDemo). Dynamic import inside the DEV guard so the module
  // is dropped from production builds.
  useEffect(() => {
    if (!import.meta.env.DEV) {
      return
    }

    let dispose: (() => void) | undefined

    void import('./dev/credits-notice-demo').then(m => {
      dispose = m.installCreditsNoticeDemo()
    })

    return () => dispose?.()
  }, [])

  // Post-turn rehydrate from stored history (same behavior as DesktopController,
  // including finished-todos restoration).
  const hydrateFromStoredSession = useCallback(
    async (
      attempts = 1,
      storedSessionId = selectedStoredSessionIdRef.current,
      runtimeSessionId = activeSessionIdRef.current
    ) => {
      if (!storedSessionId || !runtimeSessionId) {
        return
      }

      const storedProfile = $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))?.profile

      for (let index = 0; index < Math.max(1, attempts); index += 1) {
        try {
          const latest = await getLatestSessionMessages(storedSessionId, storedProfile)
          const messages = toChatMessages(latest.messages)
          updateSessionState(
            runtimeSessionId,
            state => ({ ...state, messages: preserveLocalAssistantErrors(messages, state.messages) }),
            storedSessionId
          )

          const restored = todosForHydration(latestSessionTodos(messages))

          if (restored) {
            setSessionTodos(runtimeSessionId, restored)
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

  // Refresh any active transcript changed by another process. Signature-gated
  // so a no-change event does not churn the thread.
  const refreshActiveTranscript = useCallback(
    () =>
      reconcileActiveTranscript({
        activeSessionIdRef,
        busyRef,
        requestSequenceRef: activeTranscriptRequestSequenceRef,
        resolveSession: resolveActiveTranscriptSession,
        selectedStoredSessionIdRef,
        signatureRef: activeTranscriptSignatureRef,
        updateSessionState
      }),
    [activeSessionIdRef, busyRef, selectedStoredSessionIdRef, updateSessionState]
  )

  const { handleGatewayEvent } = useMessageStream({
    activeGatewayProfile,
    activeSessionIdRef,
    hydrateFromStoredSession,
    queryClient,
    refreshHermesConfig,
    refreshSessions,
    sessionStateByRuntimeIdRef,
    updateSessionState
  })

  // Agent-driven preview routing (agent opens a URL/file -> the preview rail
  // follows) + the preview server restart handler, layered over the base
  // gateway event stream exactly like DesktopController.
  const { handleDesktopGatewayEvent, restartPreviewServer } = usePreviewRouting({
    baseHandleGatewayEvent: handleGatewayEvent,
    currentCwd,
    requestGateway
  })

  // Composer @-mention context suggestions (files/dirs under the cwd).
  useContextSuggestions({
    activeSessionId,
    activeSessionIdRef,
    currentCwd,
    gatewayState,
    requestGateway
  })

  // Expose the restart handler to the preview pane contribution (module
  // boundary crossed via atom — contrib-panes can't import this file).
  useEffect(() => {
    $restartPreviewServer.set(restartPreviewServer)

    return () => $restartPreviewServer.set(null)
  }, [restartPreviewServer])

  const {
    archiveSession,
    branchCurrentSession,
    branchStoredSession,
    createBackendSessionForSend,
    openNewSessionTile,
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
    getRoutedStoredSessionId,
    navigate,
    onFreshDraftRouteIntent: clearRoutedSessionIntent,
    requestGateway,
    resetViewSync,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    syncSessionStateToView,
    updateSessionState
  })

  // A profile switch/create drops to a fresh new-session draft so the
  // previously open session doesn't bleed across contexts. Skip initial value.
  const freshSessionRequest = useStore($freshSessionRequest)
  const lastFreshRef = useRef(freshSessionRequest)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (freshSessionRequest === lastFreshRef.current) {
      return
    }

    lastFreshRef.current = freshSessionRequest
    startFreshSessionDraft()
  }, [freshSessionRequest, startFreshSessionDraft])

  // Swapping the live gateway to another profile must re-pull that profile's
  // global model + active-profile pill (both are nanostores — the blanket
  // invalidateQueries on swap doesn't touch them).
  const lastGatewayProfileRef = useRef(activeGatewayProfile)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (activeGatewayProfile === lastGatewayProfileRef.current) {
      return
    }

    lastGatewayProfileRef.current = activeGatewayProfile
    // Force: the new profile has its own defaults, so reseed the selector even
    // if the composer already shows values from the previous profile. Both
    // refreshes carry an intent token so a picker click made in flight wins.
    void refreshCurrentModel(true)
    void refreshHermesConfig(true)
    void refreshActiveProfile()
  }, [activeGatewayProfile, refreshCurrentModel, refreshHermesConfig])

  // New session anchored to a workspace. Seeds cwd + branch from the clicked
  // workspace; an explicit worktree path also drills the sidebar into that
  // project so the new lane is visible.
  //
  // `openTab` is the sidebar "+" behavior: once a chat is loaded, stack a new
  // tab instead of replacing it (see mainChatOccupied). The composer's
  // "branch off into a new worktree" flow keeps the fresh-draft path — it
  // prefills the MAIN composer right after, so it has to own that surface.
  const startSessionInWorkspace = useCallback(
    (path: null | string, options?: { openTab?: boolean }) => {
      if (options?.openTab && mainChatOccupied(activeSessionIdRef.current, $selectedStoredSessionId.get())) {
        void openNewSessionTile('center', { cwd: path, listed: false })

        return
      }

      startWorkspaceSession({
        activeSessionIdRef,
        followActiveSessionCwd,
        onExplicitWorkspace: restoreWorktree,
        path,
        requestGateway,
        startFreshSessionDraft
      })
    },
    [activeSessionIdRef, openNewSessionTile, requestGateway, startFreshSessionDraft]
  )

  // Composer "branch off into a new worktree": open a fresh session anchored
  // to the just-created tree, then prefill the task that kicked it off.
  const startWorkSessionRequest = useStore($startWorkSessionRequest)
  const lastStartWorkTokenRef = useRef(startWorkSessionRequest?.token ?? 0)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (!startWorkSessionRequest || startWorkSessionRequest.token === lastStartWorkTokenRef.current) {
      return
    }

    lastStartWorkTokenRef.current = startWorkSessionRequest.token
    startSessionInWorkspace(startWorkSessionRequest.path, { openTab: startWorkSessionRequest.openTab })

    if (startWorkSessionRequest.draft) {
      requestComposerInsert(startWorkSessionRequest.draft, { target: 'main' })
    }
  }, [startSessionInWorkspace, startWorkSessionRequest])

  const composer = useComposerActions({ activeSessionId, currentCwd, requestGateway })

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

  const handleSkinCommand = useSkinCommand()

  const {
    cancelRun,
    editMessage,
    executeSlashCommand,
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
    getRoutedStoredSessionId,
    getRuntimeIdForStoredSession,
    getRouteToken,
    handleSkinCommand,
    openMemoryGraph: openStarmap,
    refreshSessions,
    requestGateway,
    resumeStoredSession: resumeSession,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionIdRef,
    startFreshSessionDraft,
    sttEnabled,
    updateSessionState
  })

  // Runs outside the selected ChatBar so queues belonging to background
  // sessions continue once those sessions are idle.
  useBackgroundQueueDrain({
    enabled: gatewayState === 'open',
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId,
    submitText
  })

  // Session-tile delegate (resume/submit/interrupt/slash + the session verbs
  // the tile TAB menu needs, without touching the primary view).
  useSessionTileDelegate({
    archiveSession,
    branchStoredSession,
    executeSlashCommand,
    removeSession,
    requestGateway,
    runtimeIdByStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    updateSessionState
  })

  // The popped-out pet overlay's bridge back into the app.
  usePetBridge({ requestGateway, resumeSession, submitText })

  // The global-hotkey Quick Entry window's bridge: its captured text rides the
  // SAME submit machinery the normal composer uses (current chat / picked
  // session / new session), and it hears gateway truth from this window.
  useQuickEntryBridge({ startFreshSessionDraft, submitText })

  // Leaving HUD mode hands this window the session back (see hud/handoff).
  useHudHandoff({ navigate, resumeSession })

  // Clear a failed turn's red error banner. Errors are renderer-local (never
  // persisted): a bare error placeholder is dropped entirely; a partial-output
  // failure keeps its content and sheds the error. Both the runtime cache AND
  // the live $messages view must be updated — preserveLocalAssistantErrors
  // re-grafts any still-errored view message on the next session.info flush.
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

      // View first: the cache update below triggers a re-sync that reads
      // $messages as the error-preservation baseline.
      setMessages(clearErrorIn($messages.get()))

      updateSessionState(runtimeSessionId, state => ({
        ...state,
        messages: clearErrorIn(state.messages)
      }))
    },
    [activeSessionIdRef, updateSessionState]
  )

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

  // Plugins hear the stream FIRST (isolated fan-out in contrib/events), then
  // the app dispatches as before — a plugin listener can't affect app flow.
  const handleGatewayEventWithPlugins = useCallback(
    (event: Parameters<typeof handleDesktopGatewayEvent>[0]) => {
      emitGatewayEvent(event)

      if (event.type === 'wake.detected') {
        const payload = event.payload as { profile?: null | string; start_new_session?: boolean } | undefined

        // Free the Mac mic so voice conversation can open getUserMedia.
        // Server already pauses the detector lease; this stops client PCM feed.
        stopClientCapture()

        // Audible confirmation that the wake registered, before voice capture
        // starts. Gated by the shared sound-mute toggle.
        playWakeSound()
        activateWakeIndicator()

        // Multi-profile routing: a wake phrase enrolled by another profile
        // re-homes the gateway to that profile first (live swap — same path
        // as clicking it in the profile rail), then opens the fresh session
        // and starts voice there.
        const targetProfile = payload?.profile?.trim()
        const activeProfile = normalizeProfileKey($activeGatewayProfile.get())

        if (targetProfile && normalizeProfileKey(targetProfile) !== activeProfile) {
          if (payload?.start_new_session !== false) {
            newSessionInProfile(targetProfile)
          } else {
            void ensureGatewayProfile(normalizeProfileKey(targetProfile))
          }
        } else if (payload?.start_new_session !== false) {
          startFreshSessionDraft()
        }

        requestVoiceConversationStart()

        return
      }

      handleDesktopGatewayEvent(event)
    },
    [handleDesktopGatewayEvent, startFreshSessionDraft]
  )

  useGatewayBoot({
    beforeConnectionSwitch: () => {
      startFreshSessionDraft({ preserveRoute: true, workspaceTarget: null })
      resetOverlayReturnRoute()
      resetProjectTreeState()
      closeAllTerminals()
    },
    handleGatewayEvent: handleGatewayEventWithPlugins,
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
      // Status-then-arm, syncing $wakeWord so the composer toggle reflects the
      // same listener this auto-arm claims.
      void armWakeWord(requestGateway)
    }
  }, [gatewayState, requestGateway])

  const activeIsMessaging =
    !!selectedStoredSessionId &&
    isMessagingSource(messagingSessions.find(s => sessionMatchesStoredId(s, selectedStoredSessionId))?.source)

  // sessions.changed refreshes every open transcript; only messaging retains
  // the periodic safety-net it already had before this fix.
  // Keep app data live while the gateway is open (on-connect reseed + the
  // cron / messaging / transcript visibility polls + fresh-draft reseed).
  useBackgroundSync({
    activeGatewayProfile,
    activeIsMessaging,
    activeSessionId,
    activeStoredSessionId: selectedStoredSessionId,
    freshDraftReady,
    gatewayState,
    refreshActiveTranscript,
    refreshCronJobs,
    refreshCurrentModel,
    refreshHermesConfig,
    refreshMessagingSessions,
    refreshSessions,
    requestGateway
  })

  // Electron-main / OS / cross-window integrations: update polling, ⌘W close,
  // deep links, native-notification nav, preview-shortcut enablement,
  // remembered-session restore, and cross-window session-list sync.
  const previewTarget = useStore($previewTarget)

  useDesktopIntegrations({
    activeProfile: normalizeProfileKey(activeGatewayProfile),
    chatOpen,
    hasPreview: Boolean(previewTarget),
    locationPathname: location.pathname,
    navigate,
    profileReady: boot.phase === 'renderer.ready',
    refreshSessions,
    resumeExhaustedSessionId,
    routedSessionId,
    runtimeIdByStoredSessionId: runtimeIdByStoredSessionIdRef,
    sessions
  })

  // Pin/unpin the selected session (statusbar keybind + chat header) — pinned
  // on the durable lineage-root id so it survives auto-compression.
  const toggleSelectedPin = useCallback(() => {
    const sessionId = $selectedStoredSessionId.get()

    if (!sessionId) {
      return
    }

    const session = $sessions.get().find(s => sessionMatchesStoredId(s, sessionId))
    const pinId = session ? sessionPinId(session) : sessionId

    if ($pinnedSessionIds.get().includes(pinId)) {
      unpinSession(pinId)
    } else {
      pinSession(pinId)
    }
  }, [])

  // The tab-strip "+" and ⌘T share one action: open a new session as its own
  // tab (stacked into the workspace zone) WITHOUT polluting the session list.
  // Created `listed: false`, so each new tab's in-memory session stays out of
  // the sidebar until its first message persists a turn and a refresh surfaces
  // it — Cursor-style. Every click opens a fresh "New session" tab (multiple
  // empty tabs are fine since none touch the session list).
  const openNewSessionTab = useCallback(() => {
    void openNewSessionTile('center', { listed: false })
  }, [openNewSessionTile])

  // Archive the selected session (rebindable `session.archive` hotkey).
  const archiveSelectedSession = useCallback(() => {
    const sessionId = $selectedStoredSessionId.get()

    if (!sessionId) {
      return
    }

    void archiveSession(sessionId)
  }, [archiveSession])

  // Single global listener for every rebindable hotkey plus the on-screen
  // keybind editor's capture mode (same as DesktopController).
  useKeybinds({
    archiveSelectedSession,
    openNewSessionTab,
    startFreshSession: startFreshSessionDraft,
    toggleCommandCenter,
    toggleSelectedPin
  })

  // Register the tab-strip "+" action (the generic renderer stays
  // session-agnostic; null until wired hides the glyph).
  useEffect(() => {
    $newSessionTabAction.set(openNewSessionTab)

    return () => $newSessionTabAction.set(null)
  }, [openNewSessionTab])

  // The MAIN tab's Close. The workspace pane can't leave the tree, so its
  // closer empties it instead: the next stacked session shifts in, else main
  // drops to a fresh draft. Registering it here is also what gives the tab its
  // close GESTURE (⌘-click / middle-click) — the strip reads the closer, not
  // the `uncloseable` flag, so the pane stays undismissable either way.
  useEffect(() => {
    registerPaneCloser('workspace', () => void closeWorkspaceTab(id => navigate(sessionRoute(id))))

    return () => registerPaneCloser('workspace')
  }, [navigate])

  // The controller's entire callback surface, gathered into the stable
  // `actions` bag. `nextActions` is TS-checked against WiringActions each
  // render; its fields are copied into the ref object so `actions` keeps one
  // identity for the app's life (memoized surfaces don't re-render on churn)
  // while every handler still closes over the latest values.
  const nextActions: WiringActions = {
    onAddContextRef: composer.addContextRefAttachment,
    onAddUrl: url => composer.addContextRefAttachment(`@url:${formatRefValue(url)}`, url),
    onArchiveSession: sessionId => void archiveSession(sessionId),
    onAttachDroppedItems: composer.attachDroppedItems,
    onAttachImageBlob: composer.attachImageBlob,
    onAttachPrCommentUrl: composer.attachPrCommentUrl,
    onBranchInNewChat: messageId => void branchInNewChat(messageId),
    onBranchSession: sessionId => void branchStoredSession(sessionId),
    onCancel: cancelRun,
    onDeleteSelectedSession: () => {
      const id = $selectedStoredSessionId.get()

      if (id) {
        void removeSession(id)
      }
    },
    onDeleteSession: sessionId => void removeSession(sessionId),
    onDismissError: dismissError,
    onEdit: editMessage,
    onLoadMoreMessaging: loadMoreMessagingForPlatform,
    onLoadMoreSessions: loadMoreSessions,
    onManageCronJob: jobId => {
      setCronFocusJobId(jobId)
      navigate(CRON_ROUTE)
    },
    onNavigate: selectSidebarItem,
    onNewSessionInWorkspace: path => startSessionInWorkspace(path, { openTab: true }),
    onNewSessionSplit: dir => void openNewSessionTile(dir),
    onPasteClipboardImage: opts => composer.pasteClipboardImage(opts),
    onPickFiles: () => void composer.pickContextPaths('file'),
    onPickFolders: () => void composer.pickContextPaths('folder'),
    onPickImages: () => void composer.pickImages(),
    onReload: reloadFromMessage,
    onRemoveAttachment: id => void composer.removeAttachment(id),
    onRestoreToMessage: restoreToMessage,
    // Already on screen (open tile, or the main session)? Jump to its tab;
    // otherwise load it into main. Same door every other session link uses.
    onResumeSession: sessionId => openSession(sessionId, navigate),
    onRetryResume: sessionId => void resumeSession(sessionId, true),
    onSteer: steerPrompt,
    onSubmit: submitText,
    onThreadMessagesChange: handleThreadMessagesChange,
    onToggleSelectedPin: toggleSelectedPin,
    onTranscribeAudio: transcribeVoiceAudio,
    onTriggerCronJob: jobId =>
      triggerAndRefreshCronJobs(jobId, profileScope === ALL_PROFILES ? 'all' : profileScope)
        .then(() => undefined)
        .catch(() => undefined),
    getGateway: () => gatewayRef.current,
    openAgents,
    openCommandCenterSection,
    requestGateway,
    selectModel,
    toggleCommandCenter
  }

  if (actionsRef.current) {
    Object.assign(actionsRef.current, nextActions)
  } else {
    actionsRef.current = nextActions
  }

  const actions = actionsRef.current

  // Each pane node is memoized on ONLY the reactive inputs it truly consumes;
  // everything else reaches its surface through `actions` (stable) or the
  // surface's own atom subscriptions. A wiring tick that doesn't touch a
  // node's keys leaves its element reference intact, so `WiredPane` (memoized)
  // bails on that pane subtree — panes render independently of one another.
  const sidebarNode = useMemo(
    () => <SidebarSurface actions={actions} currentView={currentView} />,
    [actions, currentView]
  )

  const terminalNode = useMemo(() => <TerminalSurface />, [])

  const statusbarNode = useMemo(
    () => (
      <StatusbarSurface
        actions={actions}
        agentsOpen={agentsOpen}
        chatOpen={chatOpen}
        commandCenterOpen={commandCenterOpen}
      />
    ),
    [actions, agentsOpen, chatOpen, commandCenterOpen]
  )

  // The voice cap changes only on config load; the gateway instance + all
  // chat reactivity are subscribed inside ChatRoutesSurface / ChatView.
  const chatRoutesNode = useMemo(
    () => <ChatRoutesSurface actions={actions} maxVoiceRecordingSeconds={voiceMaxRecordingSeconds} />,
    [actions, voiceMaxRecordingSeconds]
  )

  const api = useMemo<WiringApi>(
    () => ({
      chatRoutes: chatRoutesNode,
      sidebar: sidebarNode,
      statusbar: statusbarNode,
      terminal: terminalNode
    }),
    [chatRoutesNode, sidebarNode, statusbarNode, terminalNode]
  )

  // The REAL titlebar tool clusters (sidebar/flip toggles, haptics, keybinds,
  // settings gear) — fixed chrome positioned via the same CSS vars AppShell
  // sets, computed here from the live connection. Page-registered tools
  // (preview's monitor/devtools cluster, …) arrive as registry contributions.
  const leftTitlebarTools = useTitlebarToolContributions('left')
  const rightTitlebarTools = useTitlebarToolContributions('right')
  const connection = useStore($connection)
  const controlsPos = titlebarControlsPosition(connection?.windowButtonPosition, Boolean(connection?.isFullscreen))
  // Windows/WSLg reserve native min/max/close on the right (AppShell parity:
  // prefer the live WCO measurement, fall back to the static reservation).
  const measuredOverlayWidth = useWindowControlsOverlayWidth()
  const nativeOverlayWidth = measuredOverlayWidth ?? connection?.nativeOverlayWidth ?? 0

  const titlebarChrome = {
    darwinMajor: connection?.darwinMajor ?? 0,
    isFullscreen: Boolean(connection?.isFullscreen),
    windowButtonPosition: connection?.windowButtonPosition
  }

  const titlebarToolsRight = titlebarToolsRightCss(nativeOverlayWidth, titlebarChrome)
  // Pane-registered tools (preview's monitor/devtools cluster) anchor flush
  // against the static system cluster — in the tree layout the titlebar band
  // sits ABOVE the grid, so AppShell's pane-width anchoring doesn't apply.
  // Count every button the static cluster actually renders: four systemTools
  // (layout, haptics, keybinds, settings) PLUS the always-present
  // right-sidebar toggle (see titlebar-controls.tsx). A shared width that
  // under-counts leaves the find bar, the titlebar header padding, and the
  // pane-cluster anchor overlapping the fifth button.
  const SYSTEM_TOOL_COUNT = 5
  const paneToolCount = rightTitlebarTools.filter(tool => !tool.hidden).length
  const systemToolsWidth = titlebarToolsWidthCss(SYSTEM_TOOL_COUNT)

  const titlebarToolsWidth =
    paneToolCount > 0 ? `calc(${systemToolsWidth} + ${titlebarToolsWidthCss(paneToolCount)})` : systemToolsWidth

  return (
    <ContribWiringContext.Provider value={api}>
      <div
        className="contents"
        style={
          {
            '--titlebar-controls-left': `${controlsPos.left}px`,
            '--titlebar-controls-top': `${controlsPos.top}px`,
            '--titlebar-controls-y-nudge': titlebarControlsYNudge(titlebarChrome),
            '--titlebar-tools-right': titlebarToolsRight,
            '--titlebar-tools-width': titlebarToolsWidth,
            '--shell-preview-toolbar-gap': systemToolsWidth
          } as CSSProperties
        }
      >
        {/* HUD mode has no titlebar to hang these off — the clusters are
            `fixed`, so without this they'd float over the chat as orphaned
            buttons. Exits are the ⌘⇧H toggle and ⌘W. */}
        {!isHudWindow() && (
          <TitlebarControls
            leftTools={leftTitlebarTools}
            onOpenSettings={() => navigate(SETTINGS_ROUTE)}
            tools={rightTitlebarTools}
          />
        )}
        {children}
      </div>

      {/* The full real overlay set (mirrors DesktopController's `overlays`). */}
      <RemoteDisplayBanner />
      {!isAuxiliaryWindow() && <DesktopInstallOverlay />}
      {!isAuxiliaryWindow() && (
        <DesktopOnboardingOverlay
          enabled={gatewayState === 'open'}
          onCompleted={() => {
            void refreshHermesConfig()
            void refreshCurrentModel()
            void queryClient.invalidateQueries({ queryKey: ['model-options'] })
          }}
          profile={activeGatewayProfile}
          requestGateway={requestGateway}
        />
      )}
      <ModelPickerOverlay gateway={gateway || undefined} onSelect={selectModel} profile={activeGatewayProfile} />
      <SessionPickerOverlay onResume={sessionId => openSession(sessionId, navigate)} />
      <ModelVisibilityOverlay
        gateway={gateway || undefined}
        onOpenProviders={openProviderSettings}
        profile={activeGatewayProfile}
      />
      <UpdatesOverlay />
      <GatewayConnectingOverlay />
      <BootFailureOverlay />
      <CommandPalette />
      <PetGenerateOverlay />
      <SessionSwitcher />
      <FileActionDialogs />
      <RemoteFolderPicker />
      <FindBar />

      {settingsOpen && (
        <Suspense fallback={null}>
          <SettingsView
            gateway={gateway}
            onClose={closeOverlayToPreviousRoute}
            onConfigSaved={() => {
              void refreshHermesConfig()
              void refreshCurrentModel()
              void queryClient.invalidateQueries({ queryKey: ['model-options'] })
            }}
            onMainModelChanged={(provider, model) => {
              applySavedMainModel(provider, model)
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
            onNavigateRoute={path => navigateToWorkspacePage(navigate, path)}
            onOpenSession={sessionId => openSession(sessionId, navigate)}
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
            onOpenSession={sessionId => openSession(sessionId, navigate)}
          />
        </Suspense>
      )}

      {webhooksOpen && (
        <Suspense fallback={null}>
          <WebhooksView onClose={closeOverlayToPreviousRoute} />
        </Suspense>
      )}

      {profilesOpen && (
        <Suspense fallback={null}>
          <ProfilesView onClose={closeOverlayToPreviousRoute} />
        </Suspense>
      )}

      {starmapOpen && (
        <Suspense fallback={null}>
          <StarmapView onClose={closeOverlayToPreviousRoute} />
        </Suspense>
      )}

      {/* Toasts above everything. */}
      <NotificationStack />

      {/* Petdex floating mascot — renders nothing unless installed + enabled.
          Never in the HUD: that window is the chat bar and nothing else. */}
      {!isHudWindow() && <FloatingPet />}

      {/* Single persistent xterm host chasing the terminal pane's slot rect.
          The HUD has no terminal pane, so it has nothing to chase. */}
      {!isHudWindow() && <PersistentTerminal onAddSelectionToChat={composer.addTerminalSelectionAttachment} />}
    </ContribWiringContext.Provider>
  )
}
