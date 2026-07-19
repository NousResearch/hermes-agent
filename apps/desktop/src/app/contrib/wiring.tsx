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
import { useLocation, useNavigate } from 'react-router-dom'

import { formatRefValue } from '@/components/assistant-ui/directive-text'
import { BootFailureOverlay } from '@/components/boot-failure-overlay'
import { DesktopInstallOverlay } from '@/components/desktop-install-overlay'
import { GatewayConnectingOverlay } from '@/components/gateway-connecting-overlay'
import { NotificationStack } from '@/components/notifications'
import { DesktopOnboardingOverlay } from '@/components/onboarding'
import { FloatingPet } from '@/components/pet/floating-pet'
import { RemoteDisplayBanner } from '@/components/remote-display-banner'
import { emitGatewayEvent } from '@/contrib/events'
import { getSessionMessages, triggerCronJob } from '@/hermes'
import { type ChatMessage, chatMessageText, preserveLocalAssistantErrors, toChatMessages } from '@/lib/chat-messages'
import { sessionMessagesSignature } from '@/lib/session-signatures'
import { isMessagingSource } from '@/lib/session-source'
import { latestSessionTodos } from '@/lib/todos'
import { setCronFocusJobId } from '@/store/cron'
import { $pinnedSessionIds, pinSession, restoreWorktree, unpinSession } from '@/store/layout'
import { $filePreviewTarget, $previewTarget } from '@/store/preview'
import { $activeGatewayProfile, $freshSessionRequest, $profileScope, refreshActiveProfile } from '@/store/profile'
import { $startWorkSessionRequest, followActiveSessionCwd, resolveNewSessionCwd } from '@/store/projects'
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
  setCurrentBranch,
  setCurrentCwd,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider,
  setMessages
} from '@/store/session'
import { focusOpenSession } from '@/store/session-states'
import { clearSessionTodos, setSessionTodos, todosForHydration } from '@/store/todos'
import { isSecondaryWindow } from '@/store/windows'
import { useSkinCommand } from '@/themes/use-skin-command'

import { requestComposerInsert } from '../chat/composer/focus'
import { useComposerActions } from '../chat/hooks/use-composer-actions'
import { CommandPalette } from '../command-palette'
import { useGatewayBoot } from '../gateway/hooks/use-gateway-boot'
import { useGatewayRequest } from '../gateway/hooks/use-gateway-request'
import { useKeybinds } from '../hooks/use-keybinds'
import { ModelPickerOverlay } from '../model-picker-overlay'
import { ModelVisibilityOverlay } from '../model-visibility-overlay'
import { PetGenerateOverlay } from '../pet-generate/pet-generate-overlay'
import { FileActionDialogs } from '../right-sidebar/file-actions'
import { RemoteFolderPicker } from '../right-sidebar/files/remote-picker'
import { PersistentTerminal } from '../right-sidebar/terminal/persistent'
import { CRON_ROUTE, routeSessionId, sessionRoute, SETTINGS_ROUTE, syncWorkspaceIsPage } from '../routes'
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
import { useOverlayRouting } from '../shell/hooks/use-overlay-routing'
import { useWindowControlsOverlayWidth } from '../shell/hooks/use-window-controls-overlay-width'
import { titlebarControlsPosition } from '../shell/titlebar'
import { TitlebarControls } from '../shell/titlebar-controls'
import { UpdatesOverlay } from '../updates-overlay'

import { ContribWiringContext } from './context'
import { useBackgroundSync } from './hooks/use-background-sync'
import { useDesktopIntegrations } from './hooks/use-desktop-integrations'
import { usePetBridge } from './hooks/use-pet-bridge'
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
  const messagingTranscriptSignatureRef = useRef(new Map<string, string>())
  // Stable identity for the whole callback surface (see WiringActions). Mutated
  // in place each render so memoized surfaces never re-render on churn.
  const actionsRef = useRef<WiringActions | null>(null)

  const gatewayState = useStore($gatewayState)
  const activeSessionId = useStore($activeSessionId)
  const currentCwd = useStore($currentCwd)
  const freshDraftReady = useStore($freshDraftReady)
  const resumeFailedSessionId = useStore($resumeFailedSessionId)
  const resumeExhaustedSessionId = useStore($resumeExhaustedSessionId)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const messagingSessions = useStore($messagingSessions)
  const profileScope = useStore($profileScope)

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

  // Mirror "the workspace is showing a full page" into its atom — the
  // workspace pane contribution re-registers headerVeto from it, so the main
  // zone's tab bar stands down on pages (and returns with the chat).
  useEffect(() => {
    syncWorkspaceIsPage(location.pathname)
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
    settingsOpen,
    starmapOpen,
    toggleCommandCenter
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

  const { connectionRef, gatewayRef, requestGateway } = useGatewayRequest()

  const {
    loadMoreMessagingForPlatform,
    loadMoreSessions,
    loadMoreSessionsForProfile,
    refreshCronJobs,
    refreshMessagingSessions,
    refreshSessions
  } = useSessionListActions({ profileScope })

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
          const latest = await getSessionMessages(storedSessionId, storedProfile)
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

  // Refresh the open messaging transcript (inbound platform turns arrive via
  // the background gateway, not the desktop websocket). Signature-gated so a
  // no-change poll doesn't churn the thread.
  const refreshActiveMessagingTranscript = useCallback(async () => {
    const storedSessionId = selectedStoredSessionIdRef.current
    const runtimeSessionId = activeSessionIdRef.current

    if (!storedSessionId || !runtimeSessionId || busyRef.current) {
      return
    }

    const stored = $messagingSessions.get().find(s => sessionMatchesStoredId(s, storedSessionId))

    if (!stored || !isMessagingSource(stored.source)) {
      return
    }

    try {
      const latest = await getSessionMessages(storedSessionId, stored.profile)
      const signatureKey = `${stored.profile ?? 'default'}:${storedSessionId}`
      const sig = sessionMessagesSignature(latest.messages)

      if (messagingTranscriptSignatureRef.current.get(signatureKey) === sig) {
        return
      }

      messagingTranscriptSignatureRef.current.set(signatureKey, sig)
      const messages = toChatMessages(latest.messages)

      updateSessionState(
        runtimeSessionId,
        state => ({ ...state, messages: preserveLocalAssistantErrors(messages, state.messages) }),
        storedSessionId
      )
    } catch {
      // Non-fatal: next poll or manual refresh can hydrate.
    }
  }, [activeSessionIdRef, busyRef, selectedStoredSessionIdRef, updateSessionState])

  const { handleGatewayEvent } = useMessageStream({
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
    activeSessionIdRef,
    baseHandleGatewayEvent: handleGatewayEvent,
    currentCwd,
    currentView,
    requestGateway,
    routedSessionId,
    selectedStoredSessionId
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
  const activeGatewayProfile = useStore($activeGatewayProfile)
  const lastGatewayProfileRef = useRef(activeGatewayProfile)

  useEffect(() => {
    if (activeGatewayProfile === lastGatewayProfileRef.current) {
      return
    }

    lastGatewayProfileRef.current = activeGatewayProfile
    // Force: the new profile has its own default, so reseed even if the
    // composer already shows the previous profile's model.
    void refreshCurrentModel(true)
    void refreshActiveProfile()
  }, [activeGatewayProfile, refreshCurrentModel])

  // New session anchored to a workspace (sidebar "+" on a project/worktree).
  // Seeds cwd + branch from the clicked workspace; an explicit worktree path
  // also drills the sidebar into that project so the new lane is visible.
  const startSessionInWorkspace = useCallback(
    (path: null | string) => {
      startFreshSessionDraft()

      // A worktree lane carries its own path; the trunk "+" can be path-less
      // (the main checkout is implicit), so fall back to the active project's
      // root instead of no-op'ing on null.
      const target = path?.trim() || resolveNewSessionCwd()

      if (!target) {
        return
      }

      setCurrentCwd(target)
      void requestGateway<{ branch?: string; cwd?: string }>('config.get', { key: 'project', cwd: target })
        .then(info => {
          const resolved = info.cwd || target

          setCurrentCwd(resolved)
          setCurrentBranch(info.branch || '')

          if (path?.trim()) {
            restoreWorktree(resolved)
            void followActiveSessionCwd(resolved)
          }
        })
        .catch(() => undefined)
    },
    [requestGateway, startFreshSessionDraft]
  )

  // Composer "branch off into a new worktree": open a fresh session anchored
  // to the just-created tree, then prefill the task that kicked it off.
  const startWorkSessionRequest = useStore($startWorkSessionRequest)
  const lastStartWorkTokenRef = useRef(startWorkSessionRequest?.token ?? 0)

  useEffect(() => {
    if (!startWorkSessionRequest || startWorkSessionRequest.token === lastStartWorkTokenRef.current) {
      return
    }

    lastStartWorkTokenRef.current = startWorkSessionRequest.token
    startSessionInWorkspace(startWorkSessionRequest.path)

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
      handleDesktopGatewayEvent(event)
    },
    [handleDesktopGatewayEvent]
  )

  useGatewayBoot({
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

  // Only the open messaging transcript needs its own poll — local chats are
  // live over the websocket already.
  const activeIsMessaging =
    !!selectedStoredSessionId &&
    isMessagingSource(messagingSessions.find(s => sessionMatchesStoredId(s, selectedStoredSessionId))?.source)

  // Keep app data live while the gateway is open (on-connect reseed + the
  // cron / messaging / transcript visibility polls + fresh-draft reseed).
  useBackgroundSync({
    activeIsMessaging,
    activeSessionId,
    freshDraftReady,
    gatewayState,
    refreshActiveMessagingTranscript,
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
  const filePreviewTarget = useStore($filePreviewTarget)

  useDesktopIntegrations({
    chatOpen,
    hasPreview: Boolean(filePreviewTarget || previewTarget),
    locationPathname: location.pathname,
    navigate,
    refreshSessions,
    resumeExhaustedSessionId,
    routedSessionId,
    runtimeIdByStoredSessionId: runtimeIdByStoredSessionIdRef
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

  // Single global listener for every rebindable hotkey plus the on-screen
  // keybind editor's capture mode (same as DesktopController).
  useKeybinds({
    openNewSessionTab: () => void openNewSessionTile('center'),
    startFreshSession: startFreshSessionDraft,
    toggleCommandCenter,
    toggleSelectedPin
  })

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
    onLoadMoreProfileSessions: loadMoreSessionsForProfile,
    onLoadMoreSessions: loadMoreSessions,
    onManageCronJob: jobId => {
      setCronFocusJobId(jobId)
      navigate(CRON_ROUTE)
    },
    onNavigate: selectSidebarItem,
    onNewSessionInWorkspace: startSessionInWorkspace,
    onNewSessionSplit: dir => void openNewSessionTile(dir),
    onPasteClipboardImage: opts => composer.pasteClipboardImage(opts),
    onPickFiles: () => void composer.pickContextPaths('file'),
    onPickFolders: () => void composer.pickContextPaths('folder'),
    onPickImages: () => void composer.pickImages(),
    onReload: reloadFromMessage,
    onRemoveAttachment: id => void composer.removeAttachment(id),
    onRestoreToMessage: restoreToMessage,
    // Already on screen (open tile, or the main session)? Jump to its tab;
    // otherwise load it into main.
    onResumeSession: sessionId => {
      if (!focusOpenSession(sessionId)) {
        navigate(sessionRoute(sessionId))
      }
    },
    onRetryResume: sessionId => void resumeSession(sessionId, true),
    onSteer: steerPrompt,
    onSubmit: submitText,
    onThreadMessagesChange: handleThreadMessagesChange,
    onToggleSelectedPin: toggleSelectedPin,
    onTranscribeAudio: transcribeVoiceAudio,
    onTriggerCronJob: jobId => {
      void triggerCronJob(jobId)
        .then(() => refreshCronJobs())
        .catch(() => undefined)
    },
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
  // Exact vertical centering: titlebarControlsPosition() returns
  // (TITLEBAR_HEIGHT - TITLEBAR_CONTROL_HEIGHT) / 2, but TitlebarControls
  // also applies a hard translate-y-0.5 (+2px) to its clusters. Cancel that
  // constant so cluster center == bar center — measured, not eyeballed.
  const controlsTranslateY = 2
  // Windows/WSLg reserve native min/max/close on the right (AppShell parity:
  // prefer the live WCO measurement, fall back to the static reservation).
  const measuredOverlayWidth = useWindowControlsOverlayWidth()
  const nativeOverlayWidth = measuredOverlayWidth ?? connection?.nativeOverlayWidth ?? 0
  const titlebarToolsRight = nativeOverlayWidth > 0 ? `${nativeOverlayWidth}px` : '0.75rem'
  // Pane-registered tools (preview's monitor/devtools cluster) anchor flush
  // against the static system cluster — in the tree layout the titlebar band
  // sits ABOVE the grid, so AppShell's pane-width anchoring doesn't apply.
  const SYSTEM_TOOL_COUNT = 4
  const paneToolCount = rightTitlebarTools.filter(tool => !tool.hidden).length
  const systemToolsWidth = `calc(${SYSTEM_TOOL_COUNT} * (var(--titlebar-control-size) + 0.25rem))`

  const titlebarToolsWidth =
    paneToolCount > 0
      ? `calc(${systemToolsWidth} + ${paneToolCount} * (var(--titlebar-control-size) + 0.25rem))`
      : systemToolsWidth

  return (
    <ContribWiringContext.Provider value={api}>
      <div
        className="contents"
        style={
          {
            '--titlebar-controls-left': `${controlsPos.left}px`,
            '--titlebar-controls-top': `${controlsPos.top - controlsTranslateY}px`,
            '--titlebar-tools-right': titlebarToolsRight,
            '--titlebar-tools-width': titlebarToolsWidth,
            '--shell-preview-toolbar-gap': systemToolsWidth
          } as CSSProperties
        }
      >
        <TitlebarControls
          leftTools={leftTitlebarTools}
          onOpenSettings={() => navigate(SETTINGS_ROUTE)}
          tools={rightTitlebarTools}
        />
        {children}
      </div>

      {/* The full real overlay set (mirrors DesktopController's `overlays`). */}
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
      <PetGenerateOverlay />
      <SessionSwitcher />
      <FileActionDialogs />
      <RemoteFolderPicker />

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
              setCurrentModelSource('default')
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

      {starmapOpen && (
        <Suspense fallback={null}>
          <StarmapView onClose={closeOverlayToPreviousRoute} />
        </Suspense>
      )}

      {/* Toasts above everything. */}
      <NotificationStack />

      {/* Petdex floating mascot — renders nothing unless installed + enabled. */}
      <FloatingPet />

      {/* Single persistent xterm host chasing the terminal pane's slot rect. */}
      <PersistentTerminal onAddSelectionToChat={composer.addTerminalSelectionAttachment} />
    </ContribWiringContext.Provider>
  )
}
