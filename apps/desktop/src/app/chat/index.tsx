import { type AppendMessage, AssistantRuntimeProvider, type ThreadMessage } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import type { ReadableAtom } from 'nanostores'
import type * as React from 'react'
import { memo, Suspense, useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import { useLocation } from 'react-router'

import type { SubmitTextOptions } from '@/app/session/hooks/use-prompt-actions/utils'
import { sessionShouldHaveTranscript } from '@/app/session/hooks/use-session-actions/utils'
import { Thread } from '@/components/assistant-ui/thread'
import { TranscriptWindowProvider } from '@/components/assistant-ui/thread/transcript-window'
import { Backdrop } from '@/components/Backdrop'
import { COMPOSER_HEART_CONFIG, HeartField } from '@/components/chat/vibe-hearts'
import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { $sessionTileDragging, $sessionTileEdgeHover } from '@/components/pane-shell/tree/store'
import { PromptOverlays } from '@/components/prompt-overlays'
import { Button } from '@/components/ui/button'
import { ErrorState } from '@/components/ui/error-state'
import { TitleMenuTrigger } from '@/components/ui/title-menu-trigger'
import { type HermesGateway } from '@/hermes'
import { useI18n } from '@/i18n'
import type { ChatMessage } from '@/lib/chat-messages'
import { NEW_SESSION_TITLE, quickModelOptions, sessionTitle } from '@/lib/chat-runtime'
import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'
import { modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'
import { useStoreSelector } from '@/lib/use-session-slice'
import { cn } from '@/lib/utils'
import { migrateSessionDraft } from '@/store/composer'
import { migrateQueuedPrompts, parkQueuedPrompts } from '@/store/composer-queue'
import { $introSplash } from '@/store/intro-splash'
import { $pinnedSessionIds } from '@/store/layout'
import { $petActive } from '@/store/pet'
import { $petOverlayActive } from '@/store/pet-overlay'
import { $activeGatewayProfile, $gatewaySwapTarget, $hydrationSyncProfile, $profiles } from '@/store/profile'
import {
  $connection,
  $contextSuggestions,
  $freshDraftReady,
  $gatewayState,
  $introPersonality,
  $introSeed,
  $resumeExhaustedSessionId,
  $sessions,
  getSessionOwnerHint,
  resolveComposerSessionKey,
  sessionMatchesStoredId,
  sessionPinId,
  shouldMigrateComposerScope
} from '@/store/session'
import { $focusedStoredSessionId, sessionTileDelegate } from '@/store/session-states'
import { $transcriptTailBySessionId, transcriptTailState } from '@/store/transcript-tail'
import { isAuxiliaryWindow, isWatchWindow } from '@/store/windows'
import type { ModelOptionsResponse } from '@/types/hermes'

import { primaryRouteSelectedSessionId, routeSessionId } from '../routes'
import { titlebarHeaderBaseClass, titlebarHeaderShadowClass, titlebarHeaderTitleClass } from '../shell/titlebar'

import { ChatDropOverlay } from './chat-drop-overlay'
import { ChatSwapOverlay, ChatSyncBadge } from './chat-swap-overlay'
import { ChatBar, ChatBarFallback } from './composer'
import { requestComposerInsert } from './composer/focus'
import { droppedFileInlineRefs } from './composer/inline-refs'
import { ComposerSurfaceProvider, useComposerScope, useComposerSurfaceId } from './composer/scope'
import type { ChatBarState } from './composer/types'
import { type DroppedFile, partitionDroppedFiles } from './hooks/use-composer-actions'
import { type DragKind, useFileDropZone } from './hooks/use-file-drop-zone'
import { shouldShowIntro } from './intro-visibility'
import { ProfileTag } from './profile-tag'
import { isRouteSessionMismatch } from './route-session-state'
import { useRuntimeMessageRepository } from './runtime-repository'
import { ScrollToBottomButton } from './scroll-to-bottom-button'
import { useSessionView } from './session-view'
import { SessionActionsMenu } from './sidebar/session-actions-menu'
import { routedSessionIsLoading, threadLoadingState } from './thread-loading'
import {
  backfillOlderTranscriptPage,
  mergeOlderTranscriptPage,
  transcriptBackfillAvailable
} from './transcript-backfill'
import { advanceTranscriptWindow, type TranscriptWindowState } from './transcript-window'

interface ChatViewProps extends Omit<React.ComponentProps<'div'>, 'onSubmit'> {
  gateway: HermesGateway | null
  modelOptionsOwnerConnectionId?: string
  modelOptionsProfile?: string
  modelMenuContent?: React.ReactNode
  requestModelOptionsForOwner?: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  onToggleSelectedPin: () => void
  onDeleteSelectedSession: () => void
  onCancel: () => Promise<void> | void
  onAddContextRef: (refText: string, label?: string, detail?: string) => void
  onAddUrl: (url: string) => void
  onBranchInNewChat?: (messageId: string) => void
  maxVoiceRecordingSeconds?: number
  onAttachImageBlob: (blob: Blob) => Promise<boolean | void> | boolean | void
  onAttachDroppedItems: (candidates: DroppedFile[]) => Promise<boolean | void> | boolean | void
  onAttachPrCommentUrl?: (url: string) => boolean
  onPasteClipboardImage: (opts?: { silent?: boolean }) => Promise<boolean> | void
  onPickFiles: () => void
  onPickFolders: () => void
  onPickImages: () => void
  onRemoveAttachment: (id: string) => void
  onSteer: (text: string) => Promise<boolean> | boolean
  onSubmit: (text: string, options?: SubmitTextOptions) => Promise<boolean> | boolean
  onThreadMessagesChange: (messages: readonly ThreadMessage[]) => void
  onEdit: (message: AppendMessage) => Promise<void>
  onReload: (parentId: string | null) => Promise<void>
  onRestoreToMessage?: (messageId: string, target?: { text?: string; userOrdinal?: number | null }) => Promise<void>
  onRetryResume: (sessionId: string) => void
  onTranscribeAudio?: (audio: Blob) => Promise<string>
  onDismissError?: (messageId: string) => void
}

interface ChatHeaderProps {
  activeSessionId: null | string
  isRoutedSessionView: boolean
  onDeleteSelectedSession: () => void
  onToggleSelectedPin: () => void
  selectedSessionId: null | string
}

function ChatHeader({
  activeSessionId,
  isRoutedSessionView,
  onDeleteSelectedSession,
  onToggleSelectedPin,
  selectedSessionId
}: ChatHeaderProps) {
  const sessions = useStore($sessions)
  const pinnedSessionIds = useStore($pinnedSessionIds)
  const profiles = useStore($profiles)

  const activeStoredSession =
    (selectedSessionId && sessions.find(session => sessionMatchesStoredId(session, selectedSessionId))) || null

  const title = activeStoredSession ? sessionTitle(activeStoredSession) : NEW_SESSION_TITLE

  // Which agent/persona owns this chat — glanceable in the header once a
  // second profile exists, so the open session's ownership is never ambiguous
  // (#66003). Single-profile users see the unchanged header.
  const showProfileTag = profiles.length > 1 && Boolean(activeStoredSession)

  // Pins live on the durable lineage-root id, but selectedSessionId is the live
  // (tip) id — resolve through the loaded row so the menu reflects the pin
  // state after auto-compression rotates the id.
  const selectedIsPinned = activeStoredSession
    ? pinnedSessionIds.includes(sessionPinId(activeStoredSession))
    : selectedSessionId
      ? pinnedSessionIds.includes(selectedSessionId)
      : false

  // Secondary windows (new-session scratch, subagent watch, cmd-click pop-out)
  // are compact side panels — they drop the session-actions header + border
  // entirely. A brand-new draft has nothing to pin/delete/rename either.
  if (isAuxiliaryWindow() || (!selectedSessionId && !activeSessionId && !isRoutedSessionView)) {
    return null
  }

  return (
    <header className={cn(titlebarHeaderBaseClass, isRoutedSessionView && titlebarHeaderShadowClass)}>
      <div
        className={cn(titlebarHeaderTitleClass, showProfileTag && 'flex items-center')}
        style={{
          maxWidth:
            'calc(100vw - var(--titlebar-content-inset,0px) - var(--titlebar-tools-right) - var(--titlebar-tools-width) - 1.5rem)'
        }}
      >
        {showProfileTag && <ProfileTag className="pointer-events-auto mr-1.5" profile={activeStoredSession?.profile} />}
        <SessionActionsMenu
          align="start"
          onDelete={selectedSessionId ? onDeleteSelectedSession : undefined}
          onPin={selectedSessionId ? onToggleSelectedPin : undefined}
          pinned={selectedIsPinned}
          sessionId={selectedSessionId || activeSessionId || ''}
          sideOffset={8}
          title={title}
        >
          <TitleMenuTrigger>{title}</TitleMenuTrigger>
        </SessionActionsMenu>
      </div>
    </header>
  )
}

interface ChatRuntimeBoundaryProps {
  busy: boolean
  children: React.ReactNode
  onCancel: () => Promise<void> | void
  onEdit: (message: AppendMessage) => Promise<void>
  onReload: (parentId: string | null) => Promise<void>
  onThreadMessagesChange: (messages: readonly ThreadMessage[]) => void
  /** Route points at an unloaded session — render empty until resume swaps in
   *  the new transcript, so the previous session's messages don't linger. */
  suppressMessages: boolean
}

const NO_MESSAGES: ChatMessage[] = []

/**
 * The view's $messages, live only while this surface is the VISIBLE tab.
 *
 * Keep-alive keeps every ever-active tab MOUNTED (tree-group.tsx), so without
 * this gate a hidden tab re-renders its entire thread on every streaming
 * delta flush (~30×/s) — five busy tabs quintuple the per-token render cost
 * and the app crawls. Hidden tabs freeze their transcript instead (status
 * dots stay live through the separate status atoms) and catch up in one
 * commit on reveal — the subscribe fires immediately with the current value.
 */
function useMessagesWhileVisible($messages: ReadableAtom<ChatMessage[]>): ChatMessage[] {
  const visible = usePaneVisible()
  const [messages, setMessages] = useState(() => $messages.get())

  // nanostores types the listener value ReadonlyIfObject; the store publishes
  // a fresh array per flush, so the cast is safe and avoids a per-token clone.
  useEffect(
    () => (visible ? $messages.subscribe(value => setMessages(value as ChatMessage[])) : undefined),
    [$messages, visible]
  )

  return messages
}

/**
 * Owns the $messages subscription and the assistant-ui external-store runtime.
 *
 * Isolated from ChatView so the per-token delta flush (which replaces the
 * $messages atom ~30×/s during streaming) only re-renders this component and
 * the runtime provider. The children (Thread, ChatBar) are created by
 * ChatView, whose render output is stable across flushes — so React bails out
 * of re-rendering them by element identity and the stream's render cost stays
 * confined to the streaming message's own subtree.
 */
function ChatRuntimeBoundary({
  busy,
  children,
  onCancel,
  onEdit,
  onReload,
  onThreadMessagesChange,
  suppressMessages
}: ChatRuntimeBoundaryProps) {
  const view = useSessionView()
  const runtimeId = useStore(view.$runtimeId)
  const storeMessages = useMessagesWhileVisible(view.$messages)
  const messages = suppressMessages ? NO_MESSAGES : storeMessages

  const [windowPages, setWindowPages] = useState(1)
  const [windowSessionKey, setWindowSessionKey] = useState(runtimeId)
  // Sticky-cut continuity across flushes (advanceTranscriptWindow). A ref, not
  // state: it is derived from `messages` and must never trigger a render.
  const windowStateRef = useRef<null | TranscriptWindowState>(null)

  // Reset the window on session swap during RENDER, so a large expand from the
  // previous chat can't leak into the next one's first paint (#55191).
  if (windowSessionKey !== runtimeId) {
    setWindowSessionKey(runtimeId)
    setWindowPages(1)
    windowStateRef.current = null
  }

  const { messages: windowedMessages, windowed } = useMemo(() => {
    const next = advanceTranscriptWindow(windowStateRef.current, messages, windowPages)

    windowStateRef.current = next

    return next.window
  }, [messages, windowPages])

  const runtimeMessageRepository = useRuntimeMessageRepository(windowedMessages)

  const storedId = useStore(view.$storedId)
  const connection = useStore($connection)
  const activeProfile = useStore($activeGatewayProfile)
  // Subscribed (not read imperatively) so the "Show earlier" affordance
  // appears/retires as tail hydrations and backfill pages record their state.
  const transcriptTailStates = useStore($transcriptTailBySessionId)
  const connectionId = connection?.connectionId || (connection?.mode === 'local' ? 'local' : '')

  const ownerRoute = storedId
    ? getSessionOwnerHint(storedId, connectionId ? { connectionId, profile: activeProfile } : undefined)
    : undefined

  const tailProfile = ownerRoute
    ? { connectionId: ownerRoute.connectionId, profile: ownerRoute.targetProfile || ownerRoute.profile }
    : undefined

  const tailState = storedId && transcriptTailStates ? transcriptTailState(storedId, tailProfile) : undefined
  const restBackfillAvailable = Boolean(tailState?.possiblyTruncated)

  const expandWindow = useCallback(() => {
    // The store window still holds older messages: growing pages is enough.
    // Otherwise the whole in-memory transcript is already materialized — if
    // the REST tail hydration was truncated, fetch the next older page and
    // PREPEND it to the session store before growing, so the grown window has
    // something older to show. Fire-and-forget: the prepend lands through the
    // session-state write path and re-renders this boundary.
    if (
      !windowStateRef.current?.window.windowed &&
      runtimeId &&
      storedId &&
      transcriptBackfillAvailable(storedId, tailProfile)
    ) {
      void backfillOlderTranscriptPage({
        storedSessionId: storedId,
        profile: tailProfile,
        // Stale-response guard: a session switch remounts/re-keys this view;
        // checking the live atoms (not captured props) discards a page that
        // resolves after the user moved on — same pattern as isCurrentResume.
        isCurrent: () => view.$storedId.get() === storedId && view.$runtimeId.get() === runtimeId,
        applyOlderPage: olderPage => {
          sessionTileDelegate()?.updateSession(runtimeId, state => {
            const merged = mergeOlderTranscriptPage(state.messages, olderPage)

            return merged === state.messages ? state : { ...state, messages: merged }
          })
        }
      })
    }

    setWindowPages(pages => pages + 1)
  }, [runtimeId, storedId, tailProfile, view])

  const olderAvailable = windowed || restBackfillAvailable

  const transcriptWindow = useMemo(() => ({ olderAvailable, expandWindow }), [expandWindow, olderAvailable])

  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    messageRepository: runtimeMessageRepository,
    isRunning: busy,
    setMessages: onThreadMessagesChange,
    onNew: async () => {
      // Submission is handled explicitly by ChatBar.
      // Keeping this no-op avoids duplicate prompt.submit calls.
    },
    onEdit,
    onCancel: async () => onCancel(),
    onReload
  })

  return (
    <TranscriptWindowProvider value={transcriptWindow}>
      <AssistantRuntimeProvider runtime={runtime}>{children}</AssistantRuntimeProvider>
    </TranscriptWindowProvider>
  )
}

// Memoized: the tile caller (session-tile.tsx) and the contrib surface re-render
// on idle ticks unrelated to the chat; with stable callback props (hoisted to
// useCallback at the call sites) memo() lets the whole chat shell skip those.
export const ChatView = memo(function ChatView(props: ChatViewProps) {
  const composerSurfaceId = useId()

  return (
    <ComposerSurfaceProvider value={composerSurfaceId}>
      <ChatViewContent {...props} />
    </ComposerSurfaceProvider>
  )
})

const ChatViewContent = memo(function ChatViewContent({
  className,
  gateway,
  modelOptionsOwnerConnectionId,
  modelOptionsProfile,
  modelMenuContent,
  requestModelOptionsForOwner,
  onToggleSelectedPin,
  onDeleteSelectedSession,
  onCancel,
  onAddContextRef,
  onAddUrl,
  onAttachImageBlob,
  onAttachDroppedItems,
  onAttachPrCommentUrl,
  onBranchInNewChat,
  maxVoiceRecordingSeconds,
  onPasteClipboardImage,
  onPickFiles,
  onPickFolders,
  onPickImages,
  onRemoveAttachment,
  onSteer,
  onSubmit,
  onThreadMessagesChange,
  onEdit,
  onReload,
  onRestoreToMessage,
  onRetryResume,
  onTranscribeAudio,
  onDismissError
}: ChatViewProps) {
  const location = useLocation()
  const { t } = useI18n()
  // The view this surface renders: the primary route-driven session (global
  // atoms) or a tile's session slice — same component either way.
  const view = useSessionView()
  const composerScope = useComposerScope()
  const composerSurfaceId = useComposerSurfaceId()
  const isPrimary = view.kind === 'primary'
  const activeSessionId = useStore(view.$runtimeId)
  const storedId = useStore(view.$storedId)
  // Multi-pane dimming: only the focused surface paints at full strength, so
  // two sessions side by side read as "this one, and that one over there".
  // A selector, not a plain useStore — the focused id changes on click, and a
  // boolean bails every other surface out of the re-render. Sole surface ⇒
  // always focused (the atom falls back to the primary's selection), so a
  // single-pane workspace never dims.
  const surfaceFocused = useStoreSelector($focusedStoredSessionId, focused => focused === storedId)
  // Dock anchor for a session drop onto this surface: the workspace pane for the
  // primary, this tile's pane id for a tile. Read by the session-drop bridge.
  const sessionAnchor = isPrimary ? 'workspace' : `session-tile:${storedId ?? ''}`
  const awaitingResponse = useStore(view.$awaitingResponse)
  const busy = useStore(view.$busy)
  const activeGatewayProfile = useStore($activeGatewayProfile)
  const contextSuggestions = useStore($contextSuggestions)
  // Per-session (SessionView) reads — a tile IS its session, so these come
  // from the view slice, not the global atoms (which track the primary only).
  const currentCwd = useStore(view.$cwd)
  const currentModel = useStore(view.$model)
  const currentProvider = useStore(view.$provider)
  // A pet anywhere (in-window or popped out) owns the hearts; composer only when none.
  const petActive = useStore($petActive)
  const petOverlayActive = useStore($petOverlayActive)
  const petPresent = petActive || petOverlayActive
  const freshDraftReady = useStore($freshDraftReady)
  const gatewayState = useStore($gatewayState)
  const gatewaySwapTarget = useStore($gatewaySwapTarget)
  const hydrationSyncProfile = useStore($hydrationSyncProfile)
  const gatewayOpen = gatewayState === 'open'
  const introPersonality = useStore($introPersonality)
  const introSeed = useStore($introSeed)
  const introSplash = useStore($introSplash)
  // PERF: ChatView must not subscribe to the view's $messages — the atom is
  // replaced on every streaming delta flush (~30×/s) and a subscription here
  // re-renders the entire chat shell (header, chat bar, thread wrapper) per
  // token. The runtime that DOES need the messages lives in
  // ChatRuntimeBoundary below; this component only needs streaming-stable
  // derivations.
  const messagesEmpty = useStore(view.$messagesEmpty)
  const lastVisibleIsUser = useStore(view.$lastVisibleIsUser)
  const selectedSessionId = useStore(view.$storedId)
  const sessions = useStore($sessions)
  const resumeExhaustedSessionId = useStore($resumeExhaustedSessionId)

  // Durable composer/queue scope (lineage root) so auto-compression tip rotation
  // does not wipe an in-progress draft or orphan /queue entries. For the
  // primary view, the route is authoritative over the store selection — the
  // latter can be momentarily null/stale mid-switch, which used to leak into
  // the composer's scope key (#59305). A tile has no route, so it always uses
  // its own selection directly.
  const queueSessionKey = useMemo(() => {
    const effectiveSelectedSessionId = isPrimary
      ? primaryRouteSelectedSessionId(location.pathname, selectedSessionId)
      : selectedSessionId

    return resolveComposerSessionKey(effectiveSelectedSessionId, sessions)
  }, [isPrimary, location.pathname, selectedSessionId, sessions])

  // When the tip row arrives after compression, migrate any tip-keyed stash onto
  // the durable lineage key before the composer remounts onto that key.
  //
  // ONLY same-conversation rekeys (tip → root). The route-driven queueSessionKey
  // can flip to Session B a frame before the store selection leaves Session A;
  // migrating on bare inequality would re-home A's queued prompts onto B and
  // auto-drain them into the wrong chat.
  useEffect(() => {
    if (!shouldMigrateComposerScope(selectedSessionId, queueSessionKey, sessions)) {
      return
    }

    migrateSessionDraft(selectedSessionId, queueSessionKey)
    migrateQueuedPrompts(selectedSessionId, queueSessionKey)
  }, [queueSessionKey, selectedSessionId, sessions])

  // Transcript-side stops (the streaming message's hover Stop, the runtime's
  // cancel) are explicit halts, same as the composer's Stop button: park any
  // queued turns so the interrupt doesn't roll straight into the next one.
  // ChatBar wraps its own onCancel internally — its send-now-while-busy path
  // needs the raw interrupt — so it still receives the unwrapped prop.
  const haltRun = useCallback(() => {
    parkQueuedPrompts(queueSessionKey || activeSessionId)

    return onCancel()
  }, [activeSessionId, onCancel, queueSessionKey])

  // A tile IS its session — no route involved, never "mismatched".
  const routedSessionId = isPrimary ? routeSessionId(location.pathname) : selectedSessionId
  const isRoutedSessionView = Boolean(routedSessionId)

  // The URL points at a session the store hasn't loaded yet (sidebar / cmd-K /
  // direct nav). Derived in render so the swap reads instantly: the same frame
  // the id changes we drop the old transcript and show the loader, instead of
  // waiting for the resume effect (which paints a frame later) to clear them.
  const routeSessionMismatch = isPrimary ? isRouteSessionMismatch(routedSessionId, selectedSessionId, sessions) : false

  // The compact new-session pop-out skips the wordmark/tagline intro — it's a
  // scratch window, not the full-height empty state. The Appearance toggle
  // turns it off everywhere else.
  const showIntro = shouldShowIntro({
    activeSessionId,
    auxiliaryWindow: isAuxiliaryWindow(),
    enabled: introSplash,
    freshDraftReady,
    messagesEmpty,
    primary: isPrimary,
    routedSessionView: isRoutedSessionView,
    selectedSessionId
  })

  // Session is still loading if the route references a session we haven't
  // resumed yet. Brand-new routed drafts are empty on purpose once a runtime
  // is bound. A session the list already knows has history must keep the
  // loader up until a display-authoritative transcript arrives — including
  // the unproven warm-cache hold, where the runtime is bound but messages
  // are still suppressed.
  //
  // resumeExhausted: the bounded auto-retry in use-route-resume gave up on this
  // routed session (gateway RPC + REST fallback failed through every attempt).
  // Suppress the loader and show an explicit error + manual Retry instead of
  // spinning forever. Gated on the route matching so a stale latch from another
  // session can't blank the current one.
  const resumeExhausted = isPrimary && isRoutedSessionView && resumeExhaustedSessionId === routedSessionId

  const routedHasHistory = Boolean(
    routedSessionId &&
    sessions.some(session => sessionMatchesStoredId(session, routedSessionId) && sessionShouldHaveTranscript(session))
  )

  const loadingSession = routedSessionIsLoading({
    activeSessionId,
    knownHistory: routedHasHistory,
    messagesEmpty,
    resumeExhausted,
    routeSessionMismatch,
    routedSessionView: isRoutedSessionView
  })

  const threadLoading = threadLoadingState(loadingSession, busy, awaitingResponse, lastVisibleIsUser)
  // Hide the composer in the exhausted error state too: there's no live runtime
  // to send to until a retry rebinds one. Watch windows are pure spectators of a
  // subagent run driven elsewhere — no composer, transcript is read-only.
  const showChatBar = !loadingSession && !resumeExhausted && !isWatchWindow()
  const threadKey = selectedSessionId || activeSessionId || (isRoutedSessionView ? location.pathname : 'new')

  const modelOptionsQuery = useQuery<ModelOptionsResponse>({
    queryKey: modelOptionsQueryKey(
      modelOptionsProfile || activeGatewayProfile,
      activeSessionId,
      modelOptionsOwnerConnectionId
    ),
    queryFn: () =>
      requestModelOptions({
        gateway: gateway || undefined,
        profile: modelOptionsProfile || activeGatewayProfile,
        request: requestModelOptionsForOwner,
        sessionId: activeSessionId
      }),
    enabled: gatewayOpen
  })

  const quickModels = useMemo(
    () => quickModelOptions(modelOptionsQuery.data, currentProvider, currentModel),
    [currentModel, currentProvider, modelOptionsQuery.data]
  )

  const chatBarState = useMemo<ChatBarState>(
    () => ({
      model: {
        model: currentModel,
        provider: currentProvider,
        canSwitch: gatewayOpen,
        loading: !gatewayOpen || (!currentModel && !currentProvider),
        modelMenuContent,
        quickModels
      },
      tools: {
        enabled: true,
        label: 'Add context',
        suggestions: contextSuggestions
      },
      voice: {
        enabled: true,
        active: false
      }
    }),
    [contextSuggestions, currentModel, currentProvider, gatewayOpen, modelMenuContent, quickModels]
  )

  // Drop files anywhere in the conversation area, not just on the composer
  // input. In-app drags (project tree / gutter) carry workspace-relative paths
  // the gateway resolves directly, so they stay inline `@file:` refs. OS/Finder
  // drops carry absolute local paths that don't exist on a remote gateway (and
  // images need byte upload for vision), so route them through the attachment
  // pipeline — otherwise the local path leaks into the prompt verbatim.
  const onDropFiles = useCallback(
    (candidates: DroppedFile[]) => {
      const { inAppRefs, osDrops } = partitionDroppedFiles(candidates)
      const refs = droppedFileInlineRefs(inAppRefs, currentCwd)

      if (refs.length) {
        requestComposerInsert(refs.join(' '), { mode: 'inline', target: composerScope.target })
      }

      if (osDrops.length) {
        void onAttachDroppedItems(osDrops)
      }
    },
    [composerScope.target, currentCwd, onAttachDroppedItems]
  )

  // Session drags are POINTER drags (session-drag.ts) — never native DnD.
  // The drop zone below only handles files; session drops commit through the
  // drag session itself, which routes a center/link drop to this surface's
  // composer via `data-composer-target`.
  const { dragKind, dropHandlers } = useFileDropZone({ enabled: showChatBar, onDropFiles })

  // While a session drag targets one of this surface's EDGES or a tab strip,
  // the zone overlay/caret owns the visual — the link overlay stands down.
  // It shows for the whole drag on every chat surface otherwise (the drag
  // session's global sentinel, not a per-surface hover chain).
  // COMPUTED booleans, never the raw `$dropHint`: the hint churns on every
  // pointer-crossing of every drag (pane drags included), and a re-render
  // here is the WHOLE surface — thread, composer, header — per mounted tile.
  const sessionDragging = useStore($sessionTileDragging)
  const sessionEdgeHover = useStore($sessionTileEdgeHover)

  const overlayKind: DragKind = dragKind === 'files' ? 'files' : sessionDragging && !sessionEdgeHover ? 'session' : null

  return (
    <div
      className={cn(
        'relative isolate flex h-full min-w-0 flex-col overflow-hidden bg-(--ui-chat-surface-background)',
        className
      )}
      data-chat-surface=""
      data-chat-unfocused={surfaceFocused ? undefined : ''}
      data-composer-surface-id={composerSurfaceId}
      data-composer-target={composerScope.target}
      data-session-anchor={sessionAnchor}
    >
      <Backdrop />
      {/* Tiles get their chrome from the layout zone (chip strip); the modal
          prompt overlays stay active-session-scoped in the primary surface. */}
      {isPrimary && (
        <ChatHeader
          activeSessionId={activeSessionId}
          isRoutedSessionView={isRoutedSessionView}
          onDeleteSelectedSession={onDeleteSelectedSession}
          onToggleSelectedPin={onToggleSelectedPin}
          selectedSessionId={selectedSessionId}
        />
      )}

      {/* Mounted for the primary AND every tile, each scoped to its own session
          so a tiled/background session's blocking prompt surfaces instead of
          stalling to timeout. */}
      <PromptOverlays sessionId={activeSessionId} />

      <ChatRuntimeBoundary
        busy={busy}
        onCancel={haltRun}
        onEdit={onEdit}
        onReload={onReload}
        onThreadMessagesChange={onThreadMessagesChange}
        suppressMessages={routeSessionMismatch}
      >
        <div
          className="relative min-h-0 max-w-full flex-1 overflow-hidden bg-(--ui-chat-surface-background) contain-[layout_paint]"
          data-slot="composer-bounds"
          {...dropHandlers}
        >
          <Thread
            clampToComposer={showChatBar}
            cwd={currentCwd}
            gateway={gateway}
            intro={showIntro ? { personality: introPersonality, seed: introSeed } : undefined}
            loading={threadLoading}
            onBranchInNewChat={onBranchInNewChat}
            onCancel={haltRun}
            onDismissError={onDismissError}
            onRestoreToMessage={onRestoreToMessage}
            sessionId={activeSessionId}
            sessionKey={threadKey}
          />
          {resumeExhausted && routedSessionId && (
            <div className="absolute inset-0 z-10 grid place-items-center bg-(--ui-chat-surface-background) px-8 py-10">
              <ErrorState
                className="max-w-sm"
                description={t.desktop.resumeStrandedBody}
                title={t.desktop.resumeStrandedTitle}
              >
                <div className="grid justify-items-center">
                  <Button onClick={() => onRetryResume(routedSessionId)} size="sm" variant="outline">
                    {t.desktop.resumeRetry}
                  </Button>
                </div>
              </ErrorState>
            </div>
          )}
          {showChatBar && <ScrollToBottomButton />}
          {/* Vibe hearts rise from the composer only when no pet is out (else
              they play on the pet). Fired by the core `reaction` event. */}
          {!petPresent && (
            <HeartField
              className="absolute inset-x-0 z-30"
              config={COMPOSER_HEART_CONFIG}
              style={{
                top: 0,
                bottom: 'calc(var(--composer-measured-height) + 0.25rem)'
              }}
            />
          )}
          {/* A session drag hovering an EDGE hands the visual to the zone
              target; the link overlay shows only for the center region. */}
          <ChatDropOverlay kind={overlayKind} />
          <ChatSwapOverlay profile={gatewaySwapTarget} />
          {/* Paint-first wake (#89843): transcript is live, profile gate still
              settling in the background — subtle badge, not an overlay. */}
          {isPrimary && !gatewaySwapTarget && <ChatSyncBadge profile={hydrationSyncProfile} />}
        </div>
        {/* Composer renders OUTSIDE the contain:[layout paint] wrapper above:
            that wrapper is a containing block for — and clips — position:fixed
            descendants, so the popped-out (fixed) composer would anchor to the
            chat column (which shifts/resizes with the sidebars) and get clipped
            off-screen instead of floating against the viewport. As a sibling it
            anchors to the outer relative container instead: docked is absolute
            (identical placement), floating resolves against the viewport. Both
            states stay mounted here, so dock⇄float never remounts the editor. */}
        {showChatBar && (
          <Suspense fallback={<ChatBarFallback />}>
            <ChatBar
              busy={busy}
              cwd={currentCwd}
              disabled={!gatewayOpen}
              focusKey={activeSessionId}
              gateway={gateway}
              maxRecordingSeconds={maxVoiceRecordingSeconds}
              onAddContextRef={onAddContextRef}
              onAddUrl={onAddUrl}
              onAttachDroppedItems={onAttachDroppedItems}
              onAttachImageBlob={onAttachImageBlob}
              onAttachPrCommentUrl={onAttachPrCommentUrl}
              onCancel={onCancel}
              onPasteClipboardImage={onPasteClipboardImage}
              onPickFiles={onPickFiles}
              onPickFolders={onPickFolders}
              onPickImages={onPickImages}
              onRemoveAttachment={onRemoveAttachment}
              onSteer={onSteer}
              onSubmit={onSubmit}
              onTranscribeAudio={onTranscribeAudio}
              queueSessionKey={queueSessionKey}
              sessionId={activeSessionId}
              state={chatBarState}
            />
          </Suspense>
        )}
      </ChatRuntimeBoundary>
    </div>
  )
})
