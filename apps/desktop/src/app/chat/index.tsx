import { type AppendMessage, AssistantRuntimeProvider, type ThreadMessage } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import type * as React from 'react'
import { Suspense, useCallback, useEffect, useMemo } from 'react'
import { useLocation } from 'react-router-dom'

import type { SubmitTextOptions } from '@/app/session/hooks/use-prompt-actions/utils'
import { Thread } from '@/components/assistant-ui/thread'
import { Backdrop } from '@/components/Backdrop'
import { COMPOSER_HEART_CONFIG, HeartField } from '@/components/chat/vibe-hearts'
import { $sessionTileDragging, $sessionTileEdgeHover } from '@/components/pane-shell/tree/store'
import { PromptOverlays } from '@/components/prompt-overlays'
import { Button } from '@/components/ui/button'
import { ErrorState } from '@/components/ui/error-state'
import { TitleMenuTrigger } from '@/components/ui/title-menu-trigger'
import { getGlobalModelOptions, type HermesGateway } from '@/hermes'
import { useI18n } from '@/i18n'
import type { ChatMessage } from '@/lib/chat-messages'
import { quickModelOptions, sessionTitle } from '@/lib/chat-runtime'
import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'
import { cn } from '@/lib/utils'
import { migrateSessionDraft } from '@/store/composer'
import { migrateQueuedPrompts } from '@/store/composer-queue'
import { $pinnedSessionIds } from '@/store/layout'
import { $petActive } from '@/store/pet'
import { $petOverlayActive } from '@/store/pet-overlay'
import { $gatewaySwapTarget, $profiles } from '@/store/profile'
import {
  $contextSuggestions,
  $freshDraftReady,
  $gatewayState,
  $introPersonality,
  $introSeed,
  $resumeExhaustedSessionId,
  $sessions,
  resolveComposerSessionKey,
  sessionMatchesStoredId,
  sessionPinId
} from '@/store/session'
import { isSecondaryWindow, isWatchWindow } from '@/store/windows'
import type { ModelOptionsResponse } from '@/types/hermes'

import { routeSessionId } from '../routes'
import { titlebarHeaderBaseClass, titlebarHeaderShadowClass, titlebarHeaderTitleClass } from '../shell/titlebar'

import { ChatDropOverlay } from './chat-drop-overlay'
import { ChatSwapOverlay } from './chat-swap-overlay'
import { ChatBar, ChatBarFallback } from './composer'
import { requestComposerInsert } from './composer/focus'
import { droppedFileInlineRefs } from './composer/inline-refs'
import { useComposerScope } from './composer/scope'
import type { ChatBarState } from './composer/types'
import { type DroppedFile, partitionDroppedFiles } from './hooks/use-composer-actions'
import { type DragKind, useFileDropZone } from './hooks/use-file-drop-zone'
import { ProfileTag } from './profile-tag'
import { useRuntimeMessageRepository } from './runtime-repository'
import { ScrollToBottomButton } from './scroll-to-bottom-button'
import { useSessionView } from './session-view'
import { SessionActionsMenu } from './sidebar/session-actions-menu'
import { threadLoadingState } from './thread-loading'

interface ChatViewProps extends Omit<React.ComponentProps<'div'>, 'onSubmit'> {
  gateway: HermesGateway | null
  modelMenuContent?: React.ReactNode
  onToggleSelectedPin: () => void
  onDeleteSelectedSession: () => void
  onCancel: () => Promise<void> | void
  onAddContextRef: (refText: string, label?: string, detail?: string) => void
  onAddUrl: (url: string) => void
  onBranchInNewChat: (messageId: string) => void
  maxVoiceRecordingSeconds?: number
  onAttachImageBlob: (blob: Blob) => Promise<boolean | void> | boolean | void
  onAttachDroppedItems: (candidates: DroppedFile[]) => Promise<boolean | void> | boolean | void
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

  const title = activeStoredSession ? sessionTitle(activeStoredSession) : 'New session'

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
  if (isSecondaryWindow() || (!selectedSessionId && !activeSessionId && !isRoutedSessionView)) {
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
  const storeMessages = useStore(useSessionView().$messages)
  const messages = suppressMessages ? NO_MESSAGES : storeMessages
  const runtimeMessageRepository = useRuntimeMessageRepository(messages)

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

  return <AssistantRuntimeProvider runtime={runtime}>{children}</AssistantRuntimeProvider>
}

export function ChatView({
  className,
  gateway,
  modelMenuContent,
  onToggleSelectedPin,
  onDeleteSelectedSession,
  onCancel,
  onAddContextRef,
  onAddUrl,
  onAttachImageBlob,
  onAttachDroppedItems,
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
  const isPrimary = view.kind === 'primary'
  const activeSessionId = useStore(view.$runtimeId)
  const storedId = useStore(view.$storedId)
  // Dock anchor for a session drop onto this surface: the workspace pane for the
  // primary, this tile's pane id for a tile. Read by the session-drop bridge.
  const sessionAnchor = isPrimary ? 'workspace' : `session-tile:${storedId ?? ''}`
  const awaitingResponse = useStore(view.$awaitingResponse)
  const busy = useStore(view.$busy)
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
  const gatewayOpen = gatewayState === 'open'
  const introPersonality = useStore($introPersonality)
  const introSeed = useStore($introSeed)
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
  // does not wipe an in-progress draft or orphan /queue entries.
  const queueSessionKey = useMemo(
    () => resolveComposerSessionKey(selectedSessionId, sessions),
    [selectedSessionId, sessions]
  )

  // When the tip row arrives after compression, migrate any tip-keyed stash onto
  // the durable lineage key before the composer remounts onto that key.
  useEffect(() => {
    if (!selectedSessionId || !queueSessionKey || selectedSessionId === queueSessionKey) {
      return
    }

    migrateSessionDraft(selectedSessionId, queueSessionKey)
    migrateQueuedPrompts(selectedSessionId, queueSessionKey)
  }, [queueSessionKey, selectedSessionId])

  // A tile IS its session — no route involved, never "mismatched".
  const routedSessionId = isPrimary ? routeSessionId(location.pathname) : selectedSessionId
  const isRoutedSessionView = Boolean(routedSessionId)

  // The URL points at a session the store hasn't loaded yet (sidebar / cmd-K /
  // direct nav). Derived in render so the swap reads instantly: the same frame
  // the id changes we drop the old transcript and show the loader, instead of
  // waiting for the resume effect (which paints a frame later) to clear them.
  const routeSessionMismatch = isRoutedSessionView && routedSessionId !== selectedSessionId

  // The compact new-session pop-out skips the wordmark/tagline intro — it's a
  // scratch window, not the full-height empty state.
  const showIntro =
    isPrimary &&
    !isSecondaryWindow() &&
    freshDraftReady &&
    !isRoutedSessionView &&
    !selectedSessionId &&
    !activeSessionId &&
    messagesEmpty

  // Session is still loading if the route references a session we haven't
  // resumed yet. Once `activeSessionId` is set (runtime has resumed), the
  // session exists — even if it has zero messages (a brand-new routed
  // session). The flicker where `busy` flips true briefly during hydrate
  // is handled by `threadLoadingState`'s last-visible-user gate.
  //
  // resumeExhausted: the bounded auto-retry in use-route-resume gave up on this
  // routed session (gateway RPC + REST fallback failed through every attempt).
  // Suppress the loader and show an explicit error + manual Retry instead of
  // spinning forever. Gated on the route matching so a stale latch from another
  // session can't blank the current one.
  const resumeExhausted = isPrimary && isRoutedSessionView && resumeExhaustedSessionId === routedSessionId

  const loadingSession =
    !resumeExhausted && isRoutedSessionView && (routeSessionMismatch || (messagesEmpty && !activeSessionId))

  const threadLoading = threadLoadingState(loadingSession, busy, awaitingResponse, lastVisibleIsUser)
  // Hide the composer in the exhausted error state too: there's no live runtime
  // to send to until a retry rebinds one. Watch windows are pure spectators of a
  // subagent run driven elsewhere — no composer, transcript is read-only.
  const showChatBar = !loadingSession && !resumeExhausted && !isWatchWindow()
  const threadKey = selectedSessionId || activeSessionId || (isRoutedSessionView ? location.pathname : 'new')

  const modelOptionsQuery = useQuery<ModelOptionsResponse>({
    queryKey: ['model-options', activeSessionId || 'global'],
    queryFn: () => {
      if (!activeSessionId) {
        return getGlobalModelOptions()
      }

      if (!gateway) {
        throw new Error('Hermes gateway unavailable')
      }

      return gateway.request<ModelOptionsResponse>('model.options', {
        session_id: activeSessionId,
        explicit_only: true
      })
    },
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
        onCancel={onCancel}
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
            onCancel={onCancel}
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
                bottom: 'calc(var(--composer-measured-height) + var(--status-stack-measured-height) + 0.25rem)'
              }}
            />
          )}
          {/* A session drag hovering an EDGE hands the visual to the zone
              target; the link overlay shows only for the center region. */}
          <ChatDropOverlay kind={overlayKind} />
          <ChatSwapOverlay profile={gatewaySwapTarget} />
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
}
