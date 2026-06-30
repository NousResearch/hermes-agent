import {
  type AppendMessage,
  AssistantRuntimeProvider,
  ExportedMessageRepository,
  type ThreadMessage
} from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import type * as React from 'react'
import { Suspense, useCallback, useMemo, useRef } from 'react'
import { useLocation } from 'react-router-dom'

import { Thread } from '@/components/assistant-ui/thread'
import { Backdrop } from '@/components/Backdrop'
import { PromptOverlays } from '@/components/prompt-overlays'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ErrorState } from '@/components/ui/error-state'
import { getGlobalModelOptions, type HermesGateway } from '@/hermes'
import { useI18n } from '@/i18n'
import type { ChatMessage } from '@/lib/chat-messages'
import { quickModelOptions, sessionTitle, toRuntimeMessage } from '@/lib/chat-runtime'
import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'
import { cn } from '@/lib/utils'
import type { ComposerAttachment } from '@/store/composer'
import { $pinnedSessionIds } from '@/store/layout'
import { $gatewaySwapTarget } from '@/store/profile'
import {
  $activeSessionId,
  $awaitingResponse,
  $busy,
  $contextSuggestions,
  $currentCwd,
  $currentModel,
  $currentProvider,
  $freshDraftReady,
  $gatewayState,
  $introPersonality,
  $introSeed,
  $lastVisibleMessageIsUser,
  $messages,
  $messagesEmpty,
  $resumeExhaustedSessionId,
  $selectedStoredSessionId,
  $sessions,
  sessionPinId
} from '@/store/session'
import { isSecondaryWindow } from '@/store/windows'
import type { ModelOptionsResponse } from '@/types/hermes'

import { routeSessionId } from '../routes'
import { titlebarHeaderBaseClass, titlebarHeaderShadowClass, titlebarHeaderTitleClass } from '../shell/titlebar'

import { ChatDropOverlay } from './chat-drop-overlay'
import { ChatSwapOverlay } from './chat-swap-overlay'
import { ChatBar, ChatBarFallback } from './composer'
import { requestComposerInsert, requestComposerInsertRefs } from './composer/focus'
import { droppedFileInlineRefs, type SessionDragPayload, sessionInlineRef } from './composer/inline-refs'
import type { ChatBarState } from './composer/types'
import { type DroppedFile, partitionDroppedFiles } from './hooks/use-composer-actions'
import { useFileDropZone } from './hooks/use-file-drop-zone'
import { ScrollToBottomButton } from './scroll-to-bottom-button'
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
  onPasteClipboardImage: () => void
  onPickFiles: () => void
  onPickFolders: () => void
  onPickImages: () => void
  onRemoveAttachment: (id: string) => void
  onSteer: (text: string) => Promise<boolean> | boolean
  onSubmit: (
    text: string,
    options?: { attachments?: ComposerAttachment[]; fromQueue?: boolean }
  ) => Promise<boolean> | boolean
  onThreadMessagesChange: (messages: readonly ThreadMessage[]) => void
  onEdit: (message: AppendMessage) => Promise<void>
  onReload: (parentId: string | null) => Promise<void>
  onRestoreToMessage?: (messageId: string) => Promise<void>
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

  const activeStoredSession =
    sessions.find(session => session.id === selectedSessionId || session._lineage_root_id === selectedSessionId) || null

  const title = activeStoredSession ? sessionTitle(activeStoredSession) : 'New session'

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
        className={titlebarHeaderTitleClass}
        style={{
          maxWidth:
            'calc(100vw - var(--titlebar-content-inset,0px) - var(--titlebar-tools-right) - var(--titlebar-tools-width) - 1.5rem)'
        }}
      >
        {typeof window !== 'undefined' &&
        Boolean((window as { __HERMES_MOBILE_STANDALONE__?: boolean }).__HERMES_MOBILE_STANDALONE__) ? (
          // Mobile: plain title, no dropdown — the menu's anchor crowds the
          // Dynamic Island and the menu itself paints under it.
          <h2 className="min-w-0 truncate px-2 text-center text-[0.8125rem] font-medium leading-none text-(--ui-text-secondary)">
            {title}
          </h2>
        ) : (
          <SessionActionsMenu
            align="start"
            onDelete={selectedSessionId ? onDeleteSelectedSession : undefined}
            onPin={selectedSessionId ? onToggleSelectedPin : undefined}
            pinned={selectedIsPinned}
            sessionId={selectedSessionId || activeSessionId || ''}
            sideOffset={8}
            title={title}
          >
            <Button
              className="pointer-events-auto flex h-6 min-w-0 max-w-full gap-1 overflow-hidden border border-transparent bg-transparent px-2 py-0 text-(--ui-text-secondary) hover:border-(--ui-stroke-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground data-[state=open]:border-(--ui-stroke-tertiary) data-[state=open]:bg-(--ui-control-active-background) [-webkit-app-region:no-drag]"
              type="button"
              variant="ghost"
            >
              <h2 className="min-w-0 flex-1 truncate text-[0.75rem] font-medium leading-none">{title}</h2>
              <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="chevron-down" size="0.8125rem" />
            </Button>
          </SessionActionsMenu>
        )}
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
  const storeMessages = useStore($messages)
  const messages = suppressMessages ? NO_MESSAGES : storeMessages
  const runtimeMessageCacheRef = useRef(new WeakMap<ChatMessage, ThreadMessage>())

  const runtimeMessageRepository = useMemo(() => {
    const items: { message: ThreadMessage; parentId: string | null }[] = []
    const branchParentByGroup = new Map<string, string | null>()
    let visibleParentId: string | null = null
    let headId: string | null = null

    for (const message of messages) {
      let parentId = visibleParentId

      if (message.role === 'assistant' && message.branchGroupId) {
        if (!branchParentByGroup.has(message.branchGroupId)) {
          branchParentByGroup.set(message.branchGroupId, visibleParentId)
        }

        parentId = branchParentByGroup.get(message.branchGroupId) ?? null
      }

      const cachedMessage = runtimeMessageCacheRef.current.get(message)
      const runtimeMessage = cachedMessage ?? toRuntimeMessage(message)

      if (!cachedMessage) {
        runtimeMessageCacheRef.current.set(message, runtimeMessage)
      }

      items.push({ message: runtimeMessage, parentId })

      if (!message.hidden) {
        visibleParentId = message.id
        headId = message.id
      }
    }

    return ExportedMessageRepository.fromBranchableArray(items, { headId })
  }, [messages])

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
  const activeSessionId = useStore($activeSessionId)
  const awaitingResponse = useStore($awaitingResponse)
  const busy = useStore($busy)
  const contextSuggestions = useStore($contextSuggestions)
  const currentCwd = useStore($currentCwd)
  const currentModel = useStore($currentModel)
  const currentProvider = useStore($currentProvider)
  const freshDraftReady = useStore($freshDraftReady)
  const gatewayState = useStore($gatewayState)
  const gatewaySwapTarget = useStore($gatewaySwapTarget)
  const gatewayOpen = gatewayState === 'open'
  const introPersonality = useStore($introPersonality)
  const introSeed = useStore($introSeed)
  // PERF: ChatView must not subscribe to $messages — the atom is replaced on
  // every streaming delta flush (~30×/s) and a subscription here re-renders
  // the entire chat shell (header, chat bar, thread wrapper) per token. The
  // runtime that DOES need the messages lives in ChatRuntimeBoundary below;
  // this component only needs streaming-stable derivations.
  const messagesEmpty = useStore($messagesEmpty)
  const lastVisibleIsUser = useStore($lastVisibleMessageIsUser)
  const selectedSessionId = useStore($selectedStoredSessionId)
  const resumeExhaustedSessionId = useStore($resumeExhaustedSessionId)
  const routedSessionId = routeSessionId(location.pathname)
  const isRoutedSessionView = Boolean(routedSessionId)

  // The URL points at a session the store hasn't loaded yet (sidebar / cmd-K /
  // direct nav). Derived in render so the swap reads instantly: the same frame
  // the id changes we drop the old transcript and show the loader, instead of
  // waiting for the resume effect (which paints a frame later) to clear them.
  const routeSessionMismatch = isRoutedSessionView && routedSessionId !== selectedSessionId

  // The compact new-session pop-out skips the wordmark/tagline intro — it's a
  // scratch window, not the full-height empty state.
  const showIntro =
    !isSecondaryWindow() && freshDraftReady && !isRoutedSessionView && !selectedSessionId && !activeSessionId && messagesEmpty

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
  const resumeExhausted = isRoutedSessionView && resumeExhaustedSessionId === routedSessionId

  const loadingSession =
    !resumeExhausted && isRoutedSessionView && (routeSessionMismatch || (messagesEmpty && !activeSessionId))

  const threadLoading = threadLoadingState(loadingSession, busy, awaitingResponse, lastVisibleIsUser)
  // Hide the composer in the exhausted error state too: there's no live runtime
  // to send to until a retry rebinds one.
  const showChatBar = !loadingSession && !resumeExhausted
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

      return gateway.request<ModelOptionsResponse>('model.options', { session_id: activeSessionId })
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
        requestComposerInsert(refs.join(' '), { mode: 'inline', target: 'main' })
      }

      if (osDrops.length) {
        void onAttachDroppedItems(osDrops)
      }
    },
    [currentCwd, onAttachDroppedItems]
  )

  // Dropping a sidebar session inserts an @session link the agent can resolve
  // via session_search (carries the source profile, so cross-profile works).
  const onDropSession = useCallback((session: SessionDragPayload) => {
    requestComposerInsertRefs([sessionInlineRef(session)], { target: 'main' })
  }, [])

  const { dragKind, dropHandlers } = useFileDropZone({ enabled: showChatBar, onDropFiles, onDropSession })

  return (
    <div
      className={cn(
        'relative isolate flex h-full min-w-0 flex-col overflow-hidden bg-(--ui-chat-surface-background)',
        className
      )}
    >
      <Backdrop />
      <ChatHeader
        activeSessionId={activeSessionId}
        isRoutedSessionView={isRoutedSessionView}
        onDeleteSelectedSession={onDeleteSelectedSession}
        onToggleSelectedPin={onToggleSelectedPin}
        selectedSessionId={selectedSessionId}
      />

      <PromptOverlays />

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
          <ChatDropOverlay kind={dragKind} />
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
              queueSessionKey={selectedSessionId}
              sessionId={activeSessionId}
              state={chatBarState}
            />
          </Suspense>
        )}
      </ChatRuntimeBoundary>
    </div>
  )
}
