import { useStore } from '@nanostores/react'
import { useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { NavigateFunction } from 'react-router-dom'

import { formatRefValue } from '@/components/assistant-ui/directive-text'
import { getSession, type HermesGateway, type SessionInfo } from '@/hermes'
import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { cn } from '@/lib/utils'
import { $activeGatewayProfile, ensureGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $gatewayState } from '@/store/session'
import {
  $activePaneId,
  $splitFreshDraft,
  $splitPaneRuntimeSessionId,
  $splitPaneSession,
  type PaneId,
  registerSplitPaneModelDelegate,
  restoreSplitPaneSession,
  setActivePane,
  setSplitPaneSession
} from '@/store/split'
import { useSkinCommand } from '@/themes/use-skin-command'

import { useModelControls } from '../session/hooks/use-model-controls'
import { usePromptActions } from '../session/hooks/use-prompt-actions'
import { useSessionActions } from '../session/hooks/use-session-actions'
import {
  isSessionGoneError,
  resolveStoredSession,
  upsertResolvedSession
} from '../session/hooks/use-session-actions/utils'
import type { RegisteredSessionView, useSessionStateCache } from '../session/hooks/use-session-state-cache'
import { ModelMenuPanel } from '../shell/model-menu-panel'
import type { ClientSessionState } from '../types'

import { requestComposerFocus } from './composer/focus'
import { useComposerActions } from './hooks/use-composer-actions'
import { createPaneRequest, mainPaneSessionProfile, type PaneGatewayRequest } from './pane-request'
import { createSplitPaneView, PaneViewContext } from './pane-view'
import { mirrorPaneToGlobals } from './split-mirror'

import { ChatView } from './index'

type SessionStateCache = ReturnType<typeof useSessionStateCache>

// A NavigateFunction the split's session-actions instance can never reach a
// router through: its navigation is pane-local ($splitPaneSession), and the
// router-only actions (selectSidebarItem/openSettings) are main-pane calls.
const noopNavigate: NavigateFunction = () => undefined

// The §3.1 boot-probe row lookup. The persisted profile tag routes the probe
// with an EXPLICIT profile (the reason the tag exists), so it neither depends
// on $sessions/$profiles having finished their first load — 'open' fires at
// WS connect, BEFORE refreshSessions/refreshProfiles complete — nor violates
// the pane-context explicit-profile REST rule. Outcomes:
//   row           — authoritative hit (upserted so the follow-up resume finds
//                    it cached with its owning profile);
//   row: undefined — authoritative miss (tagged backend 404'd, and the
//                    cross-profile scan found nothing) → close the split;
//   indeterminate — the tagged backend couldn't answer (still spawning,
//                    transport error) → KEEP the split; the resume driver
//                    surfaces the explicit Retry state if it stays down.
async function probeSplitSessionRow(
  storedId: string,
  taggedProfile: null | string
): Promise<{ indeterminate: boolean; row?: SessionInfo }> {
  if (taggedProfile) {
    try {
      const row = await getSession(storedId, taggedProfile)

      upsertResolvedSession(row, storedId)

      return { indeterminate: false, row }
    } catch (err) {
      if (!isSessionGoneError(err)) {
        return { indeterminate: true }
      }

      // Authoritative 404 on the tagged profile: fall through to the scan as
      // a last chance (stale tag — e.g. the session moved profiles).
    }
  }

  return { indeterminate: false, row: await resolveStoredSession(storedId) }
}

interface SplitChatPaneProps {
  ensureSessionState: SessionStateCache['ensureSessionState']
  gateway: HermesGateway | null
  maxVoiceRecordingSeconds?: number
  /** Route a session delete through the controller's pane-aware wrapper. */
  onRemoveSession: (storedSessionId: string) => void
  onToggleSelectedPin: () => void
  openMemoryGraph: () => void
  publishPaneState: SessionStateCache['publishPaneState']
  refreshSessions: () => Promise<void>
  registerPaneView: (view: RegisteredSessionView) => () => void
  /** The shared active-gateway request — wrapped per-pane below. */
  requestGateway: PaneGatewayRequest
  runtimeIdByStoredSessionIdRef: SessionStateCache['runtimeIdByStoredSessionIdRef']
  sessionStateByRuntimeIdRef: SessionStateCache['sessionStateByRuntimeIdRef']
  sttEnabled: boolean
  syncSessionStateToView: SessionStateCache['syncSessionStateToView']
  updateSessionState: SessionStateCache['updateSessionState']
}

/**
 * The second chat pane (design §3.7): owns a factory-created view bundle, its
 * own session/prompt/composer action instances bound to that bundle, the
 * pane-local resume driver (the split's analog of use-route-resume), and the
 * activation choreography — identity mirror install/teardown, gateway profile
 * swap, composer focus. Mounted by DesktopController only while $splitOpen.
 */
export function SplitChatPane({
  ensureSessionState,
  gateway,
  maxVoiceRecordingSeconds,
  onRemoveSession,
  onToggleSelectedPin,
  openMemoryGraph,
  publishPaneState,
  refreshSessions,
  registerPaneView,
  requestGateway,
  runtimeIdByStoredSessionIdRef,
  sessionStateByRuntimeIdRef,
  sttEnabled,
  syncSessionStateToView,
  updateSessionState
}: SplitChatPaneProps) {
  const queryClient = useQueryClient()
  const [view] = useState(() => createSplitPaneView())
  const activePaneId = useStore($activePaneId)
  const paneActive = activePaneId === 'split'
  const splitSession = useStore($splitPaneSession)
  const splitFreshDraft = useStore($splitFreshDraft)
  const gatewayState = useStore($gatewayState)
  const splitActiveSessionId = useStore(view.$activeSessionId)
  const splitSelectedStoredSessionId = useStore(view.$selectedStoredSessionId)
  const splitCurrentCwd = useStore(view.$currentCwd)
  const splitTarget = splitSession?.storedId ?? null

  // Fan-out registration: the state cache projects this pane's viewed session
  // into the bundle. Unregistering on close freezes the view while the truth
  // layer keeps caching (background streaming keeps working).
  useEffect(() => registerPaneView(view), [registerPaneView, view])

  // Expose the split's runtime session id for non-React modules (native
  // notifications treat a split-visible session as foreground).
  useEffect(() => {
    const off = view.$activeSessionId.subscribe(id => $splitPaneRuntimeSessionId.set(id))

    return () => {
      off()
      $splitPaneRuntimeSessionId.set(null)
    }
  }, [view])

  const requestGatewayRef = useRef(requestGateway)
  requestGatewayRef.current = requestGateway

  // Profile-pinned request wrapper (design §4): active → settle the gateway
  // swap then the shared path; inactive (boot restore) → the split profile's
  // own socket, never the active pointer. Stable identity (latest-ref inner)
  // so the action hooks don't re-instantiate per render.
  const [paneRequest] = useState<PaneGatewayRequest>(() =>
    createPaneRequest({
      getProfile: () => $splitPaneSession.get()?.profile ?? null,
      isActive: () => view.isActive(),
      requestGateway: (method, params, timeoutMs, signal) =>
        requestGatewayRef.current(method, params, timeoutMs, signal)
    })
  )

  // Pane-local "route token": createBackendSessionForSend aborts if the pane's
  // target moved while session.create was in flight — same guard the main pane
  // gets from the URL token.
  const getSplitRouteToken = useCallback(
    () => `split:${$splitPaneSession.get()?.storedId ?? ($splitFreshDraft.get() ? 'draft' : 'closed')}`,
    []
  )

  const {
    branchCurrentSession,
    createBackendSessionForSend,
    resumeSession,
    startFreshSessionDraft
  } = useSessionActions({
    activeSessionId: splitActiveSessionId,
    activeSessionIdRef: view.activeSessionIdRef,
    busyRef: view.busyRef,
    creatingSessionRef: view.creatingSessionRef,
    ensureSessionState,
    getRouteToken: getSplitRouteToken,
    navigate: noopNavigate,
    requestGateway: paneRequest,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId: splitSelectedStoredSessionId,
    selectedStoredSessionIdRef: view.selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    syncSessionStateToView,
    updateSessionState,
    view
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
    activeSessionId: splitActiveSessionId,
    activeSessionIdRef: view.activeSessionIdRef,
    branchCurrentSession: branchInNewChat,
    busyRef: view.busyRef,
    createBackendSessionForSend,
    handleSkinCommand,
    openMemoryGraph,
    refreshSessions,
    requestGateway: paneRequest,
    resumeStoredSession: resumeSession,
    selectedStoredSessionIdRef: view.selectedStoredSessionIdRef,
    startFreshSessionDraft,
    sttEnabled,
    updateSessionState,
    view
  })

  // This pane's file pickers / clipboard / drops, writing its own attachments
  // atom and targeting its own session.
  const composer = useComposerActions({
    activeSessionId: splitActiveSessionId,
    attachmentsAtom: view.$composerAttachments,
    currentCwd: splitCurrentCwd,
    requestGateway: paneRequest
  })

  // This pane's model controls: a pick writes THIS bundle's chip (plain,
  // non-persisting setters — the sticky composer-model localStorage keys stay
  // main-pane-only), targets THIS pane's runtime session, and rides the
  // profile-pinned request. The controller's shared instance would have
  // written the MAIN chip + sticky keys for a split pick.
  const { selectModel } = useModelControls({
    activeSessionId: splitActiveSessionId,
    queryClient,
    requestGateway: paneRequest,
    view
  })

  // Route global ModelPickerOverlay picks here while this pane is focused
  // (the overlay mounts outside any PaneViewContext). Latest-ref so the
  // registration survives selectModel identity churn.
  const selectModelRef = useRef(selectModel)
  selectModelRef.current = selectModel

  useEffect(() => {
    registerSplitPaneModelDelegate(selection => selectModelRef.current(selection))

    return () => registerSplitPaneModelDelegate(null)
  }, [])

  const paneModelMenuContent = useMemo(
    () =>
      gatewayState === 'open' ? (
        <ModelMenuPanel gateway={gateway || undefined} onSelectModel={selectModel} requestGateway={paneRequest} />
      ) : null,
    [gateway, gatewayState, paneRequest, selectModel]
  )

  // ── Activation choreography (design §3.6) ─────────────────────────────────
  // Runs synchronously on $activePaneId writes (store subscription, not a
  // React effect) so call-time readers of the identity globals are correct the
  // instant pointerdown-capture flips the pane — before any click handler.
  const mirrorOffRef = useRef<(() => void) | null>(null)
  const publishPaneStateRef = useRef(publishPaneState)
  publishPaneStateRef.current = publishPaneState
  // Gateway profile the main pane's context sat on when focus left it. A
  // fresh main draft has no selected session ⇒ no row tag ⇒
  // mainPaneSessionProfile() null ⇒ ensureGatewayProfile(null) would KEEP the
  // split's profile on switch-back — and the next send would silently create
  // the draft's session on the split's profile. Captured before the split's
  // swap so switch-back restores the draft's real context.
  const mainProfileBeforeSplitFocusRef = useRef<null | string>(null)

  useEffect(() => {
    const apply = (paneId: PaneId) => {
      if (paneId === 'split') {
        if (!mirrorOffRef.current) {
          mainProfileBeforeSplitFocusRef.current = normalizeProfileKey($activeGatewayProfile.get())
          // Warm repaint from the cache, then point the identity globals at
          // this pane (the mirror's immediate fire is the activation copy).
          publishPaneStateRef.current('split')
          mirrorOffRef.current = mirrorPaneToGlobals(view)
        }

        // Swap the active gateway to this pane's profile — identical to what
        // selecting one of its sessions does today. Serialized + no-op when
        // already on-profile; the paneRequest wrapper closes the Enter race.
        void ensureGatewayProfile($splitPaneSession.get()?.profile ?? null)
      } else if (mirrorOffRef.current) {
        mirrorOffRef.current()
        mirrorOffRef.current = null
        publishPaneStateRef.current('main')
        void ensureGatewayProfile(mainPaneSessionProfile() ?? mainProfileBeforeSplitFocusRef.current)
      }

      // Typing lands in the focused pane (both composers subscribe as 'main';
      // the draft engine's active-pane gate routes it).
      requestComposerFocus('main')
    }

    // openSplitPane focuses the split before this mounts — apply immediately.
    if ($activePaneId.get() === 'split') {
      apply('split')
    }

    const off = $activePaneId.listen(apply)

    return () => {
      off()

      // Unmounted while focused (split closed via keybind mid-focus, or the
      // chat route swapped away): restore the main pane's identity globals.
      if (mirrorOffRef.current) {
        mirrorOffRef.current()
        mirrorOffRef.current = null
        publishPaneStateRef.current('main')
      }
    }
  }, [view])

  // ── Pane-local resume driver — the split's use-route-resume ───────────────
  // Waits for the gateway to open; the first target after mount is a boot
  // restore and gets the §3.1 probe (404/archived → split closes instead of
  // mounting a pane that can never resume). Resumes run through the pane's
  // view: while the split is INACTIVE this opens the profile's own socket and
  // never steals the active gateway from the pane the user is typing in.
  // Accepted v1 cosmetic: a stale persisted session shows this pane's loader
  // from first paint until the probe lands (gateway open + one row fetch),
  // then collapses cleanly — the pane is not held hidden during the probe
  // because with the gateway down that would suppress a perfectly valid
  // restore indefinitely.
  const bootProbeRef = useRef(false)
  const resumeSessionRef = useRef(resumeSession)
  resumeSessionRef.current = resumeSession
  const startFreshSessionDraftRef = useRef(startFreshSessionDraft)
  startFreshSessionDraftRef.current = startFreshSessionDraft

  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    if (!splitTarget) {
      // Fresh-draft split: seed the pane view once (openSplitPane with no
      // session, or the dispatcher's startFreshDraft).
      if (splitFreshDraft && !view.$freshDraftReady.get()) {
        startFreshSessionDraftRef.current()
      }

      return
    }

    if (view.selectedStoredSessionIdRef.current === splitTarget && view.activeSessionIdRef.current) {
      return
    }

    let cancelled = false

    void (async () => {
      if (!bootProbeRef.current) {
        bootProbeRef.current = true
        let resolvedProfile: null | string | undefined

        const restored = await restoreSplitPaneSession(async storedId => {
          const probe = await probeSplitSessionRow(storedId, $splitPaneSession.get()?.profile ?? null)

          // Indeterminate (tagged backend still spawning / transport error):
          // keep the split rather than discarding a session that may be fine —
          // returning the current session is a truthy "still valid" for
          // restoreSplitPaneSession. The resume below then either succeeds or
          // arms the pane's explicit Retry state.
          if (probe.indeterminate) {
            return $splitPaneSession.get()
          }

          if (!probe.row || probe.row.archived) {
            return null
          }

          resolvedProfile = probe.row.profile ?? null

          return probe.row
        })

        if (cancelled || !restored || restored.storedId !== splitTarget) {
          return
        }

        // Refresh a missing profile tag from the resolved row so the pinned
        // request wrapper and the gateway keep-set route correctly.
        if (restored.profile === null && resolvedProfile) {
          setSplitPaneSession(splitTarget, resolvedProfile)
        }
      }

      if (cancelled) {
        return
      }

      await resumeSessionRef.current(splitTarget)

      // No bounded auto-retry driver for the split (that's use-route-resume,
      // main-pane-only): a terminal failure surfaces the explicit error +
      // manual Retry instead of an endless loader. onRetryResume clears the
      // latch through resumeSession for a fresh attempt.
      if (!cancelled && view.$resumeFailedSessionId.get() === splitTarget) {
        view.setResumeExhaustedSessionId(splitTarget)
      }
    })()

    return () => {
      cancelled = true
    }
  }, [gatewayState, splitFreshDraft, splitTarget, view])

  const dismissError = useCallback(
    (messageId: string) => {
      const runtimeSessionId = view.activeSessionIdRef.current

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

      // View first, then the cache — same ordering rationale as the main
      // pane's dismissError (the flush merges errors from the current view).
      view.setMessages(clearErrorIn(view.$messages.get()))

      updateSessionState(runtimeSessionId, (state: ClientSessionState) => ({
        ...state,
        messages: clearErrorIn(state.messages)
      }))
    },
    [updateSessionState, view]
  )

  const activate = useCallback(() => setActivePane('split'), [])

  return (
    <PaneViewContext.Provider value={view}>
      {/* Same box contract as the main chat column (ChatView owns its header
          placement); the inset hairline flips with focus — accent while
          active, quiet stroke while not. */}
      <div
        className={cn(
          'relative flex h-full min-h-0 min-w-0 flex-col overflow-hidden',
          paneActive
            ? 'shadow-[inset_0_0_0_1px_color-mix(in_srgb,var(--dt-composer-ring,var(--ui-stroke-secondary))_45%,transparent)]'
            : 'shadow-[inset_0_0_0_1px_var(--ui-stroke-secondary)]'
        )}
        data-pane-active={paneActive ? 'true' : 'false'}
        data-pane-id="chat-split-view"
        onFocusCapture={activate}
        onPointerDownCapture={activate}
      >
        <ChatView
          gateway={gateway}
          maxVoiceRecordingSeconds={maxVoiceRecordingSeconds}
          modelMenuContent={paneModelMenuContent}
          onAddContextRef={composer.addContextRefAttachment}
          onAddUrl={url => composer.addContextRefAttachment(`@url:${formatRefValue(url)}`, url)}
          onAttachDroppedItems={composer.attachDroppedItems}
          onAttachImageBlob={composer.attachImageBlob}
          onBranchInNewChat={messageId => void branchInNewChat(messageId)}
          onCancel={cancelRun}
          onDeleteSelectedSession={() => {
            const selected = view.$selectedStoredSessionId.get()

            if (selected) {
              onRemoveSession(selected)
            }
          }}
          onDismissError={dismissError}
          onEdit={editMessage}
          onPasteClipboardImage={opts => composer.pasteClipboardImage(opts)}
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
          onToggleSelectedPin={onToggleSelectedPin}
          onTranscribeAudio={transcribeVoiceAudio}
        />
      </div>
    </PaneViewContext.Provider>
  )
}
