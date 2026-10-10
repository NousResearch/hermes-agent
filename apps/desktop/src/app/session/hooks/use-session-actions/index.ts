import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'
import type { NavigateFunction } from 'react-router'

import { NO_PROJECT_ID } from '@/app/chat/sidebar/projects/workspace-groups'
import {
  extendRefreshPageToOverlap,
  graftRefreshedTailOntoBackfill,
  olderPageReader
} from '@/app/chat/transcript-backfill'
import { defaultNewSessionTarget, prepareDefaultNewSession } from '@/app/session/new-session-route'
import { revealTreePane } from '@/components/pane-shell/tree/store'
import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'
import {
  deleteSession,
  fetchStoredTranscriptAcrossBackends,
  getAllSessionMessages,
  getLatestSessionMessages,
  setSessionArchived
} from '@/hermes'
import { useI18n } from '@/i18n'
import {
  type ChatMessage,
  preserveLocalAssistantErrors,
  QUESTION_CARD_TOOLS,
  restorePendingClarifyToolCall,
  settlePendingClarifyToolCall,
  stripPendingClarifyProjectionForCache,
  toChatMessages
} from '@/lib/chat-messages'
import { markReasoningEffortPending } from '@/lib/chat-runtime'
import { isMissingRpcMethod } from '@/lib/gateway-rpc'
import { purgeInFlightTurnJournals, recoverInFlightTurnJournal } from '@/lib/inflight-turn-journal'
import { latestSessionTodoSnapshot } from '@/lib/todos'
import { setSessionYolo } from '@/lib/yolo-session'
import { $clarifyRequests, clearClarifyRequest } from '@/store/clarify'
import { announceGoneSessionDraft, announceNewSessionDraftKey, migrateSessionDraft } from '@/store/composer'
import { clearQueuedPrompts, migrateQueuedPrompts } from '@/store/composer-queue'
import { resetSessionBackground } from '@/store/composer-status'
import { $connectionRequests } from '@/store/connection-request'
import {
  isActivePrimary,
  openGatewayForAgent,
  openGatewayForProfile,
  pendingSessionReplay,
  requestGatewayForAgent,
  retainGatewayForAgent
} from '@/store/gateway'
import { $gatewaySwitching } from '@/store/gateway-switch'
import { clearSessionGoal } from '@/store/goals'
import { $pinnedSessionIds } from '@/store/layout'
import { clearNotifications, notify, notifyError } from '@/store/notifications'
import { adoptDraftPreviewTabs, prunePreviewTabsForSession } from '@/store/preview'
import {
  $activeGatewayProfile,
  $gatewaySwapTarget,
  $newChatProfile,
  $profiles,
  $showAllProfiles,
  type AgentProfileRoute,
  ensureGatewayAgent,
  ensureGatewayProfile,
  isLegacyNewChatProfile,
  normalizeProfileKey,
  resolveActiveSourceOwnerRoute,
  resolveNewChatOwnerRoute
} from '@/store/profile'
import { $projectScope } from '@/store/project-scope'
import { projectProfile, resolveNewSessionCwd } from '@/store/projects'
import { clearAllPrompts } from '@/store/prompts'
import { clearStoredTranscriptReadOnly, markStoredTranscriptReadOnly } from '@/store/read-only-transcript'
import {
  $activeSessionStoredIdRotation,
  $connection,
  $currentCwd,
  $currentCwdExplicit,
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $currentServiceTier,
  $messages,
  $newChatWorkspaceTarget,
  $sessions,
  $yoloActive,
  getCurrentModelSource,
  idsShareLineage,
  type NewChatWorkspaceTarget,
  resolveComposerSessionKey,
  rotateFreshDraftKey,
  sessionPinId,
  setActiveSessionId,
  setActiveSessionStoredIdRotation,
  setAwaitingResponse,
  setBusy,
  setCurrentBranch,
  setCurrentCwd,
  setCurrentCwdExplicit,
  setCurrentCwdTransient,
  setCurrentUsage,
  setFreshDraftReady,
  setIntroSeed,
  setMessages,
  setNewChatWorkspaceTarget,
  setResumeExhaustedSessionId,
  setResumeFailedSessionId,
  setSelectedStoredSessionId,
  setSessionOwnerHint,
  setSessionStartedAt,
  setTurnStartedAt,
  setWorkspaceCwdOwner,
  setYoloActive
} from '@/store/session'
import { clearSessionControl } from '@/store/session-control'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { isSessionOwnerResolutionError } from '@/store/session-owner-resolution'
import {
  beginSessionMutation,
  endSessionMutation,
  isSessionRemovalPending,
  tombstoneSessions,
  untombstoneSessions
} from '@/store/session-removal'
import {
  requestForSessionProfile,
  type SessionOwnerRoute,
  sessionOwnerRouteFromRow,
  type SessionOwnerScope,
  type SessionProfileRoute
} from '@/store/session-request-router'
import {
  $sessionTiles,
  closeSessionTile,
  dropSessionState,
  focusOpenSession,
  holdSessionOwnerUntilForeground,
  isSessionInForeground,
  openSessionTile,
  patchSessionTile,
  publishSessionState,
  releaseSessionOwnerHold,
  type SessionTileWorkspaceScope,
  type TileDock
} from '@/store/session-states'
import { broadcastSessionsChanged } from '@/store/session-sync'
import { forgetSessionUnread } from '@/store/session-unread'
import { $archivedSessions } from '@/store/sidebar-archive'
import { clearSessionSubagents } from '@/store/subagents'
import { clearSessionTodos, restoreSessionTodosFromSnapshot } from '@/store/todos'
import { dropTranscriptTail, dropTranscriptTailEverywhere, saveTranscriptTail } from '@/store/transcript-tail-cache'
import { isWatchWindow } from '@/store/windows'
import type {
  SessionCreateResponse,
  SessionMessage,
  SessionMessagesResponse,
  SessionResumeResult,
  UsageStats
} from '@/types/hermes'

import { navigateToWorkspacePage, NEW_CHAT_ROUTE, sessionRoute, SETTINGS_ROUTE } from '../../../routes'
import type { ClientSessionState, SidebarNavItem } from '../../../types'
import {
  pinStoredSessionForOwner,
  releaseStoredSessionPins,
  resumeRouteStillCurrent,
  sessionContextDrift
} from '../session-context-drift'
import { singleFlightSessionResume } from '../use-prompt-actions/single-flight-resume'

import { branchCreateKey } from './branch-create-key'
import { sessionCreateOverrideParams, type SessionCreateOverrides, type SessionSeedMessage } from './create-overrides'
import { markSessionCreatedThisRun, sessionCreatedThisRun } from './created-this-run'
import { captureDisplayHydration } from './display-hydration'
import { reconcilePersistedLiveTurn } from './persisted-live-turn'
import { provisionalTranscriptPaint, transcriptRestScope } from './provisional-transcript'
import { rememberedOwnerForResume } from './remembered-owner'
import { restorePendingApproval } from './restore-pending-approval'
import { pendingClarifyToolPayload, restorePendingClarifyFromSnapshot } from './restore-pending-clarify'
import { projectPendingConnection, restorePendingConnectionFromSnapshot } from './restore-pending-connection'
import { createGatewaySession } from './session-create-request'
import {
  createPersistedDisplayTranscriptProvenance,
  hasPersistedDisplayTranscriptProvenance,
  withoutTranscriptProvenance
} from './transcript-provenance'
import {
  appendLiveSessionProjection,
  applyRuntimeInfo,
  applyStoredSessionPreviewRuntimeInfo,
  type BranchMessage,
  cachedSessionRow,
  chatMessageArraysEquivalent,
  dedupeInflightUserAgainstTranscript,
  dropListedSession,
  findListedSession,
  goneSessionVerdict,
  isSessionGoneError,
  overlayConcurrentMessageChanges,
  patchSessionWorkspace,
  preserveEquivalentTranscript,
  preserveLocalPendingTurnMessages,
  reconcileDurableHistory,
  removeRepresentedLocalLiveProjection,
  resolveResumedBusy,
  resolveSessionProfile,
  resolveStoredSession,
  restoreListedSession,
  selectBranchMessages,
  sessionMatchesStoredId,
  sessionShouldHaveTranscript,
  toBranchMessages,
  upsertOptimisticSession,
  upsertUnlistedSessionOwner
} from './utils'

interface SessionActionsOptions {
  activeSessionId: string | null
  activeSessionIdRef: MutableRefObject<string | null>
  busyRef: MutableRefObject<boolean>
  creatingSessionRef: MutableRefObject<boolean>
  ensureSessionState: (sessionId: string, storedSessionId?: string | null) => ClientSessionState
  getRouteToken: () => string
  getRoutedStoredSessionId: () => null | string
  holdSessionTranscriptView?: (runtimeId: string) => () => void
  navigate: NavigateFunction
  onFreshDraftRouteIntent?: () => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  resetViewSync: () => void
  // Live route session id from the router. Used to drop creatingSessionRef only
  // after navigate to a freshly created/forked stored id has actually landed
  // (setTimeout(0) cleared the guard before the route caught up — #66057).
  routedSessionId: string | null
  runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>>
  selectedStoredSessionId: string | null
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
  syncSessionStateToView: (sessionId: string, state: ClientSessionState) => void
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}

export interface BranchLoadedSessionOptions {
  busy: boolean
  contextDrift?: () => null | string
  cwd?: string
  messageId?: string
  messages: ChatMessage[]
  runtimeId: null | string
  storedSessionId: null | string
}

// How long we keep creatingSessionRef after create/fork navigate before giving up
// if the router never lands on the pending stored id (stuck navigate / lost race).
const CREATE_GUARD_RELEASE_MS = 3_000

// Reflect a stored row's persisted token counts into the live usage atom
// (total is derived, so callers can't drift it out of sync with input/output).
function applyStoredUsage(stored: { input_tokens?: number | null; output_tokens?: number | null }) {
  const input = stored.input_tokens || 0
  const output = stored.output_tokens || 0

  setCurrentUsage(current => ({ ...current, input, output, total: input + output }))
}

function reconcileAuthoritativeChatMessages(
  authoritativeMessages: ChatMessage[],
  previousMessages: ChatMessage[],
  liveProjection?: Pick<SessionResumeResult, 'inflight' | 'queued' | 'session_id'>,
  sourceRows?: SessionMessage[]
): ChatMessage[] {
  if (liveProjection && sourceRows) {
    const reconciled = reconcilePersistedLiveTurn(authoritativeMessages, previousMessages, sourceRows, liveProjection)

    if (reconciled) {
      return reconciled
    }
  }

  return reconcileDurableHistory(
    liveProjection ? appendLiveSessionProjection(authoritativeMessages, liveProjection) : authoritativeMessages,
    previousMessages
  )
}

function reconcileAuthoritativeMessages(
  authoritativeMessages: SessionResumeResult['messages'],
  previousMessages: ChatMessage[],
  liveProjection?: Pick<SessionResumeResult, 'inflight' | 'queued' | 'session_id'>
): ChatMessage[] {
  return reconcileAuthoritativeChatMessages(
    toChatMessages(authoritativeMessages),
    previousMessages,
    liveProjection,
    authoritativeMessages
  )
}

// `session.create` params from the current profile + sticky-UI model/effort/fast,
// ensuring the gateway is on that profile first. Shared by the primary send path
// and the "open in split" tile path; `cwd` is the one thing that differs (the
// live composer cwd for a send, the resolved new-session cwd for a fresh tile).
//
// Resolving null profile to the active gateway's is load-bearing: in global-remote
// mode one backend serves every profile, so an omitted profile silently lands the
// chat on the launch (default) profile — the "rubberbands back to default" bug.
// A no-op for single-profile/local-pooled users (a backend resolves its own launch
// profile to None). Effort/fast still ride as per-session overrides. Model and
// provider only ride when the composer source is 'manual' — a default-sourced
// value is a mirror of Settings → Model and must not pin the new chat.
async function desktopSessionCreateParams(
  cwd: string,
  capturedRoute = resolveNewChatOwnerRoute(),
  requestedProfile?: string,
  legacyProfileIntent = false,
  includeComposerSelection = true
): Promise<Record<string, unknown>> {
  // Treat Send as the linearization point for the visible selector state. The
  // profile handshake below can yield long enough for background config/model
  // refreshes to finish; reading atoms afterward would silently create the
  // session with a different selection than the one the user submitted.
  // Settings → Model while a session is live leaves $currentModel painted with
  // the live agent (applySavedMainModel) and only flips the source to 'default'.
  // Shipping that stale value as an override pins every new chat to the old
  // model. Omit model/provider unless the source is 'manual'.
  const isManualSelection = getCurrentModelSource() === 'manual'

  const selection = {
    effort: $currentReasoningEffort.get().trim(),
    fast: $currentFastMode.get(),
    serviceTier: $currentServiceTier.get().trim(),
    model: isManualSelection ? $currentModel.get().trim() : '',
    provider: isManualSelection ? $currentProvider.get().trim() : ''
  }

  const profile =
    capturedRoute?.profile ||
    requestedProfile ||
    $newChatProfile.get() ||
    normalizeProfileKey($activeGatewayProfile.get())

  if (capturedRoute) {
    await ensureGatewayAgent(capturedRoute.connectionId, profile)
  } else if (legacyProfileIntent) {
    await ensureGatewayProfile(profile, { forceLegacyRoute: true })
  } else {
    await ensureGatewayProfile(profile)
  }

  return {
    cols: 96,
    source: 'desktop',
    ...(cwd && { cwd }),
    // #52589: explicit provenance for the shipped cwd — an inherited app-global
    // workspace must not override the target profile's configured terminal.cwd.
    ...(cwd && { cwd_explicit: $currentCwdExplicit.get() }),
    ...(profile ? { profile: capturedRoute?.targetProfile || profile } : {}),
    ...(includeComposerSelection
      ? {
          ...(selection.model
            ? { model: selection.model, ...(selection.provider ? { provider: selection.provider } : {}) }
            : {}),
          ...(selection.effort ? { reasoning_effort: selection.effort } : {}),
          fast: selection.fast,
          // Only Ultrafast needs the tier: `fast` already pins Priority/normal, and a
          // pre-Ultrafast backend rejects the field (createGatewaySession drops it).
          ...(selection.serviceTier === 'ultrafast' ? { service_tier: 'ultrafast' } : {})
        }
      : {})
  }
}

interface FreshSessionDraftOptions {
  preserveRoute?: boolean
  replaceRoute?: boolean
  rotateFreshDraftKey?: boolean
  workspaceTarget?: NewChatWorkspaceTarget
}

/** Drop the pre-hydration request-id row without copying the other messages.
 *  A copied clarify row looks like a concurrent edit and hydration keeps both. */
function withoutEarlyClarifyProjection(messages: ChatMessage[], requestId: string): ChatMessage[] {
  let changed = false
  const next: ChatMessage[] = []

  for (const message of messages) {
    const parts = message.parts.filter(
      part =>
        !(
          part.type === 'tool-call' &&
          QUESTION_CARD_TOOLS.has(part.toolName) &&
          part.result === undefined &&
          part.toolCallId === requestId
        )
    )

    if (parts.length === message.parts.length) {
      next.push(message)

      continue
    }

    changed = true

    if (parts.length > 0) {
      next.push({ ...message, parts })
    }
  }

  return changed ? next : messages
}

/** Session-state patch for a restored blocking prompt row; the first non-null projection is used. */
function livePromptStreamId(
  ...projections: ({ streamId: string } | null)[]
): { awaitingResponse: false; sawAssistantPayload: true; streamId: string } | Record<string, never> {
  const live = projections.find(Boolean)

  return live ? { awaitingResponse: false, sawAssistantPayload: true, streamId: live.streamId } : {}
}

function normalizeNewChatWorkspaceTarget(target: NewChatWorkspaceTarget): NewChatWorkspaceTarget {
  return typeof target === 'string' ? target.trim() || null : target
}

export function useSessionActions({
  activeSessionId,
  activeSessionIdRef,
  busyRef,
  creatingSessionRef,
  ensureSessionState,
  getRouteToken,
  getRoutedStoredSessionId,
  holdSessionTranscriptView,
  navigate,
  onFreshDraftRouteIntent,
  requestGateway,
  resetViewSync,
  routedSessionId,
  runtimeIdByStoredSessionIdRef,
  selectedStoredSessionId,
  selectedStoredSessionIdRef,
  sessionStateByRuntimeIdRef,
  syncSessionStateToView,
  updateSessionState
}: SessionActionsOptions) {
  const { t } = useI18n()
  const copy = t.desktop
  const resumeRequestRef = useRef(0)
  const transcriptHydrationByRuntimeRef = useRef(new Map<string, symbol>())
  const coldDisplayReadsRef = useRef(new Map<string, symbol>())
  const branchCreateFlightsRef = useRef(new Map<string, Promise<SessionCreateResponse>>())

  // Stored id we just created/forked and navigated to. creatingSessionRef stays
  // true until routedSessionId + selection both agree on this id — clearing via
  // setTimeout(0) let use-route-resume resume the stale route as "stuck" (#66057).
  const pendingCreatedStoredSessionIdRef = useRef<string | null>(null)
  // Route id at the moment we armed pending (often the stale previous session).
  // Distinguishes "router still lagging on A" from "user navigated to C".
  const pendingCreatedFromRouteRef = useRef<string | null>(null)
  const pendingGuardTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const releaseCreatingSessionGuard = useCallback(() => {
    if (pendingGuardTimeoutRef.current != null) {
      clearTimeout(pendingGuardTimeoutRef.current)
      pendingGuardTimeoutRef.current = null
    }

    pendingCreatedStoredSessionIdRef.current = null
    pendingCreatedFromRouteRef.current = null
    creatingSessionRef.current = false
  }, [creatingSessionRef])

  // Arm the create/fork hold: keep creatingSessionRef until the route lands on
  // `storedId`, the user leaves for another route, navigate throws, or the
  // safety timeout fires (so a stuck router can't block resumes forever).
  const armPendingCreatedSession = useCallback(
    (storedId: string) => {
      pendingCreatedStoredSessionIdRef.current = storedId
      pendingCreatedFromRouteRef.current = routedSessionId

      if (pendingGuardTimeoutRef.current != null) {
        clearTimeout(pendingGuardTimeoutRef.current)
      }

      pendingGuardTimeoutRef.current = setTimeout(() => {
        pendingGuardTimeoutRef.current = null

        if (pendingCreatedStoredSessionIdRef.current !== storedId) {
          return
        }

        // Route never caught up. Retry navigate so ChatView can leave the
        // route/selection mismatch loading state; then drop the guard so
        // use-route-resume can self-heal to the URL if navigate still fails.
        try {
          navigate(sessionRoute(storedId), { replace: true })
        } catch {
          // Ignore — release below still unblocks recovery.
        }

        releaseCreatingSessionGuard()
      }, CREATE_GUARD_RELEASE_MS)
    },
    [navigate, releaseCreatingSessionGuard, routedSessionId]
  )

  useEffect(
    () => () => {
      if (pendingGuardTimeoutRef.current != null) {
        clearTimeout(pendingGuardTimeoutRef.current)
      }
    },
    []
  )

  // Drop the create/fork guard once the router catches up — or if the user
  // navigates somewhere other than the pending id (left the pre-create route).
  useEffect(() => {
    const pending = pendingCreatedStoredSessionIdRef.current

    if (!creatingSessionRef.current || !pending) {
      return
    }

    if (routedSessionId === pending && selectedStoredSessionIdRef.current === pending) {
      releaseCreatingSessionGuard()

      return
    }

    const fromRoute = pendingCreatedFromRouteRef.current

    if (routedSessionId !== fromRoute && routedSessionId !== pending) {
      releaseCreatingSessionGuard()
    }
  }, [
    creatingSessionRef,
    releaseCreatingSessionGuard,
    routedSessionId,
    selectedStoredSessionId,
    selectedStoredSessionIdRef
  ])

  // Follow auto-compression's stored-id rotation only while the exact runtime,
  // selection, and route intent still belong to the rotating conversation.
  // The previous implementation carried only the next stored id and navigated
  // unconditionally; a fast A → B → C switch could therefore be overwritten
  // by A's delayed session.info event and visibly jump back to A.
  const storedIdRotation = useStore($activeSessionStoredIdRotation)
  const storedSessions = useStore($sessions)
  const focusedStoredSessionId = useStore($focusedStoredSessionId)
  const routedStoredSessionId = getRoutedStoredSessionId()

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (!storedIdRotation) {
      return
    }

    const selectedAtEffect = selectedStoredSessionIdRef.current
    const previousId = storedIdRotation.previousStoredSessionId
    const nextId = storedIdRotation.nextStoredSessionId

    // A tile can adopt the exact successor before the refreshed sessions list
    // contains it. The rotation itself proves that this focus is the same live
    // runtime; unrelated focused chats still fail the foreground check.
    const focusedOnRotatedLineage = Boolean(
      focusedStoredSessionId &&
      (focusedStoredSessionId === nextId || idsShareLineage(focusedStoredSessionId, nextId, storedSessions))
    )

    const rotationIsStale =
      activeSessionIdRef.current !== storedIdRotation.runtimeSessionId ||
      selectedAtEffect !== previousId ||
      (routedStoredSessionId !== null && routedStoredSessionId !== previousId)

    if (rotationIsStale) {
      // The user moved to another conversation, so this proof must not replay.
      setActiveSessionStoredIdRotation(current => (current === storedIdRotation ? null : current))

      return
    }

    if (!isSessionInForeground(previousId) && !focusedOnRotatedLineage) {
      // Focus moved to an unrelated tile, but route and selection still name
      // this conversation. Keep the proof so steering can use it until focus
      // or the session list catches up.
      return
    }

    // Consume only once the successor can safely take over the visible session.
    setActiveSessionStoredIdRotation(current => (current === storedIdRotation ? null : current))

    // Park unsent draft/queue on the durable lineage key (not the new tip).
    // ChatBar scopes composer state on resolveComposerSessionKey(); migrating
    // onto the tip while the composer is still bound to the root can lose newer
    // live editor text on a brief remount. If the new tip row is not in
    // $sessions yet, resolveComposerSessionKey falls back to the tip id — prefer
    // the previous id (usually the lineage root) in that gap.
    const resolvedNext = resolveComposerSessionKey(nextId, storedSessions)

    const durableKey =
      resolvedNext && resolvedNext !== nextId
        ? resolvedNext
        : (resolveComposerSessionKey(previousId, storedSessions) ?? previousId)

    migrateSessionDraft(previousId, durableKey)
    migrateSessionDraft(nextId, durableKey)
    migrateQueuedPrompts(previousId, durableKey)
    migrateQueuedPrompts(nextId, durableKey)

    setSelectedStoredSessionId(nextId)
    selectedStoredSessionIdRef.current = nextId

    // A route overlay/page has no routed session id, but the underlying selected
    // chat still needs to follow the continuation. Update that selection in
    // place without navigating out of the surface the user deliberately opened.
    if (routedStoredSessionId === previousId) {
      navigate(sessionRoute(nextId), { replace: true })
    }
  }, [
    activeSessionId,
    activeSessionIdRef,
    focusedStoredSessionId,
    getRoutedStoredSessionId,
    navigate,
    routedStoredSessionId,
    selectedStoredSessionId,
    selectedStoredSessionIdRef,
    storedIdRotation,
    storedSessions
  ])

  const startFreshSessionDraft = useCallback(
    (options: boolean | FreshSessionDraftOptions = false) => {
      const draftOptions = typeof options === 'boolean' ? { replaceRoute: options } : options
      const preserveRoute = draftOptions.preserveRoute ?? false
      const replaceRoute = draftOptions.replaceRoute ?? false

      const hasWorkspaceTarget =
        Object.hasOwn(draftOptions, 'workspaceTarget') && draftOptions.workspaceTarget !== undefined

      const workspaceTarget = hasWorkspaceTarget
        ? normalizeNewChatWorkspaceTarget(draftOptions.workspaceTarget)
        : undefined

      if (draftOptions.rotateFreshDraftKey !== false) {
        rotateFreshDraftKey()
      }

      resetViewSync()
      busyRef.current = false
      setBusy(false)
      setAwaitingResponse(false)
      clearNotifications()
      setIntroSeed(seed => seed + 1)
      // A fresh chat takes the screen. Front the workspace — and ONLY that:
      // `$terminalTakeover` is the terminal's open/closed state in every
      // layout, not a Focus-only overlay flag, so clearing it here would close
      // a terminal sitting harmlessly in its own zone (Default, Terminal deck,
      // Quad) and would persist a `false` that leaves the Focus tab unable to
      // mount its workspace on the next boot. Behind another tab the terminal
      // is hidden, not closed: it keeps its PTYs and the overlay stops
      // painting on the pane-hidden marker, which is what actually cleared the
      // chat.
      revealTreePane('workspace')
      // Clear the durable route intent synchronously, before React Router
      // publishes /new. Submit uses that intent to heal an existing-session
      // rebind race, so leaving the old id here could revive it on a very fast
      // New Chat -> Enter sequence.
      onFreshDraftRouteIntent?.()

      if (!preserveRoute) {
        navigate(NEW_CHAT_ROUTE, { replace: replaceRoute })
      }

      setActiveSessionId(null)
      activeSessionIdRef.current = null
      setSelectedStoredSessionId(null)
      selectedStoredSessionIdRef.current = null
      setMessages([])
      setCurrentUsage({
        calls: 0,
        input: 0,
        output: 0,
        total: 0
      })
      setSessionStartedAt(null)
      setTurnStartedAt(null)
      // The composer's model/effort/fast is sticky UI state (persisted in
      // localStorage) — a new chat FOLLOWS your last pick instead of snapping
      // back to the profile default, so we deliberately don't reset it here. The
      // profile default still owns first-run seeding and profile switches (see
      // refreshCurrentModel). Keep the canonical service tier too: clearing
      // it while retaining fast=true would silently downgrade Ultrafast.
      setYoloActive(false)
      setNewChatWorkspaceTarget(hasWorkspaceTarget ? workspaceTarget : undefined)
      // #52589 provenance: only a deliberate string workspace target is an explicit
      // cwd choice. A plain new chat (or a detached `null`) is not — its inherited
      // launch/project workspace must yield to a named profile's configured cwd.
      setCurrentCwdExplicit(typeof workspaceTarget === 'string')

      if (!hasWorkspaceTarget) {
        // In a project → the repo's default-branch checkout; not in a project →
        // detached. So cmd-n does not inherit an unrelated linked worktree.
        // Transient: a resolved default is not the user naming a workspace, and
        // remembering it here would make the NEXT new chat inherit it.
        setCurrentCwdTransient(resolveNewSessionCwd())
      } else if (workspaceTarget === null) {
        setCurrentCwdTransient('')
      } else if (typeof workspaceTarget === 'string') {
        setCurrentCwd(workspaceTarget)
      }

      // A fresh draft resolves its own workspace right here, so it owns it. The
      // selected stored id is null for a draft, and so is the owner — they match,
      // which keeps workspace surfaces live on a new chat instead of treating the
      // draft as an un-re-homed switch (#71254).
      setWorkspaceCwdOwner(null)
      setCurrentBranch('')
      // Never clear the composer here — ChatBar's per-thread draft swap owns it.
      setFreshDraftReady(true)
    },
    [activeSessionIdRef, busyRef, navigate, onFreshDraftRouteIntent, resetViewSync, selectedStoredSessionIdRef]
  )

  const createBackendSessionForSend = useCallback(
    async (
      preview: string | null = null,
      seedMessages?: SessionSeedMessage[],
      // Create the session titled or at a pinned reasoning effort (guided
      // onboarding mints its welcome chat this way). The owning profile is NOT
      // an override — point $newChatProfile at it first (selectProfile-style)
      // so the create lands on that profile's own backend and every later
      // ambient RPC follows.
      createOverrides?: SessionCreateOverrides
    ): Promise<string | null> => {
      const startingStoredSessionId = selectedStoredSessionIdRef.current
      const startingRouteToken = getRouteToken()

      creatingSessionRef.current = true

      try {
        // An explicit one-shot workspace target (null → detached, string → that
        // folder) wins; otherwise the live cwd, then the project-aware default
        // (resolveNewSessionCwd — a project's new session keeps its repo cwd).
        // Home is an explicit detached scope: do not let a stale live cwd from
        // the previously selected project leak into this new session (#84220).
        const workspaceTarget = $newChatWorkspaceTarget.get()
        const homeScope = $projectScope.get() === NO_PROJECT_ID

        const cwd =
          workspaceTarget === null || (workspaceTarget === undefined && homeScope)
            ? ''
            : typeof workspaceTarget === 'string'
              ? workspaceTarget.trim()
              : $currentCwd.get().trim() || resolveNewSessionCwd()

        // The EXACT owner for this create: an explicit agent route, else the
        // (registry source, profile) pair the draft was made on. Read ONCE at
        // the send linearization point and threaded through the create RPC,
        // the owner hint, the optimistic row and the failure cleanup, so the
        // profile-rail path (selectProfile clears $newChatRoute) can no longer
        // reduce the owner to a bare profile name that later RPCs dial on a
        // different socket than the one that minted the runtime.
        const capturedRoute = resolveNewChatOwnerRoute()
        const capturedProfile = $newChatProfile.get() || normalizeProfileKey($activeGatewayProfile.get())
        const legacyProfileIntent = isLegacyNewChatProfile(capturedProfile)

        const params = {
          ...(await desktopSessionCreateParams(cwd, capturedRoute, capturedProfile, legacyProfileIntent)),
          ...sessionCreateOverrideParams(createOverrides, seedMessages)
        }

        // Lease the owner socket for the whole create → owner-publication
        // sequence (#93602 primitive). The per-request lease inside
        // requestGatewayForAgent ends when session.create returns; the
        // foreground hold below takes over from that point until the created
        // chat is selected. Between the two, nothing may close the socket
        // that just minted the runtime.
        //
        // 'foreground' spawn priority (#102281 primitive): this is the user
        // hitting send on a fresh chat, so a cold spawn must not queue behind
        // background roster hydration on a saturated pool. The retain is the
        // first dial, so it carries the tag as well as the create RPC.
        const releaseCreateLease = capturedRoute
          ? await retainGatewayForAgent(capturedRoute.connectionId, capturedRoute.profile, {
              spawnPriority: 'foreground'
            })
          : () => undefined

        let created: SessionCreateResponse
        let stored: null | string

        try {
          created = await createGatewaySession(capturedRoute, params, requestGateway)

          stored = created.stored_session_id ?? null

          // Record the EXACT owner the moment a routed create returns a stored
          // id — before the drift check, the optimistic row, navigation, or any
          // session-scoped RPC can resolve this session's owner. The route is
          // the only authority: in All-profiles / Bot routing the ambient
          // $activeGatewayProfile stays on `default` while the session lives on
          // `capturedRoute` (e.g. local::omar). Without this hint the optimistic
          // row (stamped from ambient) was the only owner record, so the first
          // turn ran on omar and every later session-scoped RPC resolved the row
          // as `default` and 4001'd "session not found".
          if (stored && capturedRoute) {
            setSessionOwnerHint(stored, capturedRoute)
            // Pin the owner socket until the foreground publication (route →
            // $selectedStoredSessionId) covers it, so a prune or lease release
            // in that gap cannot close the runtime before the first prompt.
            holdSessionOwnerUntilForeground(stored, capturedRoute)
          }
        } finally {
          releaseCreateLease()
        }

        // Only a genuine move to a DIFFERENT chat mid-create should orphan the
        // session we just minted. The active runtime ref is deliberately not a
        // prong: background gateway events retarget it while other sessions
        // stream (#47709 class), and the seconds-long session.create round-trip
        // (server-side agent + MCP init) makes that churn near-certain — every
        // genuine user switch retargets selection AND route synchronously
        // anyway. submitTargetStoredId is the just-created stored session, so
        // our own upcoming re-home onto it never reads as drift.
        const drift = sessionContextDrift({
          startRouteToken: startingRouteToken,
          nowRouteToken: getRouteToken(),
          startSelectedStoredId: startingStoredSessionId,
          nowSelectedStoredId: selectedStoredSessionIdRef.current,
          submitTargetStoredId: stored
        })

        if (drift) {
          console.warn('[submit-drift-abort]', drift, { phase: 'mid-create' })

          // Close on the backend that minted the session: the ambient socket
          // is a different machine/profile for a routed create and would
          // 4001 while the orphan lives on (and later ws-orphan-reaps) there.
          const closeCreated = capturedRoute
            ? requestGatewayForAgent(capturedRoute.connectionId, capturedRoute.profile, 'session.close', {
                session_id: created.session_id
              })
            : requestGateway('session.close', { session_id: created.session_id })

          await closeCreated.catch(() => undefined)

          if (stored) {
            releaseSessionOwnerHold(stored)
          }

          return null
        }

        resetViewSync()
        activeSessionIdRef.current = created.session_id
        selectedStoredSessionIdRef.current = stored
        ensureSessionState(created.session_id, stored)

        if (stored) {
          markSessionCreatedThisRun(stored)
          // Seed the sidebar preview with the user's first message so the row
          // reads meaningfully while the turn is in flight, instead of flashing
          // "Untitled session" until the turn persists and auto-title runs. The
          // server later returns its own preview/title and supersedes this.
          // The row carries the create route's exact owner (backend profile +
          // connection), never the ambient profile — see upsertOptimisticSession.
          upsertOptimisticSession(created, stored, null, preview?.trim() || null, null, undefined, capturedRoute)
          // Anything still parked under the pre-session draft bucket belongs
          // to this chat now (#114122); the composer moves it on scope swap.
          announceNewSessionDraftKey(stored)
          // The draft's preview tabs follow it the same way (#73890).
          adoptDraftPreviewTabs(stored)
          createOverrides?.onComposerScopeAssigned?.(stored)
          // Hold creatingSessionRef until the route lands on `stored` (release
          // effect above). setTimeout(0) raced use-route-resume back onto the
          // previous session (#66057).
          armPendingCreatedSession(stored)

          try {
            navigate(sessionRoute(stored), { replace: true })
          } catch {
            releaseCreatingSessionGuard()
          }

          // Other windows (e.g. the main window when this is the pop-out) can't
          // see this session until they re-pull the shared list.
          broadcastSessionsChanged()
        }

        setFreshDraftReady(false)
        setNewChatWorkspaceTarget(undefined)
        setActiveSessionId(created.session_id)
        setSelectedStoredSessionId(stored)
        const runtimeStartedAt = Date.now()
        setSessionStartedAt(runtimeStartedAt)
        const yoloArmed = $yoloActive.get()
        const runtimeInfo = applyRuntimeInfo(created.info)

        updateSessionState(
          created.session_id,
          state => ({ ...state, ...(runtimeInfo ?? {}), runtimeStartedAt }),
          stored
        )

        // User may have armed YOLO on the new-chat draft before the runtime
        // session existed — apply it to the freshly created session.
        if (yoloArmed) {
          await setSessionYolo(requestGateway, created.session_id, true).catch(() => undefined)
        }

        return created.session_id
      } finally {
        // Keep the guard up while a navigate to the new stored id is pending;
        // otherwise clear immediately (abort, error, or create without stored id).
        if (!pendingCreatedStoredSessionIdRef.current) {
          creatingSessionRef.current = false
        }
      }
    },
    [
      activeSessionIdRef,
      armPendingCreatedSession,
      creatingSessionRef,
      ensureSessionState,
      getRouteToken,
      navigate,
      releaseCreatingSessionGuard,
      requestGateway,
      resetViewSync,
      selectedStoredSessionIdRef,
      updateSessionState
    ]
  )

  const submitTextToNewSession = useCallback(
    async (text: string, owner?: string): Promise<{ runtimeSessionId: string; sessionId: string }> => {
      // IPC delivers the quick-entry submit as one task, and the drift guard
      // classifies by route/selection tokens. Capture them BEFORE the create:
      // the session.create round-trip is seconds long, and this call's own
      // re-home onto the created session must never read as user drift
      // (same contract as createBackendSessionForSend's starting tokens).
      const startingRouteToken = getRouteToken()
      const startingSelectedStoredId = selectedStoredSessionIdRef.current
      const params = await desktopSessionCreateParams(resolveNewSessionCwd())
      const created = await requestGateway<SessionCreateResponse>('session.create', params)
      const stored = created.stored_session_id

      if (!stored) {
        throw new Error('The new session did not return a stored id.')
      }

      // Only a genuine user move to a DIFFERENT chat mid-create orphans the
      // minted session; our own re-home below names it, so it is not drift.
      const drift = sessionContextDrift({
        startRouteToken: startingRouteToken,
        nowRouteToken: getRouteToken(),
        startSelectedStoredId: startingSelectedStoredId,
        nowSelectedStoredId: selectedStoredSessionIdRef.current,
        submitTargetStoredId: stored
      })

      if (drift) {
        console.warn('[submit-drift-abort]', drift, { phase: 'quick-entry-new' })
        throw new Error(`Quick Entry destination changed mid-create: ${drift}`)
      }

      // The owner is the requesting submit's correlation when the caller knows
      // it (quick entry); otherwise this call owns its own generation.
      const pinOwner = owner ?? `new-session-${created.session_id}`
      pinStoredSessionForOwner(pinOwner, stored)

      try {
        markSessionCreatedThisRun(stored)
        runtimeIdByStoredSessionIdRef.current.set(stored, created.session_id)
        ensureSessionState(created.session_id, stored)
        upsertOptimisticSession(created, stored, null, text.trim())
        // Submit the exact runtime id returned by session.create so this
        // atomic path cannot fall back to a route token (#85590).
        await requestGateway('prompt.submit', { session_id: created.session_id, text })
        navigate(sessionRoute(stored), { replace: true })

        return { runtimeSessionId: created.session_id, sessionId: stored }
      } finally {
        // Terminal transition for this owner: accepted, failed, or cancelled.
        // Owner-scoped pins cannot strand another request, so no tick budget is
        // needed to force-release.
        releaseStoredSessionPins(pinOwner)
      }
    },
    [
      ensureSessionState,
      getRouteToken,
      navigate,
      requestGateway,
      runtimeIdByStoredSessionIdRef,
      selectedStoredSessionIdRef
    ]
  )

  const selectSidebarItem = useCallback(
    (item: SidebarNavItem) => {
      if (item.action === 'new-session') {
        prepareDefaultNewSession()
        setWorkspaceScope('sessions')
        startFreshSessionDraft()

        return
      }

      if (item.route) {
        navigateToWorkspacePage(navigate, item.route)
      }
    },
    [navigate, startFreshSessionDraft]
  )

  /** Create a fresh session and open it as a tile — leaves the primary chat alone.
   *  Used by the New session row's "Open in split" menu and the tab-strip "+".
   *
   *  `listed` (default true) controls sidebar visibility. A brand-new backend
   *  session is IN-MEMORY only until its first turn persists a row, so
   *  `listSessions(min_messages=1)` already hides an unused one — the sidebar
   *  pollution comes solely from the optimistic upsert here. The tab-strip "+"
   *  passes `listed: false` so an unused new tab never clutters the session
   *  list (Cursor-style draft tab); it surfaces on the next refresh once the
   *  first message persists a turn. "Open in split" keeps the listed behavior. */
  const openNewSessionTile = useCallback(
    async (
      dir: TileDock = 'right',
      options?: {
        anchor?: string
        before?: null | string
        cwd?: null | string
        listed?: boolean
        profile?: string
        route?: AgentProfileRoute | null
        workspaceScope?: SessionTileWorkspaceScope
      }
    ) => {
      const listed = options?.listed ?? true

      try {
        // A tile anchored at a project path belongs to the profile the project
        // tree is rendered under — the same owner the fresh-draft "+" pins
        // (#79005). The occupied-chat "+" and project-row drags reach this
        // path instead, where a stale $newChatProfile pin would otherwise win
        // and land the session in the wrong profile (#124265). All-profiles
        // view has no owner and keeps the ordinary fallback.
        const projectOwnerProfile =
          options?.profile === undefined && typeof options?.cwd === 'string'
            ? (projectProfile() ?? undefined)
            : undefined

        const optionProfile = options?.profile ?? projectOwnerProfile

        // Fresh tile → the caller's workspace when one was named (the sidebar
        // "+" on a project/worktree lane), explicit null means Home/detached,
        // else the resolved new-session cwd (project scope → configured default).
        // `options?.cwd || resolve…` is wrong for Home: null is falsy and used
        // to fall through into the last project folder while main chat was
        // occupied (openTab path for "New session in Home").
        const explicitTarget =
          optionProfile !== undefined || options?.cwd !== undefined || options?.workspaceScope?.ownerRoute !== undefined

        const defaultTarget = options?.route === undefined && !explicitTarget ? defaultNewSessionTarget() : null

        // The project tree is rendered by the ACTIVE source: its profile pairs
        // with that source, never with the source a stale new-chat pin captured
        // on another connection (right profile, wrong host).
        const capturedRoute =
          options?.route !== undefined
            ? options.route
            : (options?.workspaceScope?.ownerRoute ??
              (defaultTarget
                ? defaultTarget.route
                : projectOwnerProfile
                  ? resolveActiveSourceOwnerRoute(projectOwnerProfile)
                  : resolveNewChatOwnerRoute(optionProfile)))

        // A named local profile uses the legacy profile-only transport (no
        // connectionId). Tab-strip "+" omits `options.profile`; the draft or
        // active profile is still the owner. Unique non-default local roster
        // names stay authoritative; default/remote/duplicate stay unresolved.
        const requestedProfile = normalizeProfileKey(
          optionProfile || defaultTarget?.profile || $newChatProfile.get() || $activeGatewayProfile.get()
        )

        const legacyOwnerProfile =
          options?.route === undefined &&
          !capturedRoute &&
          requestedProfile !== null &&
          requestedProfile !== 'default' &&
          $connection.get()?.mode !== 'remote' &&
          $profiles.get().filter(profile => normalizeProfileKey(profile.name) === requestedProfile).length === 1
            ? requestedProfile
            : undefined

        const workspaceScope: SessionTileWorkspaceScope = {
          ...(options?.workspaceScope ?? { workspaceMode: 'sessions' }),
          ...(legacyOwnerProfile ? { ownerProfile: legacyOwnerProfile } : {})
        }

        const cwd =
          options?.cwd === null ? '' : typeof options?.cwd === 'string' ? options.cwd.trim() : resolveNewSessionCwd()

        // #52589 provenance for the tile path: an explicitly-passed cwd is a
        // deliberate workspace pick; a resolved default is inherited.
        setCurrentCwdExplicit(typeof options?.cwd === 'string')

        // Bot-workspace tabs target an agent profile without switching the
        // window's ambient composer. Do not leak that unrelated session's
        // composer selection (manual model/provider, reasoning effort, fast
        // flag) into the bot's chat; omitting them lets the selected profile
        // supply its configured defaults. Ordinary Sessions tiles keep the
        // sticky composer override.
        //
        // No `hidden` here, in either mode. Only Bot Mode's PLUMBING sessions
        // are born hidden, and each mints its own row: the canonical Bot Chat
        // (`hermes-bots/canonical-chat.ts`) and group member sessions
        // (`hermes-bots/group-turns.ts`). Every session this path creates is a
        // side chat the user asked for by hand — "New chat with this bot" and
        // the Bot Mode tab-strip "+" / ⌘T — so it is an ordinary conversation
        // in the bot's profile and stays listed, exactly as
        // `apps/desktop/src/AGENTS.md` and the hide sweep's title allow-list
        // (`hermes-bots/session-sweep.ts`) already promise. Blanket-hiding the
        // mode stranded them: unlisted in the Sessions sidebar, skipped by
        // `/resume`, and reachable only while their tab stayed open, since the
        // bot row opens the canonical chat and "Open recent session" reads
        // `last_session`, which never reports a hidden row.
        const params = await desktopSessionCreateParams(
          cwd,
          capturedRoute,
          requestedProfile,
          options?.route === null || defaultTarget?.route === null,
          workspaceScope.workspaceMode !== 'bots'
        )

        // Same lease chain as createBackendSessionForSend: owner socket held
        // across the create, then the foreground hold carries it until the
        // tile is mounted ($sessionTiles names the owner from then on). Same
        // 'foreground' spawn priority too: "New session" / tab-strip "+" is a
        // direct user click, not background hydration.
        const releaseCreateLease = capturedRoute
          ? await retainGatewayForAgent(capturedRoute.connectionId, capturedRoute.profile, {
              spawnPriority: 'foreground'
            })
          : () => undefined

        let created: SessionCreateResponse
        let stored: string | undefined

        try {
          created = await createGatewaySession(capturedRoute, params, requestGateway)

          stored = created.stored_session_id

          if (stored && capturedRoute) {
            // Same ownership transition as createBackendSessionForSend: the
            // route that minted the session is its exact owner from this
            // moment on, and its socket stays pinned until the tile mounts.
            setSessionOwnerHint(stored, capturedRoute)
            holdSessionOwnerUntilForeground(stored, capturedRoute)
          } else if (stored && legacyOwnerProfile) {
            // The tile below persists this bare owner as the stored-id hint;
            // bridge the create-to-mount gap with the same profile pool.
            holdSessionOwnerUntilForeground(stored, legacyOwnerProfile)
          }
        } finally {
          releaseCreateLease()
        }

        if (!stored) {
          const closeCreated = capturedRoute
            ? requestGatewayForAgent(capturedRoute.connectionId, capturedRoute.profile, 'session.close', {
                session_id: created.session_id
              })
            : requestGateway('session.close', { session_id: created.session_id })

          await closeCreated.catch(() => undefined)
          notify({ kind: 'error', title: copy.sessionUnavailable, message: copy.createSessionFailed })

          return
        }

        markSessionCreatedThisRun(stored)

        // Seed the per-runtime cache so the tile renders immediately without a
        // redundant resume. Only add the row to the SIDEBAR when `listed` — an
        // unlisted (draft) tab stays out of the session list until its first
        // turn persists and a refresh surfaces it. An unlisted draft still
        // records an ownership stub (same stamps, off-list atom): without it a
        // draft minted on the legacy ambient route (null route) owns NOTHING
        // the ladder reads — no tile route, no hint, no row — and its
        // immediate session.resume fails closed on multi-profile installs
        // (#102792).
        if (listed) {
          upsertOptimisticSession(created, stored, null, null, null, undefined, capturedRoute)
        } else {
          upsertUnlistedSessionOwner(created, stored, capturedRoute)
        }

        // A tile lives in its OWN worktree, so it must not run the full
        // foreground composer publish. A CENTER tile is the focused surface,
        // though, and the Files pane still keys off the global `$currentCwd` —
        // so the right rail kept showing the previous session's tree when a
        // Project "+" created a session while the main chat was occupied
        // (#76696). Split/side tiles deliberately stay isolated.
        const runtimeInfo = applyRuntimeInfo(created.info, { foreground: false })
        updateSessionState(created.session_id, state => (runtimeInfo ? { ...state, ...runtimeInfo } : state), stored)

        openSessionTile(stored, dir, options?.anchor, options?.before, workspaceScope)
        patchSessionTile(stored, { runtimeId: created.session_id })

        if (dir === 'center' && runtimeInfo?.cwd) {
          setCurrentCwdTransient(runtimeInfo.cwd)
          setWorkspaceCwdOwner(stored)
        }

        focusOpenSession(stored, workspaceScope)

        if (listed) {
          broadcastSessionsChanged()
        }
      } catch (error) {
        notifyError(error, copy.createSessionFailed)
      }
    },
    [copy, requestGateway, updateSessionState]
  )

  const openSettings = useCallback(() => {
    navigate(SETTINGS_ROUTE)
  }, [navigate])

  const closeSettings = useCallback(() => {
    if (selectedStoredSessionId) {
      navigate(sessionRoute(selectedStoredSessionId))

      return
    }

    navigate(NEW_CHAT_ROUTE)
  }, [navigate, selectedStoredSessionId])

  const resumeSession = useCallback(
    async (storedSessionId: string, replaceRoute = false, capturedOwner?: SessionProfileRoute) => {
      // Delete/archive tombstones the durable id before the route flips, and
      // requestSessionResume already refuses to queue for a doomed id. This is
      // the actuator-side half of the same rule: a resume that was queued
      // BEFORE the tombstone (an idle-reap 4001 racing the delete) must not
      // re-select the chat and toast "Resume failed / Session not found".
      if (isSessionRemovalPending(storedSessionId)) {
        return
      }

      const requestId = resumeRequestRef.current + 1
      const routeToken = getRouteToken()
      resumeRequestRef.current = requestId
      const resumedSameSelectedSession = selectedStoredSessionIdRef.current === storedSessionId

      const isCurrentResume = () =>
        resumeRequestRef.current === requestId &&
        selectedStoredSessionIdRef.current === storedSessionId &&
        resumeRouteStillCurrent(routeToken, getRouteToken(), storedSessionId)

      // A reconnect re-resumes the runtime this view is streaming. Let its
      // replay land while that runtime still owns the view. Otherwise the REST
      // read paints the finished turn first and the replayed rows are then
      // overlaid onto it as concurrent runtime changes: the turn shows twice.
      const viewRuntimeId = resumedSameSelectedSession ? activeSessionIdRef.current : null
      const viewReplay = viewRuntimeId ? pendingSessionReplay(viewRuntimeId) : undefined

      if (viewReplay) {
        await viewReplay

        if (!isCurrentResume()) {
          return
        }
      }

      const resumeStartMessages = resumedSameSelectedSession ? $messages.get() : []

      // Paint the click before the profile-resolve / gateway-swap awaits below,
      // so there's zero dead air: highlight the row instantly (the sidebar reads
      // $selectedStoredSessionId) and drop the previous session's transcript so
      // the thread shows this session's proven warm transcript or its loader
      // instead of the old session lingering until resume lands. Setting the
      // ref here is also what use-route-resume's self-heal assumes ("set
      // synchronously at resume entry").
      setFreshDraftReady(false)
      clearNotifications()
      resetViewSync()
      setSelectedStoredSessionId(storedSessionId)
      selectedStoredSessionIdRef.current = storedSessionId

      // A session is EITHER the main thread OR a tile — never both. openSessionTile
      // enforces this from the tile side (it refuses to tile the selected session);
      // this enforces it from the main side. Loading an existing session into main
      // (cold-start restore, a pasted/⌘K route, a notification jump) while it's also
      // an open tile would paint the same transcript twice — the workspace pane from
      // the route and the tile pane in parallel, both fighting one runtime. Drop the
      // now-redundant tile so main owns it. Runs before the async awaits below (and
      // before the selection listener homes focus) so the tile is gone the same tick
      // the route takes over; the warm cache/runtime binding survives for main to reuse.
      if ($sessionTiles.get().some(t => t.storedSessionId === storedSessionId)) {
        closeSessionTile(storedSessionId)
      }

      // Optimistically clear any prior resume-failure latch for this session:
      // we're attempting a fresh resume, so the self-heal in use-route-resume
      // must not keep treating it as stranded. It's re-armed below only if THIS
      // attempt fails terminally (RPC reject + REST fallback failure).
      setResumeFailedSessionId(current => (current === storedSessionId ? null : current))
      // Also clear the exhausted-latch: a fresh attempt (manual Retry, reconnect,
      // reselect) gives the bounded auto-retry counter a clean cycle, so the
      // chat view drops the error state and shows the loader again.
      setResumeExhaustedSessionId(current => (current === storedSessionId ? null : current))

      // A warm cache entry is only trustworthy when it still BELONGS to the
      // session being resumed. A pooled profile backend that gets idle-reaped
      // and respawned (pruneSecondaryGateways) re-mints runtime ids, so a
      // recycled id can resolve to a live-but-DIFFERENT session's cache entry.
      // The session.activate 404 guard below only catches a fully-DEAD id — a
      // recycled-live id 200s, so an unchecked hit paints the wrong transcript
      // under the current route (the "open chat A, chat B loads" bug). On a
      // mismatch the mapping is cross-wired: purge both sides and report a miss
      // so the caller falls through to a full resume that rebinds a correct id.
      const takeWarmCache = (): { runtimeId: string; state: ClientSessionState } | null => {
        const runtimeId = runtimeIdByStoredSessionIdRef.current.get(storedSessionId)
        const state = runtimeId ? sessionStateByRuntimeIdRef.current.get(runtimeId) : undefined

        if (!runtimeId || !state) {
          return null
        }

        if (state.storedSessionId !== storedSessionId) {
          runtimeIdByStoredSessionIdRef.current.delete(storedSessionId)
          sessionStateByRuntimeIdRef.current.delete(runtimeId)
          dropSessionState(runtimeId)

          return null
        }

        return { runtimeId, state }
      }

      // The selection moved above, but a warm runtime is only (re)bound after
      // the awaits below. Until then the foreground may stay on a runtime only
      // if it already IS this session's own warm runtime (a same-session
      // re-resume). Leaving another chat's runtime bound is #89696: the primary
      // view reads the active runtime's slice and the view sync admits only the
      // active runtime, so a turn still streaming there keeps painting under
      // this session's route, and a re-entrant resume of this target would
      // snapshot it as this session's local pending turn (resumeStartMessages).
      const warmAtEntry = takeWarmCache()
      const foregroundIsTarget = warmAtEntry !== null && warmAtEntry.runtimeId === activeSessionIdRef.current

      if (!foregroundIsTarget) {
        setActiveSessionId(null)
        activeSessionIdRef.current = null
        // History load is not turn-busy. Drop the previous session's leftover
        // lock so focusing this session cannot inherit another chat's run.
        busyRef.current = false
        setBusy(false)

        if (!resumedSameSelectedSession) {
          setMessages([])
        }
      }

      // Swap the single live gateway to this session's profile before any
      // gateway call (no-op when it's already on that profile / single-profile).
      // resolveStoredSession finds the row by id (cheap), so an uncached pasted
      // id loads as fast as a sidebar click instead of hanging on a list scan.
      //
      // Only the REMEMBERED hint is validated (remembered-owner.ts); an explicitly captured owner
      // (requestSessionResume with a row route, a plugin open) is authoritative as given.
      const rememberedOwner = capturedOwner ? undefined : rememberedOwnerForResume(storedSessionId)

      // An explicit capture outranks the remembered hint; the hint only
      // fills in when the caller had no route to give.
      const ownerRoute = capturedOwner || rememberedOwner

      // A connection switch clears/reloads the session rows before this path
      // runs, so an untagged row belongs to the connection that supplied the
      // current list. Capture that source before the async metadata lookup. If
      // we reduce it to the profile string `default`, requestForSessionProfile
      // resolves the local default socket and sends an SSH session id to the
      // wrong machine ("resume failed: session not found").
      const ambientConnection = $connection.get()

      // Keep the legacy primary-local profile door: main may resolve a named
      // profile to its own remote override (#94166). A registry secondary is
      // already an explicit source, even when it is This device under Home.
      const ambientConnectionId =
        ambientConnection?.mode === 'remote' || (ambientConnection?.registryScoped && !isActivePrimary())
          ? ambientConnection.connectionId?.trim() || ''
          : ''

      const provisional = provisionalTranscriptPaint(
        storedSessionId,
        () => isCurrentResume() && !resumedSameSelectedSession && !takeWarmCache()
      )

      const listedStored = $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))

      if (ownerRoute || listedStored?.profile) {
        provisional.paint(transcriptRestScope(ownerRoute, listedStored, ambientConnectionId))
      }

      // An unbound warm target whose transcript is already persisted-display
      // authority paints now, display-only, instead of waiting out the awaits
      // below; the fast path re-proves it against the resolved owner before
      // binding. An unproven warm cache can be a compressed runtime tail, so it
      // keeps the loader (#73646).
      if (warmAtEntry && !foregroundIsTarget && listedStored) {
        const entryProvenance = createPersistedDisplayTranscriptProvenance({
          lineageRootId: listedStored._lineage_root_id ?? null,
          scope: transcriptRestScope(ownerRoute, listedStored, ambientConnectionId),
          storedSessionId
        })

        if (hasPersistedDisplayTranscriptProvenance(warmAtEntry.state, entryProvenance)) {
          setMessages(warmAtEntry.state.messages)
        }
      }

      const storedForProfile = await resolveStoredSession(storedSessionId, ownerRoute)
      const sessionProfile = storedForProfile?.profile

      if (!isCurrentResume()) {
        return
      }

      const resolvedConnectionId = ownerRoute?.connectionId || storedForProfile?.connection_id || ambientConnectionId

      // A row spliced from a CONNECTED registry gateway (#88880) carries its
      // owning connection. A row fetched directly after activating a registry
      // gateway can be untagged, so retain the captured ambient connection too.
      // Either way, route by the composite (connection, profile), never by a
      // same-named profile alone.
      const sessionOwner: SessionOwnerScope =
        ownerRoute ||
        (resolvedConnectionId
          ? {
              connectionId: resolvedConnectionId,
              profile: sessionProfile || 'default'
            }
          : sessionProfile)

      // Preserve this resolved source for later prompt/approval RPCs too;
      // otherwise an untagged row falls back to its bare profile after resume.
      // Only for a row the ambient source actually returned: an id that did not
      // resolve (deep link, routed restore) proves nothing about its owner, and
      // a persisted hint would pin it to whichever source was in front.
      if (
        !ownerRoute &&
        storedForProfile &&
        !storedForProfile.connection_id &&
        sessionOwner &&
        typeof sessionOwner === 'object'
      ) {
        setSessionOwnerHint(storedSessionId, sessionOwner)
      }

      const sessionRestScope = transcriptRestScope(ownerRoute, storedForProfile, ambientConnectionId)
      provisional.paint(sessionRestScope)

      // All-profiles / plugin navigation must not steal chrome API-home:
      // dial the owning backend without moving $activeGatewayProfile.
      if ($showAllProfiles.get()) {
        if (resolvedConnectionId) {
          await openGatewayForAgent(resolvedConnectionId, ownerRoute?.profile || sessionProfile || 'default', {
            spawnPriority: 'foreground'
          })
        } else if (sessionProfile) {
          await openGatewayForProfile(normalizeProfileKey(sessionProfile), { spawnPriority: 'foreground' })
        }
      } else if (resolvedConnectionId) {
        await ensureGatewayAgent(resolvedConnectionId, ownerRoute?.profile || sessionProfile || 'default')
      } else {
        await ensureGatewayProfile(sessionProfile)
      }

      // Request-time routing guard for every session-scoped RPC below. The
      // await above REQUESTS the swap, but by dispatch time the active gateway
      // can be back on another profile: a concurrent switch won the
      // gatewaySwitch mutex, an eviction path (idle reap, connection edit,
      // profile delete) re-pointed the active route at the primary, or the
      // target's dial failed and scheduleReconnect left the previous socket
      // active. Sending this session's resume/activate on whatever socket
      // happens to be active then lands it on a backend that has never heard
      // of the session — the backend boots, sits idle, and the renderer burns
      // its bounded retries into the "retries gave up" screen while the bot's
      // own backend is healthy one port over (#89206: local pool AND SSH).
      // requestForSessionProfile re-resolves the route at each call.
      const requestForSession = <T>(method: string, params: Record<string, unknown> = {}): Promise<T> =>
        requestForSessionProfile<T>(sessionOwner, requestGateway, method, params)

      if (!isCurrentResume()) {
        return
      }

      // Re-check after the profile-resolve / gateway-swap awaits above: the
      // cache may have changed, and takeWarmCache re-validates belongs-to and
      // purges a cross-wired mapping before we trust the fast-path.
      const warmHit = takeWarmCache()

      if (warmHit) {
        const cachedRuntimeId = warmHit.runtimeId
        const cachedState = warmHit.state

        const stored =
          $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId)) ?? storedForProfile

        let cachedViewState =
          !cachedState.model && stored?.model != null
            ? {
                ...cachedState,
                model: stored.model || ''
              }
            : cachedState

        if (resumedSameSelectedSession) {
          const messages = preserveLocalPendingTurnMessages(cachedViewState.messages, resumeStartMessages)

          if (messages !== cachedViewState.messages) {
            cachedViewState = { ...cachedViewState, messages }
          }
        }

        if (cachedViewState !== cachedState) {
          sessionStateByRuntimeIdRef.current.set(cachedRuntimeId, cachedViewState)
          publishSessionState(cachedRuntimeId, cachedViewState)
        }

        const expectedProvenance = stored
          ? createPersistedDisplayTranscriptProvenance({
              lineageRootId: stored._lineage_root_id ?? null,
              scope: sessionRestScope,
              storedSessionId
            })
          : null

        const hasValidProvenance = Boolean(
          expectedProvenance && hasPersistedDisplayTranscriptProvenance(cachedViewState, expectedProvenance)
        )

        if (!hasValidProvenance) {
          cachedViewState = withoutTranscriptProvenance(cachedViewState)
        }

        if (sessionShouldHaveTranscript(stored) && cachedViewState.messages.length === 0) {
          runtimeIdByStoredSessionIdRef.current.delete(storedSessionId)
          sessionStateByRuntimeIdRef.current.delete(cachedRuntimeId)
          dropSessionState(cachedRuntimeId)
        } else {
          // Bind the warm runtime immediately so cwd/workspace ownership don't
          // wait on session.activate (#71254). The armed transcript gate is the
          // single suppression authority: it hides only the unproven cached
          // prefix (rows captured at arm time, including a compressed runtime
          // tail — exactly the session-switch flicker, #73646) while rows that
          // arrive live during the hold still paint (#117867). Proven caches
          // and same-session re-resumes never arm the gate, so they paint
          // immediately. The persisted refresh itself still starts after
          // activate reattaches the live transport, so a turn finishing
          // between snapshot and reattach cannot leave a stale partial on
          // screen.
          const shouldRefreshPersistedTranscript = !isWatchWindow()

          const suppressUnprovenWarmTranscript =
            !resumedSameSelectedSession && shouldRefreshPersistedTranscript && !hasValidProvenance

          let releaseHeldTranscriptView = suppressUnprovenWarmTranscript
            ? holdSessionTranscriptView?.(cachedRuntimeId)
            : undefined

          const releaseTranscriptView = () => {
            releaseHeldTranscriptView?.()
            releaseHeldTranscriptView = undefined
          }

          const publishDegradedWarmCache = () => {
            releaseTranscriptView()
            syncSessionStateToView(cachedRuntimeId, cachedViewState)
          }

          setFreshDraftReady(false)
          clearNotifications()
          setSelectedStoredSessionId(storedSessionId)
          selectedStoredSessionIdRef.current = storedSessionId
          setActiveSessionId(cachedRuntimeId)
          activeSessionIdRef.current = cachedRuntimeId
          syncSessionStateToView(cachedRuntimeId, cachedViewState)
          setCurrentCwdTransient(cachedViewState.cwd)
          // The warm cache IS this conversation's own workspace truth, so the
          // switch is already re-homed here. This claim cannot wait for
          // `session.activate`: its missing-RPC compat branch returns before
          // `applyRuntimeInfo` runs, which would leave the workspace marked
          // un-owned for the life of the session (#71254).
          setWorkspaceCwdOwner(storedSessionId)
          setCurrentBranch(cachedViewState.branch)
          setSessionStartedAt(cachedViewState.runtimeStartedAt)

          try {
            const replay = pendingSessionReplay(cachedRuntimeId)

            // Only ordering matters here. A lost socket (false) still goes on
            // to session.activate so its existing branches own the outcome:
            // degraded warm cache on a transport error, cold resume when the
            // runtime is gone, or a normal rebind on a redialed socket (whose
            // history publication is re-gated after the REST read below).
            if (replay) {
              await replay

              if (!isCurrentResume()) {
                return
              }
            }

            let activated: SessionResumeResult | null = null
            const activateStartedAt = Date.now() / 1000
            const activateBaselineState = sessionStateByRuntimeIdRef.current.get(cachedRuntimeId) ?? cachedViewState
            const clarifyRequestIdAtActivateStart = $clarifyRequests.get()[cachedRuntimeId]?.requestId
            const connectionOpIdAtActivateStart = $connectionRequests.get()[cachedRuntimeId]?.opId

            try {
              activated = await requestForSession<SessionResumeResult>('session.activate', {
                session_id: cachedRuntimeId,
                cols: 96,
                // This reattach is usually an automatic reconnect after the socket dropped, not
                // a surface the user changed. The live record keeps whatever source minted it,
                // so name ours or a reconnect can leave the agent pinned to another surface.
                source: 'desktop',
                omit_messages: true
              })
            } catch (error) {
              // Compatibility for older backends. Modern backends require
              // session.activate here because it rebinds the live session's
              // event transport to this newly-opened WebSocket.
              if (!isMissingRpcMethod(error)) {
                throw error
              }

              const usage = await requestForSession<UsageStats>('session.usage', { session_id: cachedRuntimeId })

              if (!isCurrentResume()) {
                return
              }

              if (usage) {
                setCurrentUsage(current => ({ ...current, ...usage, compressions: usage.compressions }))
              }

              publishDegradedWarmCache()

              return
            }

            if (!isCurrentResume()) {
              return
            }

            if (activated.session_key && activated.session_key !== storedSessionId) {
              runtimeIdByStoredSessionIdRef.current.delete(storedSessionId)
              sessionStateByRuntimeIdRef.current.delete(cachedRuntimeId)
              dropSessionState(cachedRuntimeId)
            } else {
              const pendingApproval = restorePendingApproval(activated, cachedRuntimeId)

              const pendingClarifyState = restorePendingClarifyFromSnapshot(
                activated,
                cachedRuntimeId,
                activateStartedAt,
                clarifyRequestIdAtActivateStart
              )

              const pendingClarify = pendingClarifyState.request

              const pendingConnection = restorePendingConnectionFromSnapshot(
                activated,
                cachedRuntimeId,
                activateStartedAt,
                connectionOpIdAtActivateStart
              ).request

              const clarifyAuthoritativelyAbsent =
                pendingClarifyState.authoritativeAbsent && !$clarifyRequests.get()[cachedRuntimeId]

              const staleClarifyAtActivateStart = clarifyAuthoritativelyAbsent
                ? Boolean(settlePendingClarifyToolCall(cachedViewState.messages, {}, false).streamId)
                : false

              const runtimeInfo = applyRuntimeInfo(activated.info)

              // `omit_messages` means the response carries NO transcript, not
              // an empty one — the cache is the base and the live projection is
              // a tail to graft onto it. Reconciling against the empty list
              // instead rebuilds the thread out of the projection alone, so
              // activating a session that is mid-turn somewhere else (leaving
              // HUD mode is exactly that) collapsed the whole conversation down
              // to the in-flight prompt until the turn finished and the
              // post-turn hydrate restored it.
              let activatedMessages = activated.messages_omitted
                ? appendLiveSessionProjection(cachedViewState.messages, activated)
                : activated.messages.length || activated.inflight || activated.queued
                  ? reconcileAuthoritativeMessages(activated.messages, cachedViewState.messages, activated)
                  : cachedViewState.messages

              // #70449: never let the activate snapshot's stale running:false
              // rewind a turn that started while the RPC was in flight — read
              // the freshest cache entry, not the pre-await cachedViewState.
              const latestCachedState = sessionStateByRuntimeIdRef.current.get(cachedRuntimeId)

              const busyChangedWhileActivating = Boolean(
                latestCachedState?.busy &&
                (latestCachedState.turnStartedAt !== activateBaselineState.turnStartedAt ||
                  (latestCachedState.turnLive && !activateBaselineState.turnLive))
              )

              const running =
                (pendingClarifyState.cleared || staleClarifyAtActivateStart) &&
                activated.running === false &&
                !busyChangedWhileActivating
                  ? false
                  : resolveResumedBusy(activated.running ?? cachedViewState.busy, Boolean(latestCachedState?.busy))

              restoreSessionTodosFromSnapshot(cachedRuntimeId, activated.todo_state, running)

              const activatedTurnStartedAt =
                typeof activated.turn_started_at === 'number' && activated.turn_started_at > 0
                  ? activated.turn_started_at * 1000
                  : null

              // Settle the activation snapshot before transcript hydration.
              // Once the attached transport reports a later terminal event,
              // that live state is authoritative and must not be overwritten
              // by the older `running` value after the REST request resolves.
              // The activate snapshot is enough to answer a still-pending clarify.
              // Publish that row in the same view update as needsInput, before
              // transcript REST. An armed hold still hides unproven history, so
              // the question rides a new row the hold cannot swallow by grafting
              // it onto a cutoff assistant. Hydration strips that synthetic id
              // and re-derives one authoritative row.
              const clarifyPayload = pendingClarify ? pendingClarifyToolPayload(pendingClarify) : null

              const earlyClarifyProjection = clarifyPayload
                ? restorePendingClarifyToolCall(suppressUnprovenWarmTranscript ? [] : activatedMessages, clarifyPayload)
                : null

              const projectedTail = earlyClarifyProjection?.messages.at(-1)

              const earlyClarifyMessages =
                earlyClarifyProjection && suppressUnprovenWarmTranscript && projectedTail
                  ? [...activatedMessages, projectedTail]
                  : earlyClarifyProjection?.messages

              const activatedLivenessState = updateSessionState(
                cachedRuntimeId,
                state => ({
                  ...state,
                  ...(runtimeInfo ?? {}),
                  busy: running,
                  awaitingResponse: running && !pendingClarify,
                  // Resumed onto an already-running turn — that IS backend
                  // proof the turn is live (no message.start will replay).
                  turnLive: state.turnLive || running,
                  needsInput:
                    pendingApproval ||
                    Boolean(pendingClarify) ||
                    Boolean(pendingConnection) ||
                    (clarifyAuthoritativelyAbsent ? false : state.needsInput),
                  // Adopting someone else's turn: we'll stream its reply
                  // without ever having received its prompt, so the settle
                  // path must not take the "I saw it all" shortcut.
                  adoptedRunningTurn: state.adoptedRunningTurn || running,
                  turnStartedAt: running ? (activatedTurnStartedAt ?? state.turnStartedAt ?? Date.now()) : null,
                  ...(earlyClarifyProjection
                    ? {
                        messages: earlyClarifyMessages ?? state.messages,
                        ...livePromptStreamId(null, earlyClarifyProjection)
                      }
                    : {})
                }),
                storedSessionId
              )

              busyRef.current = running
              setBusy(running)
              setAwaitingResponse(running && !pendingClarify)
              syncSessionStateToView(cachedRuntimeId, activatedLivenessState)

              // session.activate is the ordering barrier for reconnect recovery:
              // it atomically rebinds a running turn before returning. If the
              // turn is already terminal, this post-barrier REST read sees its
              // durable final row; if it is still running, later deltas/finish
              // events arrive on the newly attached transport. Hydration below
              // reconciles only messages, so those events also retain liveness
              // authority while the request is pending.
              const hydration = captureDisplayHydration({
                flights: transcriptHydrationByRuntimeRef.current,
                key: cachedRuntimeId,
                runtimeIdByStoredSessionIdRef,
                sessionStateByRuntimeIdRef,
                stored,
                storedSessionId
              })

              const persistedTranscriptPromise = shouldRefreshPersistedTranscript
                ? getLatestSessionMessages(storedSessionId, sessionRestScope).catch(() => null)
                : null

              // The persisted REST transcript is the display authority: a live
              // runtime may carry only the agent's compressed context projection,
              // which is intentionally smaller than the user-visible conversation.
              // Reconcile its in-flight/queued tail onto the complete transcript
              // instead of replacing durable history while the turn is running.
              let acceptedPersistedDisplayTranscript = false
              let reconciledCurrentLiveTurn = false

              if (persistedTranscriptPromise) {
                const persisted = await persistedTranscriptPromise
                const replayAtReturn = pendingSessionReplay(cachedRuntimeId)

                if (replayAtReturn && !(await replayAtReturn)) {
                  hydration.release()

                  return
                }

                // Navigation only revokes foreground publication, not this
                // runtime's display read. Edits/rebinds revoke both.
                if (!hydration.owns()) {
                  hydration.release()

                  return
                }

                const activatedStoredSessionId = activated.session_key || activated.resumed

                const persistedMatchesActivatedSession =
                  !persisted?.session_id ||
                  !activatedStoredSessionId ||
                  persisted.session_id === activatedStoredSessionId

                // An empty REST page is not proof the transcript is empty — it's
                // also what a backend respawn returns while its state.db read
                // races the activate response. Reconciling against it anyway
                // wipes the just-restored activate/cache transcript (the same
                // wipe the `activated.messages.length || ...` guard above
                // already prevents for the activate payload itself).
                if (
                  persisted &&
                  persistedMatchesActivatedSession &&
                  (persisted.messages.length || !activatedMessages.length)
                ) {
                  acceptedPersistedDisplayTranscript = Boolean(expectedProvenance)

                  // The REST hydration is a newest-tail page; graft it onto any
                  // older pages the previous view already backfilled so
                  // re-activating a scrolled-back session keeps its history.
                  // A long turn can push every rendered row off the newest page;
                  // read older pages until they overlap so the graft keeps history.
                  const persistedTail = await extendRefreshPageToOverlap(
                    toChatMessages(persisted.messages),
                    cachedViewState.messages,
                    olderPageReader(storedSessionId, sessionRestScope, persisted)
                  )

                  // The extra page reads await; re-check ownership like the read above.
                  if (!hydration.owns()) {
                    hydration.release()

                    return
                  }

                  const persistedMessages = graftRefreshedTailOntoBackfill(persistedTail, cachedViewState.messages)

                  const runtimeMessages = toChatMessages(activated.messages)
                  const previousMessages = removeRepresentedLocalLiveProjection(cachedViewState.messages, activated)

                  const liveProjection = dedupeInflightUserAgainstTranscript(
                    persistedMessages,
                    runtimeMessages,
                    activated
                  )

                  const latestCachedMessages = sessionStateByRuntimeIdRef.current.get(cachedRuntimeId)?.messages

                  const cachedWithoutEarlyClarify =
                    latestCachedMessages && pendingClarify
                      ? withoutEarlyClarifyProjection(latestCachedMessages, pendingClarify.requestId)
                      : latestCachedMessages

                  const currentLiveTurn = reconcilePersistedLiveTurn(
                    persistedMessages,
                    cachedWithoutEarlyClarify ?? previousMessages,
                    persisted.messages,
                    liveProjection
                  )

                  // `null` does not depend on `previous`; retrying the live-turn
                  // reconcile inside the fallback would return `null` again.
                  reconciledCurrentLiveTurn = currentLiveTurn !== null
                  activatedMessages =
                    currentLiveTurn ??
                    reconcileAuthoritativeChatMessages(persistedMessages, previousMessages, liveProjection)
                }
              }

              const currentMessages = sessionStateByRuntimeIdRef.current.get(cachedRuntimeId)?.messages

              // The early publish may have appended a synthetic request-id row so
              // the hold could not hide the question. Drop it before overlaying
              // concurrent edits; the projection below re-derives one row.
              const currentForOverlay =
                currentMessages && pendingClarify
                  ? withoutEarlyClarifyProjection(currentMessages, pendingClarify.requestId)
                  : currentMessages

              // The occurrence-aware path already read the latest cache. An
              // additional identity overlay would restore its consumed tools.
              if (currentForOverlay && !reconciledCurrentLiveTurn) {
                activatedMessages = overlayConcurrentMessageChanges(
                  activatedMessages,
                  cachedViewState.messages,
                  currentForOverlay
                )
              }

              const pendingClarifyProjection = pendingClarify
                ? restorePendingClarifyToolCall(activatedMessages, pendingClarifyToolPayload(pendingClarify))
                : null

              const clearedClarifyProjection = clarifyAuthoritativelyAbsent
                ? settlePendingClarifyToolCall(
                    activatedMessages,
                    pendingClarifyState.cleared ? pendingClarifyToolPayload(pendingClarifyState.cleared) : {},
                    running
                  )
                : null

              const pendingConnectionProjection = projectPendingConnection(
                pendingClarifyProjection?.messages ?? clearedClarifyProjection?.messages ?? activatedMessages,
                pendingConnection
              )

              const visibleActivatedMessages =
                pendingConnectionProjection?.messages ??
                pendingClarifyProjection?.messages ??
                clearedClarifyProjection?.messages ??
                activatedMessages

              if (!running) {
                restoreSessionTodosFromSnapshot(
                  cachedRuntimeId,
                  latestSessionTodoSnapshot(visibleActivatedMessages),
                  false
                )
              }

              releaseTranscriptView()

              const reconcileActivatedState = (state: ClientSessionState): ClientSessionState => {
                // #95595: the reconcilers above always produce fresh
                // message objects, so an unconditional publish replaces the
                // warm-cached array with new-object equivalents and every
                // visible row re-normalizes + remounts (markdown re-parse +
                // shiki re-highlight per row, seconds of main-thread work).
                // Keep the existing array when the content is unchanged —
                // same guard the cold-resume path uses below.
                const messages = preserveEquivalentTranscript(state.messages, visibleActivatedMessages)

                return {
                  ...state,
                  messages,
                  transcriptProvenance:
                    acceptedPersistedDisplayTranscript || hasValidProvenance
                      ? (expectedProvenance ?? undefined)
                      : undefined,
                  ...livePromptStreamId(pendingConnectionProjection, pendingClarifyProjection),
                  ...(clearedClarifyProjection
                    ? {
                        streamId: state.busy ? (clearedClarifyProjection.streamId ?? state.streamId) : null
                      }
                    : {})
                }
              }

              if (isCurrentResume()) {
                const activatedState = updateSessionState(cachedRuntimeId, reconcileActivatedState, storedSessionId)
                syncSessionStateToView(cachedRuntimeId, activatedState)
              } else {
                // updateSessionState stages a view sync based on active runtime,
                // which can lag a newer route intent. Background hydration only
                // publishes the cache; it cannot stage that foreground write.
                const latestState = sessionStateByRuntimeIdRef.current.get(cachedRuntimeId)!
                const activatedState = reconcileActivatedState(latestState)
                sessionStateByRuntimeIdRef.current.set(cachedRuntimeId, activatedState)
                publishSessionState(cachedRuntimeId, activatedState)
              }

              hydration.release()
              // Cache backend transcript truth only. The pending/running bit and
              // any synthetic clarify row are a live resume projection and must
              // not survive after the server-side request expires.
              saveTranscriptTail(
                storedSessionId,
                stripPendingClarifyProjectionForCache(
                  activatedMessages,
                  pendingClarify?.requestId ??
                    pendingClarifyState.cleared?.requestId ??
                    $clarifyRequests.get()[cachedRuntimeId]?.requestId
                ),
                sessionRestScope
              )

              return
            }
          } catch (error) {
            // The cached runtime id was minted by a prior backend instance. A
            // pooled profile backend that gets idle-reaped (pruneSecondaryGateways)
            // and respawned across a profile swap mints fresh ids, so this mapping
            // now 404s ("session not found"). Drop it and fall through to a full
            // resume that rebinds a live runtime id. A transient timeout or
            // transport error is NOT proof that the session is dead: keep the
            // cache and optimistic turn intact for the next reconnect attempt.
            if (!isCurrentResume()) {
              return
            }

            if (!isSessionGoneError(error)) {
              publishDegradedWarmCache()

              return
            }

            runtimeIdByStoredSessionIdRef.current.delete(storedSessionId)
            sessionStateByRuntimeIdRef.current.delete(cachedRuntimeId)
            dropSessionState(cachedRuntimeId)
          } finally {
            releaseTranscriptView()
          }
        }
      }

      setFreshDraftReady(false)
      setActiveSessionId(null)
      activeSessionIdRef.current = null

      // A warm-cache hit at entry skipped the cold-path transcript clear, but the
      // warm path can still bail down to here — an empty-transcript drop, or the
      // cache getting purged during the profile-swap await — so the PREVIOUS
      // session's transcript would leak into this cold resume ("switching
      // sessions shows the same messages"). Clear it so the loader/prefetch
      // paints fresh; guarded so the normal cold path (already cleared) no-ops.
      if (!resumedSameSelectedSession && $messages.get().length > 0 && $messages.get() !== provisional.messages) {
        setMessages([])
      }

      // Retry after a warm runtime was discarded, without clearing a cold
      // tail that already painted before metadata/dial. This remains display-
      // only: REST replaces it rather than grafting onto stale cached history.
      provisional.paint(sessionRestScope)
      const cachedTailPaint = provisional.messages

      // The reconciler's notion of "what was already on screen": a durable
      // cached paint is provisional, not history — report empty so the
      // authoritative transcript replaces it wholesale.
      const viewMessagesForReconcile = (): ChatMessage[] => {
        const current = $messages.get()

        return cachedTailPaint !== null && current === cachedTailPaint ? [] : current
      }

      // A history load is not a live turn. Do not mark the incoming session
      // busy — running ≠ loading, and a leftover true locked the composer.
      busyRef.current = false
      setBusy(false)
      setAwaitingResponse(false)
      clearNotifications()
      setSelectedStoredSessionId(storedSessionId)
      selectedStoredSessionIdRef.current = storedSessionId
      const runtimeStartedAt = Date.now()
      setSessionStartedAt(runtimeStartedAt)

      const stored =
        $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId)) ?? storedForProfile

      applyStoredSessionPreviewRuntimeInfo(stored, storedSessionId)

      if (stored) {
        applyStoredUsage(stored)
      }

      const displayRead = captureDisplayHydration({
        flights: coldDisplayReadsRef.current,
        key: JSON.stringify([sessionRestScope, storedSessionId]),
        runtimeIdByStoredSessionIdRef,
        sessionStateByRuntimeIdRef,
        stored,
        storedSessionId
      })

      let resumedRunning = false
      // A recovered in-flight tail means the turn already produced output, so
      // it resumes into the streaming state rather than the "awaiting first
      // token" spinner.
      let recoveredInFlightTail = false

      try {
        const watchWindow = isWatchWindow()

        let localSnapshot = resumedSameSelectedSession
          ? preserveLocalPendingTurnMessages(viewMessagesForReconcile(), resumeStartMessages)
          : viewMessagesForReconcile()

        let prefetchApplied = false
        let prefetchedStoredSessionId: string | null = null
        let prefetchedTranscriptMessages: ChatMessage[] | null = null

        // REST transcript prefetch and the gateway resume RPC are independent
        // — run them concurrently so a big session's wall time is
        // max(prefetch, resume) instead of their sum. The prefetch paints the
        // transcript as soon as it lands; the RPC binds the runtime id.
        // Watch windows skip the prefetch — lazy resume attaches the live mirror.
        const prefetchPromise = watchWindow ? null : getLatestSessionMessages(storedSessionId, sessionRestScope)

        let resumeRuntimeBaselineMessages: ChatMessage[] = []
        const resumeStartedAt = Date.now() / 1000

        const resumePromise = singleFlightSessionResume(storedSessionId, () =>
          requestForSession<SessionResumeResult>('session.resume', {
            session_id: storedSessionId,
            cols: 96,
            source: 'desktop',
            defer_history: !watchWindow,
            // REST is the transcript authority for Desktop. Avoid duplicating a
            // potentially huge compression lineage in the WebSocket response.
            // Watch windows attach lazily (live mirror). Every other cold resume
            // gets the gateway's default deferred build: the RPC returns the
            // transcript immediately instead of blocking the switch on _make_agent
            // (MCP discovery / prompt build), and the agent pre-warms in the
            // background while the prefetch above paints the transcript.
            ...(watchWindow ? { lazy: true } : { omit_messages: true }),
            ...(sessionProfile ? { profile: sessionProfile } : {})
          })
        ).then(resumed => {
          resumeRuntimeBaselineMessages =
            sessionStateByRuntimeIdRef.current.get(resumed.session_id)?.messages ?? resumeRuntimeBaselineMessages

          return resumed
        })

        // The rejection is consumed by the `await` below; this guard only
        // keeps it from surfacing as unhandled while the prefetch settles.
        resumePromise.catch(() => undefined)

        let prefetchedResult: SessionMessagesResponse | null = null

        try {
          if (prefetchPromise) {
            prefetchedResult = await prefetchPromise
          }
        } catch {
          // Non-fatal: gateway resume below can still hydrate the session.
        }

        // The socket can drop and redial while REST is in flight. Painting now
        // would let the new socket's replay append the same turn again; a lost
        // socket (false) drops this read and the resume below binds without it.
        const viewReplayAtReturn = viewRuntimeId ? pendingSessionReplay(viewRuntimeId) : undefined

        if (prefetchedResult && viewReplayAtReturn && !(await viewReplayAtReturn)) {
          prefetchedResult = null
        }

        // A completed read still warms its exact durable scope after navigation.
        // It must not adopt a runtime or touch the foreground on that path.
        if (
          prefetchedResult &&
          stored &&
          displayRead.owns() &&
          (!prefetchedResult.session_id || prefetchedResult.session_id === stored.id)
        ) {
          saveTranscriptTail(storedSessionId, toChatMessages(prefetchedResult.messages), sessionRestScope)
        }

        // Paint the persisted transcript as soon as REST returns instead of
        // holding it until the runtime resume settles. A cold profile build
        // (skills, MCP, memory) can keep `session.resume` pending far longer
        // than the hydration budget while the complete history is already in
        // hand — holding it stranded Bot Chats on the loader (#90130). The
        // runtime path below grafts only its live projection onto this same
        // snapshot, so an unchanged acknowledgement keeps reference identity
        // and never rebuilds the transcript a second time.
        if (prefetchedResult && isCurrentResume()) {
          const previousMessages = resumedSameSelectedSession
            ? preserveLocalPendingTurnMessages(viewMessagesForReconcile(), resumeStartMessages)
            : viewMessagesForReconcile()

          // Tail page + previously backfilled prefix (same-session re-resume).
          // A long turn can push every rendered row off the newest page; read
          // older pages until they overlap so the graft keeps earlier history.
          const prefetchedTail = await extendRefreshPageToOverlap(
            toChatMessages(prefetchedResult.messages),
            previousMessages,
            olderPageReader(storedSessionId, sessionRestScope, prefetchedResult)
          )

          const graftedPrefetch = graftRefreshedTailOntoBackfill(prefetchedTail, previousMessages)

          prefetchedTranscriptMessages = graftedPrefetch
          localSnapshot = reconcileAuthoritativeChatMessages(graftedPrefetch, previousMessages)
          prefetchApplied = true
          prefetchedStoredSessionId = prefetchedResult.session_id || storedSessionId

          // The overlap reads await; skip painting if this resume went stale.
          if (isCurrentResume() && !chatMessageArraysEquivalent($messages.get(), localSnapshot)) {
            setMessages(localSnapshot)
          }
        }

        const resumed = await resumePromise

        if (!isCurrentResume()) {
          return
        }

        const currentMessages = viewMessagesForReconcile()

        // Keep the local snapshot when resume would only reshuffle runtime
        // projection. When the REST prefetch already hydrated the transcript,
        // skip converting/reconciling the resume payload entirely — on a
        // 1000+-message session that second conversion plus the deep
        // equivalence compare costs over a second of main-thread time.
        const resumedStoredSessionId = resumed.session_key || resumed.resumed

        const prefetchMatchesResumedSession =
          !prefetchedStoredSessionId || !resumedStoredSessionId || prefetchedStoredSessionId === resumedStoredSessionId

        const hasLiveProjection = Boolean(resumed.inflight || resumed.queued)

        const preferredMessages = (() => {
          if (prefetchApplied && prefetchMatchesResumedSession) {
            if (hasLiveProjection && prefetchedTranscriptMessages) {
              const runtimeMessages = toChatMessages(resumed.messages)
              const previousMessages = removeRepresentedLocalLiveProjection(currentMessages, resumed)

              // Omitted-messages resumes stay safe here: `resumed.messages`
              // is empty, so `runtimeMessages` has no anchor and the dedupe
              // helper returns the projection unchanged, while the REST
              // prefetch below remains the authoritative transcript — the
              // same "graft, don't rebuild" outcome the pre-restructure
              // messages_omitted branch produced.
              const liveProjection = dedupeInflightUserAgainstTranscript(
                prefetchedTranscriptMessages,
                runtimeMessages,
                resumed
              )

              const resumedMessages = reconcileAuthoritativeChatMessages(
                prefetchedTranscriptMessages,
                previousMessages,
                liveProjection,
                prefetchedResult?.messages
              )

              const withConcurrentChanges = overlayConcurrentMessageChanges(
                resumedMessages,
                localSnapshot,
                currentMessages
              )

              return chatMessageArraysEquivalent(currentMessages, withConcurrentChanges)
                ? currentMessages
                : withConcurrentChanges
            }

            if (!hasLiveProjection) {
              return localSnapshot
            }
          }

          const previousMessages = resumedSameSelectedSession
            ? preserveLocalPendingTurnMessages(currentMessages, resumeStartMessages)
            : currentMessages

          const resumedMessages = reconcileAuthoritativeMessages(resumed.messages, previousMessages, resumed)

          return chatMessageArraysEquivalent(currentMessages, resumedMessages) ? currentMessages : resumedMessages
        })()

        const currentRuntimeMessages =
          sessionStateByRuntimeIdRef.current.get(resumed.session_id)?.messages ?? resumeRuntimeBaselineMessages

        const preferredWithRuntimeChanges = overlayConcurrentMessageChanges(
          preferredMessages,
          resumeRuntimeBaselineMessages,
          currentRuntimeMessages
        )

        // #70449: same stale-snapshot guard as the warm path — a turn that
        // started while the resume RPC was in flight has already marked the
        // rebound runtime busy via gateway events; the snapshot must not
        // rewind it to idle just because the user opened the chat.
        resumedRunning = resolveResumedBusy(
          (resumed as { running?: boolean }).running,
          Boolean(sessionStateByRuntimeIdRef.current.get(resumed.session_id)?.busy)
        )

        restoreSessionTodosFromSnapshot(resumed.session_id, resumed.todo_state, resumedRunning)

        if (!resumedRunning && prefetchApplied && prefetchMatchesResumedSession && prefetchedTranscriptMessages) {
          restoreSessionTodosFromSnapshot(
            resumed.session_id,
            latestSessionTodoSnapshot(prefetchedTranscriptMessages),
            false
          )
        }

        // Crash-survivable turn progress: fold a journaled in-flight tail
        // (persisted by use-session-state-cache while the turn streamed;
        // survives renderer/app death) back onto the restored transcript. The
        // backend's own inflight projection is already inside
        // `preferredWithRuntimeChanges`, so this merge only adds the locally
        // recorded structure that the backend's text-only snapshot cannot carry.
        const inFlightRecovery = recoverInFlightTurnJournal(storedSessionId, preferredWithRuntimeChanges, {
          keepPending: resumedRunning
        })

        recoveredInFlightTail = inFlightRecovery.applied

        // Prefetch-hit fast path: reuse the live array when neither runtime
        // changes nor in-flight recovery changed the reconciled transcript.
        const messagesForView =
          inFlightRecovery.messages === currentMessages
            ? currentMessages
            : preserveLocalAssistantErrors(inFlightRecovery.messages, currentMessages)

        // Fail-latch on the PRE-recovery transcript: an orphan journal tail
        // must not mask a lost transcript (a retry that reloads real history
        // is safer than surfacing the in-flight turn alone). Recovery only
        // ever appends, so this matches the final transcript's emptiness.
        //
        // "Should have a transcript" is not the cached sessions-list row alone.
        // That row is a cache of backend truth and lags the two flows that
        // report a vanished thread: after a wake/reconnect the list can still
        // carry the respawned backend's session at message_count 0, and a
        // compression tip can show 0 rows while the stored transcript is
        // intact. Conditioning the latch on it alone paints a blank thread
        // UNLATCHED — no retry, no error, just an empty chat that looks like
        // lost history. The resume RPC is authoritative and always reports the
        // stored size (`message_count`, filled from state.db even when
        // `messages_omitted`), so treat it — and a non-empty REST page — as the
        // other rungs of the same ladder.
        const saidToHaveTranscript =
          sessionShouldHaveTranscript(stored) ||
          (resumed.message_count || 0) > 0 ||
          Boolean(prefetchedResult?.messages.length)

        if (saidToHaveTranscript && preferredMessages.length === 0) {
          // Roll back a provisional cached-tail paint and drop its entry: the
          // latched attempt painted no history from any source, so the
          // display-only cache must not survive to mask the retry (or the next
          // wake) as a transcript that loaded.
          if (cachedTailPaint !== null && $messages.get() === cachedTailPaint) {
            setMessages([])
            dropTranscriptTail(storedSessionId, sessionRestScope)
          }

          setActiveSessionId(null)
          activeSessionIdRef.current = null
          setResumeFailedSessionId(storedSessionId)
          resumedRunning = false

          return
        }

        setActiveSessionId(resumed.session_id)
        activeSessionIdRef.current = resumed.session_id
        // A live resume proves the owner routed — retire any read-only latch
        // a previous no-owner open left behind (#94724: the backfill stamped
        // the row, or a topology change made the owner resolvable again).
        clearStoredTranscriptReadOnly(storedSessionId)
        const pendingApproval = restorePendingApproval(resumed, resumed.session_id)
        const pendingClarifyState = restorePendingClarifyFromSnapshot(resumed, resumed.session_id, resumeStartedAt)
        const pendingClarify = pendingClarifyState.request

        const pendingConnection = restorePendingConnectionFromSnapshot(
          resumed,
          resumed.session_id,
          resumeStartedAt
        ).request

        const clarifyAuthoritativelyAbsent =
          pendingClarifyState.authoritativeAbsent && !$clarifyRequests.get()[resumed.session_id]

        const runtimeInfo = applyRuntimeInfo(resumed.info)

        patchSessionWorkspace(storedSessionId, runtimeInfo?.cwd)

        // Preserve the turn-elapsed timer across cold resume: the gateway
        // reports when the in-flight turn started so the desktop can restore
        // the clock instead of resetting it to 0:00.
        const resumedTurnStartedAt =
          typeof resumed.turn_started_at === 'number' && resumed.turn_started_at > 0
            ? resumed.turn_started_at * 1000
            : null

        const pendingClarifyProjection = pendingClarify
          ? restorePendingClarifyToolCall(messagesForView, pendingClarifyToolPayload(pendingClarify))
          : null

        const clearedClarifyProjection = clarifyAuthoritativelyAbsent
          ? settlePendingClarifyToolCall(
              messagesForView,
              pendingClarifyState.cleared ? pendingClarifyToolPayload(pendingClarifyState.cleared) : {},
              resumedRunning
            )
          : null

        const pendingConnectionProjection = projectPendingConnection(
          pendingClarifyProjection?.messages ?? clearedClarifyProjection?.messages ?? messagesForView,
          pendingConnection
        )

        const visibleMessagesForView =
          pendingConnectionProjection?.messages ??
          pendingClarifyProjection?.messages ??
          clearedClarifyProjection?.messages ??
          messagesForView

        // The eagerly painted REST page is persisted-display authority: stamp
        // its provenance so the next warm switch to this session paints it
        // immediately instead of holding it as an unproven runtime tail.
        const transcriptProvenance =
          prefetchApplied && prefetchMatchesResumedSession && stored
            ? createPersistedDisplayTranscriptProvenance({
                lineageRootId: stored._lineage_root_id ?? null,
                scope: sessionRestScope,
                storedSessionId
              })
            : undefined

        updateSessionState(
          resumed.session_id,
          state => ({
            // The deferred build reports the session's own effort later (#79807).
            ...markReasoningEffortPending({ ...state, ...(runtimeInfo ?? {}) }),
            runtimeStartedAt,
            messages: visibleMessagesForView,
            transcriptProvenance,
            busy: resumedRunning,
            awaitingResponse: resumedRunning && !recoveredInFlightTail,
            // Backend reported this turn running at resume time — live proof.
            turnLive: state.turnLive || resumedRunning,
            needsInput:
              pendingApproval ||
              Boolean(pendingClarify) ||
              Boolean(pendingConnection) ||
              (clarifyAuthoritativelyAbsent ? false : state.needsInput),
            adoptedRunningTurn: state.adoptedRunningTurn || resumedRunning,
            ...(inFlightRecovery.applied
              ? {
                  sawAssistantPayload: true,
                  // Point live deltas at the recovered row when the backend is
                  // still mid-turn; a settled recovery keeps the stream idle.
                  streamId: resumedRunning ? inFlightRecovery.streamId : null,
                  turnStartedAt: resumedRunning ? (inFlightRecovery.turnStartedAt ?? resumedTurnStartedAt) : null
                }
              : {
                  turnStartedAt: resumedRunning && resumedTurnStartedAt !== null ? resumedTurnStartedAt : null
                }),
            ...livePromptStreamId(pendingConnectionProjection, pendingClarifyProjection),
            ...(clearedClarifyProjection
              ? {
                  streamId: resumedRunning ? (clearedClarifyProjection.streamId ?? state.streamId) : null
                }
              : {})
          }),
          storedSessionId
        )

        // updateSessionState stages its view sync through requestAnimationFrame.
        // Commit the final, already-reconciled transcript now so resume has one
        // additive DOM build instead of an eager prefetch build plus a later
        // runtime projection build.
        if (!chatMessageArraysEquivalent($messages.get(), visibleMessagesForView)) {
          setMessages(visibleMessagesForView)
        }

        // Refresh the durable tail cache with backend transcript truth only;
        // the live pending clarify projection expires with the server request.
        saveTranscriptTail(
          storedSessionId,
          stripPendingClarifyProjectionForCache(
            messagesForView,
            pendingClarify?.requestId ??
              pendingClarifyState.cleared?.requestId ??
              $clarifyRequests.get()[resumed.session_id]?.requestId
          ),
          sessionRestScope
        )
      } catch (err) {
        if (!isCurrentResume()) {
          return
        }

        // The gateway resume RPC failed. Try the REST transcript as a fallback
        // so the window at least shows history. CRITICAL: this fallback must be
        // wrapped in its own try — if it ALSO throws (wedged/unreachable backend,
        // the common case when resume failed in the first place), an unguarded
        // throw here skips setMessages AND leaves activeSessionId null with an
        // empty transcript. That is the exact state the thread loader latches on
        // forever (messagesEmpty && !activeSessionId) with no recovery path —
        // the "open in new window stays stuck loading, even after a nap" bug.
        let fallbackError: unknown = null

        try {
          const fallback = await getLatestSessionMessages(storedSessionId, sessionRestScope)

          if (!isCurrentResume()) {
            return
          }

          const previousMessages = resumedSameSelectedSession
            ? preserveLocalPendingTurnMessages(viewMessagesForReconcile(), resumeStartMessages)
            : viewMessagesForReconcile()

          // Resume failed, so there is no live projection — the journal is the
          // only carrier of a crashed turn's progress on this path.
          const fallbackRecovery = recoverInFlightTurnJournal(
            storedSessionId,
            reconcileAuthoritativeMessages(fallback.messages, previousMessages)
          )

          // The eager prefetch paint above may already show this transcript.
          if (!chatMessageArraysEquivalent($messages.get(), fallbackRecovery.messages)) {
            setMessages(fallbackRecovery.messages)
          }
        } catch (e) {
          // Fallback also failed: nothing to paint. Leave whatever messages are
          // already shown and fall through to arm the resume-failure latch so
          // use-route-resume re-attempts the resume on the next render / window
          // focus / gateway reconnect instead of stranding the loader.
          fallbackError = e
        }

        if (!isCurrentResume()) {
          return
        }

        // #94724 no-owner recovery: the owner ladder failed closed — which is
        // CORRECT under registry topology — but the stored transcript may be
        // fully intact in some backend's state.db. If the ambient REST
        // fallback above didn't already paint it, probe the registered
        // backends READ-ONLY (id-only GET; no live session is routed or
        // minted anywhere). When history is reachable, open the session
        // read-only instead of dead-ending on the resolution error: writes
        // stay blocked, and a later resume (after the single-match owner
        // backfill stamps the row) upgrades it back to a live session.
        if (isSessionOwnerResolutionError(err)) {
          let painted = !fallbackError && viewMessagesForReconcile().length > 0

          if (!painted) {
            const stored = await fetchStoredTranscriptAcrossBackends(storedSessionId).catch(() => null)

            if (!isCurrentResume()) {
              return
            }

            if (stored && stored.messages.length > 0) {
              const previousMessages = resumedSameSelectedSession
                ? preserveLocalPendingTurnMessages(viewMessagesForReconcile(), resumeStartMessages)
                : viewMessagesForReconcile()

              setMessages(reconcileAuthoritativeMessages(stored.messages, previousMessages))
              painted = true
            }
          }

          if (painted) {
            markStoredTranscriptReadOnly(storedSessionId)
            notify({
              kind: 'info',
              title: copy.readOnlyTranscriptTitle,
              message: copy.readOnlyTranscriptBody
            })

            return
          }
        }

        // The session is genuinely gone (deleted, or a stale id from a wiped /
        // rotated backend): the resume RPC and the authoritative REST transcript
        // both 404. There's nothing to recover — silently drop to a fresh draft
        // instead of toasting an error and hot-looping the bounded retry on a
        // permanently-dead id. (Booting straight into a no-longer-existent
        // last-session id is the common trigger.)
        if (viewMessagesForReconcile().length === 0 && isSessionGoneError(fallbackError)) {
          // A 404 is only trustworthy from the backend that OWNS the session.
          // A cross-profile open (Bots pane) races the gateway swap, so both
          // lookups can land on a backend that never heard of the id (#88540).
          // Re-resolve before discarding: a row still listed on any profile —
          // or a swap in flight — means "retry once things settle", not gone.
          let stillListed = false

          try {
            stillListed = Boolean(await resolveStoredSession(storedSessionId))
          } catch {
            // Resolution itself failed — inconclusive, treat as not listed.
          }

          if (!isCurrentResume()) {
            return
          }

          const verdict = goneSessionVerdict({
            createdThisRun: sessionCreatedThisRun(storedSessionId),
            stillListed,
            switchInFlight:
              $gatewaySwitching.get() ||
              Boolean($gatewaySwapTarget.get()) ||
              // Known owner ≠ active gateway: the 404 came from the wrong
              // backend. An UNKNOWN owner must not count — it would block the
              // draft fallback for genuinely dead ids on secondary profiles.
              Boolean(
                sessionProfile?.trim() &&
                normalizeProfileKey(sessionProfile) !== normalizeProfileKey($activeGatewayProfile.get())
              )
          })

          if (verdict === 'retry') {
            setResumeFailedSessionId(storedSessionId)

            return
          }

          // The id is verifiably dead, but the text the user typed into it is
          // still stashed under that key (#111868). Announce it so the
          // composer's swap onto the fresh draft carries it over with an
          // inline, undoable notice instead of leaving it stranded.
          announceGoneSessionDraft(storedSessionId)
          startFreshSessionDraft(true)

          return
        }

        if (viewMessagesForReconcile().length === 0) {
          // Arm the self-heal ONLY when the window is still empty: the gateway
          // resume rejected AND the REST fallback failed to paint a transcript.
          // A durable cached-tail paint counts as EMPTY here — it's provisional
          // display, not proof of a live transcript, and must not mask the
          // stranded state from the retry machinery.
          // That is the exact stranded state the loader latches on
          // (messagesEmpty && !activeSessionId), and matches $resumeFailedSessionId's
          // documented contract. If the REST fallback DID paint history, the
          // window is readable — arming here would needlessly auto-retry and,
          // once retries exhaust, blank that visible transcript behind the
          // exhausted-state error overlay (a regression vs. plain fallback success).
          setResumeFailedSessionId(storedSessionId)
        }

        notifyError(err, copy.resumeFailed)
      } finally {
        displayRead.release()

        if (isCurrentResume()) {
          busyRef.current = resumedRunning
          setBusy(resumedRunning)
          setAwaitingResponse(resumedRunning && !recoveredInFlightTail)
        }
      }
    },
    [
      activeSessionIdRef,
      busyRef,
      copy,
      getRouteToken,
      holdSessionTranscriptView,
      requestGateway,
      resetViewSync,
      runtimeIdByStoredSessionIdRef,
      selectedStoredSessionIdRef,
      sessionStateByRuntimeIdRef,
      startFreshSessionDraft,
      syncSessionStateToView,
      updateSessionState
    ]
  )

  // Shared fork: create a child session seeded with `branchMessages`, linked to
  // `parentStoredId` so it nests under its parent, then open it as its own tab
  // and switch to it — the parent chat stays put (mirrors openNewSessionTile).
  // `idempotencyKey` lets a caller-driven retry reuse the SAME key so the
  // backend can dedupe (without it, every call generates a fresh key and a
  // response-lost retry would spawn a duplicate child).
  const forkBranch = useCallback(
    async (
      branchMessages: BranchMessage[],
      sourceSessionId: null | string,
      parentStoredId: null | string,
      cwd?: string,
      profile?: null | string,
      branchCount?: number,
      ownerRoute?: SessionOwnerRoute,
      idempotencyKey?: string
    ): Promise<boolean> => {
      creatingSessionRef.current = true

      // Stable per-attempt key so a backend retry after a lost response returns
      // the SAME child session instead of spawning a duplicate. Generated here
      // for first-time calls; supplied by the retry action on subsequent tries.
      const key = idempotencyKey ?? `branch-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`

      try {
        // A branch belongs to its parent's OWNING backend. Two facets, and both
        // matter once more than one connection is configured:
        //
        // 1. PROFILE — passing `profile` on the create mirrors
        //    desktopSessionCreateParams/resumeSession: in app-global remote mode
        //    one backend serves every profile, so an omitted profile silently
        //    lands the branch on the launch (default) profile — the "session
        //    jumps between profiles after branching" bug.
        // 2. CONNECTION — a profile name alone does not identify a backend when
        //    several connections expose the same name. Routing on profile only
        //    sends session.create to whatever socket happens to be active, so
        //    branching a remote-owned parent from another connection creates the
        //    child on the wrong backend (or nowhere), while the optimistic
        //    sidebar row below still points at an id no backend owns — the
        //    "Couldn't load this session" strand. removeSession already routes
        //    by (connection, profile); this is the same ownership contract.
        //
        // An untagged parent keeps the historic profile-only path exactly.
        if (ownerRoute) {
          await ensureGatewayAgent(ownerRoute.connectionId, ownerRoute.profile)
        } else {
          await ensureGatewayProfile(profile)
        }

        const requestBranchGateway = <T>(method: string, params: Record<string, unknown>): Promise<T> =>
          ownerRoute
            ? requestGatewayForAgent<T>(ownerRoute.connectionId, ownerRoute.profile, method, params)
            : requestGateway<T>(method, params)

        // The owner is part of the identity: the same parent id on two
        // connections is two different sessions, so a route-blind key would
        // coalesce them onto one create.
        const createKey = branchCreateKey({
          branchCount,
          branchMessages,
          cwd,
          ownerRoute,
          parentStoredId,
          profile,
          sourceSessionId
        })

        let createFlight = branchCreateFlightsRef.current.get(createKey)

        // No title: the backend auto-names the branch from its parent's lineage.
        if (!createFlight) {
          const branchParams = {
            session_id: sourceSessionId,
            // Stable per-attempt key: a lost-response retry of session.branch /
            // session.branch_whole returns the SAME child (#65410).
            idempotency_key: key,
            ...(branchCount !== undefined ? { count: branchCount } : {})
          }

          const createParams = {
            cols: 96,
            source: 'desktop',
            ...(cwd && { cwd }),
            ...(profile ? { profile } : {}),
            ...(parentStoredId && { parent_session_id: parentStoredId }),
            // Stable per-attempt key: a backend retry after a lost response
            // returns the SAME child instead of spawning a duplicate (#65410).
            idempotency_key: key
          }

          createFlight = (
            sourceSessionId
              ? requestBranchGateway<SessionCreateResponse>(
                  branchCount === undefined ? 'session.branch_whole' : 'session.branch',
                  branchParams
                ).catch(err => {
                  if (!isMissingRpcMethod(err)) {
                    throw err
                  }

                  return requestBranchGateway<SessionCreateResponse>('session.branch', branchParams)
                })
              : branchMessages.length
                ? requestBranchGateway<SessionCreateResponse>('session.create', {
                    ...createParams,
                    messages: branchMessages.map(({ content, role }) => ({ content, role }))
                  })
                : requestBranchGateway<SessionCreateResponse>('session.branch_stored', createParams).catch(
                    async err => {
                      if (!isMissingRpcMethod(err)) {
                        throw err
                      }

                      const { messages } = await getAllSessionMessages(parentStoredId ?? '', ownerRoute ?? profile)

                      if (!messages.length) {
                        throw new Error('nothing to branch — send a message first')
                      }

                      return requestBranchGateway<SessionCreateResponse>('session.create', {
                        ...createParams,
                        messages: messages.map(({ content, role }) => ({ content, role }))
                      })
                    }
                  )
          ).catch(err => {
            // Drop the flight so a genuine retry re-issues the create; a
            // resolved flight is cleared once the child is fully published.
            branchCreateFlightsRef.current.delete(createKey)
            throw err
          })
          branchCreateFlightsRef.current.set(createKey, createFlight)
        }

        const branched = await createFlight

        const responseBranchMessages =
          sourceSessionId && branched.messages?.length ? toBranchMessages(toChatMessages(branched.messages)) : []

        const effectiveBranchMessages = responseBranchMessages.length ? responseBranchMessages : branchMessages
        const routedSessionId = branched.stored_session_id ?? branched.session_id
        const runtimeStartedAt = Date.now()
        const preview = effectiveBranchMessages.map(({ content }) => content).find(Boolean) ?? null

        // Record the exact owner and pin its socket THE MOMENT the create
        // returns, before the optimistic row / tile publication can lose a
        // race with the gateway pruner. A draft branch child exists only as a
        // runtime on the owning backend (the stored row lands on first turn),
        // so a prune in this gap orphan-reaps it and the tile enters the
        // resume→reclaim flicker loop (#93892 shape). Mirrors the two routed
        // creates at the top of this file.
        if (ownerRoute) {
          setSessionOwnerHint(routedSessionId, ownerRoute)
          holdSessionOwnerUntilForeground(routedSessionId, ownerRoute)
        }

        // Draft until submit: nest under the parent at the parent's recency so it
        // doesn't bubble to the top until a real message lands (backend persists
        // + auto-names it then). The selected row survives refreshes (sessionsToKeep).
        const rows = $sessions.get()
        const parent = parentStoredId ? rows.find(session => sessionMatchesStoredId(session, parentStoredId)) : null

        const siblings = parentStoredId
          ? rows.filter(session => session.parent_session_id?.trim() === parentStoredId).length
          : 0

        setFreshDraftReady(false)
        // Stamp the optimistic row with the branch's EXACT owner. Without it the
        // row inherits $activeGatewayProfile and carries no connection_id, so a
        // child correctly created on the parent's remote backend is listed as
        // belonging to whichever backend happens to be active. Every later
        // owner lookup off that row (resume, hydrate, prompt) then routes to the
        // wrong machine and the chat pane spins on a session that backend never
        // had — the create is right, the row is a lie. Mirrors the routed
        // creates at the top of this file, which already pass their route here.
        upsertOptimisticSession(
          branched,
          routedSessionId,
          copy.branchTitle(siblings + 1).toLowerCase(),
          preview,
          parentStoredId,
          parent ? parent.last_active || parent.started_at : undefined,
          ownerRoute ?? null
        )
        ensureSessionState(branched.session_id, routedSessionId)
        updateSessionState(
          branched.session_id,
          state => ({
            ...state,
            runtimeStartedAt,
            messages: effectiveBranchMessages.map(({ source }) => source),
            busy: false,
            awaitingResponse: false
          }),
          routedSessionId
        )

        const runtimeInfo = applyRuntimeInfo(branched.info, { foreground: false })
        patchSessionWorkspace(routedSessionId, runtimeInfo?.cwd)

        if (runtimeInfo) {
          updateSessionState(branched.session_id, state => ({ ...state, ...runtimeInfo }), routedSessionId)
        }

        // Only take over the main pane when the chat being branched is the one
        // already open there — branching a background/sidebar session must
        // not yank the user's current view away from what they're looking at
        // (the #69750 focus-stealing bug, reintroduced if this fires
        // unconditionally). resumeSession reuses the runtime warm-cached above
        // (ensureSessionState/updateSessionState) instead of an extra resume RPC.
        if (parentStoredId !== null && selectedStoredSessionIdRef.current === parentStoredId) {
          navigate(sessionRoute(routedSessionId), { replace: true })
          await resumeSession(routedSessionId)
        } else {
          // Carry the exact owner onto the tile: its persisted ownerRoute is
          // what pins the owning backend's socket in the gateway keep-set
          // (openTileGatewayScopes) for the tile's whole lifetime. Without it
          // a remote-owned branch child's tile pinned nothing, the pruner
          // closed the owner socket, the backend reaped the draft runtime,
          // and the tile looped resume→reclaim until the storm breaker
          // latched "Couldn't open this session".
          openSessionTile(routedSessionId, 'center', undefined, null, {
            ownerRoute,
            workspaceMode: 'sessions'
          })
          patchSessionTile(routedSessionId, { runtimeId: branched.session_id })
          revealTreePane(`session-tile:${routedSessionId}`)
        }

        branchCreateFlightsRef.current.delete(createKey)
        broadcastSessionsChanged()

        return true
      } catch (err) {
        // Navigate throw or earlier failure after arming pending — never leave
        // creatingSessionRef stuck true.
        releaseCreatingSessionGuard()
        // Backend restart / WS drop mid-RPC leaves the branch uncreated with no
        // recovery path. Surface a persistent error with a retry action so the
        // user can re-attempt without re-doing the whole branch flow. The retry
        // passes the SAME idempotency key so the backend can dedupe if the
        // first create actually committed but its response was lost.
        notifyError(err, copy.branchFailed, {
          action: {
            label: t.common.retry,
            onClick: () => {
              void forkBranch(
                branchMessages,
                sourceSessionId,
                parentStoredId,
                cwd,
                profile,
                branchCount,
                ownerRoute,
                key
              )
            }
          }
        })

        return false
      } finally {
        if (!pendingCreatedStoredSessionIdRef.current) {
          creatingSessionRef.current = false
        }
      }
    },
    [
      copy,
      creatingSessionRef,
      ensureSessionState,
      navigate,
      requestGateway,
      resumeSession,
      selectedStoredSessionIdRef,
      t,
      updateSessionState
    ]
  )

  // Branch a session whose live transcript is already loaded in this renderer.
  // Both the main chat and session tiles use this path so a clicked message id
  // is resolved against the exact message array that rendered the action bar.
  const branchLoadedSession = useCallback(
    async ({
      busy,
      contextDrift,
      cwd,
      messageId,
      messages,
      runtimeId,
      storedSessionId
    }: BranchLoadedSessionOptions) => {
      if (!runtimeId) {
        notify({ kind: 'warning', title: copy.nothingToBranch, message: copy.branchNeedsChat })

        return false
      }

      if (busy) {
        notify({ kind: 'warning', title: copy.sessionBusy, message: copy.branchStopCurrent })

        return false
      }

      // Message-level branches still need the local message id to choose their
      // prefix. Whole-chat branches send only the parent identity below; the
      // backend reads the durable display projection without materializing it in
      // the renderer.
      let authoritativeMessages: ChatMessage[] | null = null
      const profile = await resolveSessionProfile(storedSessionId)

      // The open chat's exact owner, when its row carries a connection tag.
      // Same contract as branchStoredSession: the transcript read and the
      // branch RPC must both land on the backend that owns the parent, not on
      // whichever socket is active.
      const ownerRoute = storedSessionId ? sessionOwnerRouteFromRow(cachedSessionRow(storedSessionId)) : undefined

      if (messageId && storedSessionId) {
        try {
          const persisted = await getAllSessionMessages(storedSessionId, ownerRoute ?? profile)
          const hydrated = toChatMessages(persisted.messages)

          if (hydrated.length) {
            authoritativeMessages = hydrated
          }
        } catch {
          // The branch RPC has a backend-side display projection fallback.
        }
      }

      const drift = contextDrift?.()

      if (drift) {
        console.warn('[branch-drift-abort]', drift, {
          phase: 'transcript-hydration'
        })

        return false
      }

      const branchMessages = messageId ? selectBranchMessages(messages, authoritativeMessages, messageId) : []

      if (messageId && !branchMessages.length) {
        notify({ kind: 'warning', title: copy.nothingToBranch, message: copy.branchNoText })

        return false
      }

      clearNotifications()

      return forkBranch(
        branchMessages,
        runtimeId,
        storedSessionId,
        cwd?.trim(),
        profile,
        messageId ? branchMessages.length : undefined,
        ownerRoute
      )
    },
    [copy, forkBranch]
  )

  // Branch the open chat — optionally from a specific message — off its live transcript.
  const branchCurrentSession = useCallback(
    (messageId?: string): Promise<boolean> => {
      const runtimeId = activeSessionIdRef.current
      const storedSessionId = selectedStoredSessionIdRef.current
      const routeToken = getRouteToken()

      return branchLoadedSession({
        busy: busyRef.current,
        contextDrift: () => {
          const drift = sessionContextDrift({
            startRouteToken: routeToken,
            nowRouteToken: getRouteToken(),
            startSelectedStoredId: storedSessionId,
            nowSelectedStoredId: selectedStoredSessionIdRef.current
          })

          if (drift) {
            return drift
          }

          if (activeSessionIdRef.current !== runtimeId) {
            return 'runtime-changed'
          }

          return selectedStoredSessionIdRef.current === storedSessionId ? null : 'selection-changed'
        },
        cwd: $currentCwd.get(),
        messageId,
        messages: $messages.get(),
        runtimeId,
        storedSessionId
      })
    },
    [activeSessionIdRef, branchLoadedSession, busyRef, getRouteToken, selectedStoredSessionIdRef]
  )

  // Branch any listed session, not just the open one. Reads the target's stored
  // transcript directly (no resume/active-session dependency), so it works on
  // right-click and nests under its parent.
  const branchStoredSession = useCallback(
    async (storedSessionId: string, sessionProfile?: string | null): Promise<boolean> => {
      clearNotifications()

      // Right-clicking a session outside the paginated sidebar window is a cache
      // miss: resolve it (cache → active backend → cross-profile) so the branch
      // is created on the parent's OWNING profile, not whichever is live (#67603).
      // cachedSessionRow spans Recents, cron/messaging and the profile-scoped
      // project tree, and prefers the self-describing row — an ownerless legacy
      // Recents copy of the same id must not mask the row carrying the owner.
      const stored =
        cachedSessionRow(storedSessionId) ?? (sessionProfile ? undefined : await resolveStoredSession(storedSessionId))

      const profile = sessionProfile ?? stored?.profile

      // An exact owner from the parent row — connection AND profile. Undefined
      // for an untagged row, which keeps the ambient/profile-only path.
      const ownerRoute = sessionOwnerRouteFromRow(stored)

      try {
        if (ownerRoute) {
          await ensureGatewayAgent(ownerRoute.connectionId, ownerRoute.profile)
        } else {
          await ensureGatewayProfile(profile)
        }

        // Ask the owning backend to read and copy the parent transcript. The
        // renderer deliberately does not materialize the complete history.
        return await forkBranch(
          [],
          null,
          stored?.id ?? storedSessionId,
          stored?.cwd?.trim(),
          profile,
          undefined,
          ownerRoute
        )
      } catch (err) {
        notifyError(err, copy.branchFailed)

        return false
      }
    },
    [copy, forkBranch]
  )

  const removeSession = useCallback(
    async (storedSessionId: string) => {
      clearNotifications()

      // The row may live in the main list, the messaging/cron sidebar slices,
      // OR the archived view's own store (archived rows are excluded from
      // $sessions by design). Resolve from all of them so deleting a
      // messaging/cron row (or from the Archived filter) evicts the row
      // instead of leaving a ghost that resumes into a dead id.
      const listed = findListedSession(storedSessionId)

      const removed =
        listed?.session ?? $archivedSessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))

      // Messaging/cron rows frequently arrive without an inline profile; fall
      // back to the stored-session ownership lookup so their DELETE routes to
      // the owning profile instead of the ambient one.
      const stampedProfile = removed?.profile?.trim()
      const profile = stampedProfile || (await resolveSessionProfile(storedSessionId))

      // Listed profile-less row + multiple profiles + unresolved owner:
      // never fall through to the primary backend (fake already_absent).
      if (
        listed &&
        !stampedProfile &&
        !profile?.trim() &&
        $profiles.get().filter(item => item.name.trim()).length > 1
      ) {
        notifyError(new Error('Session ownership could not be resolved'), copy.deleteFailed)

        return
      }

      // Selection and runtime refs are updated synchronously at routing
      // boundaries. React props can still describe the previous render when a
      // delete lands in the same tick, which used to leave the doomed route in
      // place and let the generic 4001 recovery rebind it.
      const wasSelected = selectedStoredSessionIdRef.current === storedSessionId

      // Resolve the doomed session's live runtime from the SELECTION or the
      // stored→runtime map. Deleting a NON-selected (sidebar/background) session
      // used to skip this entirely, so its in-flight turn kept running and could
      // surface an approval/clarify prompt for a conversation that no longer
      // exists (#75587).
      const closingRuntimeId =
        (wasSelected ? activeSessionIdRef.current : null) ??
        runtimeIdByStoredSessionIdRef.current.get(storedSessionId) ??
        null

      const previousMessages = $messages.get()
      const previousPinned = $pinnedSessionIds.get()

      const removedOwner: SessionOwnerScope = removed?.connection_id
        ? {
            connectionId: removed.connection_id,
            profile: removed.profile || 'default'
          }
        : profile

      const previousArchived = $archivedSessions.get()
      // Pins are keyed on the durable lineage-root id; the stored id may be the
      // live tip after compression. Drop both so the pin can't linger.
      const removedPinId = removed ? sessionPinId(removed) : storedSessionId
      const removedIds = [storedSessionId, removed?.id, removed?._lineage_root_id]

      dropListedSession(storedSessionId)
      $archivedSessions.set(previousArchived.filter(session => !sessionMatchesStoredId(session, storedSessionId)))
      // Evict from the project tree's optimistic layer too (the backend snapshot
      // still lists it until its next refresh), so grouped + flat views drop the
      // row in lockstep. Pin the tombstone against the projects.tree prune while
      // the delete RPC is in flight, so a racing refresh can't flash it back.
      tombstoneSessions(removedIds)
      beginSessionMutation(removedIds)
      $pinnedSessionIds.set(previousPinned.filter(id => id !== storedSessionId && id !== removedPinId))

      // Tear down before awaiting so the route effect can't resume the
      // doomed session via the stale /<sid> URL.
      if (wasSelected) {
        startFreshSessionDraft(true)
      }

      try {
        if (closingRuntimeId) {
          // Deleting a session must END its turn, not just drop the row.
          // `session.close` tears down the runtime but does not walk the
          // interrupt path that releases approval / clarify / sudo / secret
          // waits, so a blocked run could outlive its sidebar row and surface a
          // blocking prompt (and native notification) for a conversation that is
          // gone (#75587). Mark the runtime interrupted first so a
          // blocking-input request already queued on the transport is dropped
          // instead of parking its overlay, then interrupt, then close.
          let previousInterruptState: Pick<ClientSessionState, 'interrupted' | 'needsInput'> | null = null

          updateSessionState(closingRuntimeId, state => {
            previousInterruptState = { interrupted: state.interrupted, needsInput: state.needsInput }

            return { ...state, interrupted: true, needsInput: false }
          })

          try {
            await requestForSessionProfile(removedOwner, requestGateway, 'session.interrupt', {
              session_id: closingRuntimeId
            })
          } catch (error) {
            // A missing runtime has no turn left to stop. Any other failure means
            // deletion cannot safely continue: restore the live state and let the
            // outer rollback put the conversation back in the sidebar.
            if (!isSessionGoneError(error)) {
              updateSessionState(closingRuntimeId, state =>
                previousInterruptState ? { ...state, ...previousInterruptState } : state
              )
              throw error
            }
          }

          // Catch a blocking-input request already queued before the interrupted
          // flag became visible to this renderer.
          clearAllPrompts(closingRuntimeId)
          clearClarifyRequest(undefined, closingRuntimeId)

          await requestForSessionProfile(removedOwner, requestGateway, 'session.close', {
            session_id: closingRuntimeId
          }).catch(() => undefined)
        }

        await deleteSession(storedSessionId, removedOwner)

        dropTranscriptTailEverywhere(storedSessionId)
        // Only after the RPC lands — the optimistic eviction above can roll
        // back, and a rolled-back row must keep its watermark/marker.
        forgetSessionUnread(removedIds, profile)
        clearQueuedPrompts(storedSessionId)
        // The journaled in-flight tail holds this session's prompt and tool
        // calls in localStorage; a deleted session must not leave that copy
        // behind to age out on its own. Purge after the RPC lands (same
        // rollback argument as the unread watermark above), passing every id
        // the delete holds: the stored tip, the row id, the lineage root, and
        // the closing runtime id — the journal keys on the stored id.
        purgeInFlightTurnJournals([...removedIds, closingRuntimeId])

        // Preview tabs are session-owned: drop them with the session (pinned
        // tabs survive — they belong to the workspace, not the session).
        for (const id of removedIds) {
          if (id) {
            prunePreviewTabsForSession(id)
          }
        }

        if (closingRuntimeId) {
          clearQueuedPrompts(closingRuntimeId)
          clearSessionControl(closingRuntimeId)
        }

        // A tiled copy of this session must not outlive it: collapse the pane
        // and evict its mirrored runtime state so nothing submits to (or renders)
        // a deleted session.
        const tiledRuntimeId = runtimeIdByStoredSessionIdRef.current.get(storedSessionId)
        closeSessionTile(storedSessionId)

        if (tiledRuntimeId) {
          runtimeIdByStoredSessionIdRef.current.delete(storedSessionId)
          sessionStateByRuntimeIdRef.current.delete(tiledRuntimeId)
          dropSessionState(tiledRuntimeId)
        }

        // Live per-session stores (the same four the stop paths clear) key on
        // the gateway event's session_id, i.e. the runtime id. When the deleted
        // row is selected, closingRuntimeId is the foreground runtime, which can
        // differ from the stored→runtime mapping (cached/tiled runtime), so
        // clear the stored id and both runtime ids — once per distinct id.
        for (const sid of new Set([storedSessionId, closingRuntimeId, tiledRuntimeId].filter(Boolean) as string[])) {
          clearSessionSubagents(sid)
          clearSessionTodos(sid)
          clearSessionGoal(sid)
          resetSessionBackground(sid)
        }
      } catch (err) {
        if (listed?.session) {
          restoreListedSession(listed.session, listed.slice)
        }

        // Restore the archived-view row too (no-op when it wasn't archived).
        $archivedSessions.set(previousArchived)

        untombstoneSessions(removedIds)
        $pinnedSessionIds.set(previousPinned)

        if (wasSelected) {
          setFreshDraftReady(false)
          setSelectedStoredSessionId(storedSessionId)
          selectedStoredSessionIdRef.current = storedSessionId
          const stored = findListedSession(storedSessionId)?.session

          if (stored) {
            applyStoredUsage(stored)
          }

          setMessages(previousMessages)
          navigate(sessionRoute(storedSessionId), { replace: true })

          if (closingRuntimeId) {
            setActiveSessionId(closingRuntimeId)
            activeSessionIdRef.current = closingRuntimeId
          }
        }

        notifyError(err, copy.deleteFailed)
      } finally {
        // Release the tombstone to the normal projects.tree prune now the RPC has
        // settled (kept on success — the backend has deleted it; cleared on the
        // rollback above on failure).
        endSessionMutation(removedIds)
      }
    },
    [
      activeSessionIdRef,
      copy,
      navigate,
      requestGateway,
      runtimeIdByStoredSessionIdRef,
      selectedStoredSessionIdRef,
      sessionStateByRuntimeIdRef,
      startFreshSessionDraft,
      updateSessionState
    ]
  )

  const archiveSession = useCallback(
    async (storedSessionId: string) => {
      clearNotifications()

      const listed = findListedSession(storedSessionId)
      const archived = listed?.session
      const stampedProfile = archived?.profile?.trim()
      const profile = stampedProfile || (await resolveSessionProfile(storedSessionId))

      if (
        listed &&
        !stampedProfile &&
        !profile?.trim() &&
        $profiles.get().filter(item => item.name.trim()).length > 1
      ) {
        notifyError(new Error('Session ownership could not be resolved'), copy.archiveFailed)

        return
      }

      const wasSelected = selectedStoredSessionIdRef.current === storedSessionId
      const previousPinned = $pinnedSessionIds.get()
      // Pins are keyed on the durable lineage-root id; the stored id may be the
      // live tip after compression. Drop both so the pin can't linger.
      const archivedPinId = archived ? sessionPinId(archived) : storedSessionId
      const archivedIds = [storedSessionId, archived?.id, archived?._lineage_root_id]

      // Soft-hide: drop from every sidebar slice immediately, keep the data.
      dropListedSession(storedSessionId)
      tombstoneSessions(archivedIds)
      beginSessionMutation(archivedIds)
      $pinnedSessionIds.set(previousPinned.filter(id => id !== storedSessionId && id !== archivedPinId))

      if (wasSelected) {
        startFreshSessionDraft(true)
      }

      try {
        await setSessionArchived(storedSessionId, true, profile)
        // Archived rows never reach the sidebar, so their persisted unread can
        // only rot. Dropped after the RPC so a failed archive keeps it.
        forgetSessionUnread(archivedIds, profile)
        // An archived session is hidden from the sidebar; its tile must go too.
        const tiledRuntimeId = runtimeIdByStoredSessionIdRef.current.get(storedSessionId)
        closeSessionTile(storedSessionId)

        if (tiledRuntimeId) {
          runtimeIdByStoredSessionIdRef.current.delete(storedSessionId)
          sessionStateByRuntimeIdRef.current.delete(tiledRuntimeId)
          dropSessionState(tiledRuntimeId)
        }

        notify({ durationMs: 2_000, kind: 'success', message: copy.archived })
      } catch (err) {
        if (archived) {
          restoreListedSession(archived, listed?.slice)
        }

        untombstoneSessions(archivedIds)
        $pinnedSessionIds.set(previousPinned)
        notifyError(err, copy.archiveFailed)
      } finally {
        endSessionMutation(archivedIds)
      }
    },
    [
      copy,
      runtimeIdByStoredSessionIdRef,
      selectedStoredSessionIdRef,
      sessionStateByRuntimeIdRef,
      startFreshSessionDraft
    ]
  )

  // The Archived view reuses the sidebar row menu; its already-archived rows
  // dispatch here through the same archive verb (#98813). Mirrors the Settings
  // → Archived Chats restore (sessions-settings.tsx): flip the persisted flag
  // back off, drop the archived-view row, and resurface the session in the
  // sidebar without waiting for a full refresh.
  const unarchiveSession = useCallback(
    async (storedSessionId: string) => {
      clearNotifications()

      const archived = $archivedSessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))
      const profile = archived?.profile?.trim() || undefined

      try {
        await setSessionArchived(storedSessionId, false, profile)

        // Drop the archived-view row first so the view reflects the restore
        // even when the session cannot be re-listed below (e.g. it belongs to
        // a profile the current query does not cover).
        $archivedSessions.set(
          $archivedSessions.get().filter(session => !sessionMatchesStoredId(session, storedSessionId))
        )

        if (archived) {
          // Lift any optimistic eviction so the grouped tree shows it again,
          // and re-list through the slice router so a messaging/cron row lands
          // back in its own list, not the sessions one.
          untombstoneSessions([storedSessionId, archived._lineage_root_id])
          restoreListedSession({ ...archived, archived: false })
        }

        notify({ durationMs: 2_000, kind: 'success', message: copy.restored })
      } catch (err) {
        notifyError(err, copy.unarchiveFailed)
      }
    },
    [copy]
  )

  return {
    archiveSession,
    branchCurrentSession,
    branchLoadedSession,
    branchStoredSession,
    closeSettings,
    createBackendSessionForSend,
    openNewSessionTile,
    openSettings,
    removeSession,
    resumeSession,
    selectSidebarItem,
    startFreshSessionDraft,
    submitTextToNewSession,
    unarchiveSession
  }
}
