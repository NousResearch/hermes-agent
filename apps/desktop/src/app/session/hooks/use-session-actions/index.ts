import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'
import type { NavigateFunction } from 'react-router'

import { NO_PROJECT_ID } from '@/app/chat/sidebar/projects/workspace-groups'
import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
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
  restorePendingClarifyToolCall,
  settlePendingClarifyToolCall,
  stripPendingClarifyProjectionForCache,
  toChatMessages
} from '@/lib/chat-messages'
import { isMissingRpcMethod } from '@/lib/gateway-rpc'
import { recoverInFlightTurnJournal } from '@/lib/inflight-turn-journal'
import { setSessionYolo } from '@/lib/yolo-session'
import { $clarifyRequests } from '@/store/clarify'
import { migrateSessionDraft } from '@/store/composer'
import { clearQueuedPrompts, migrateQueuedPrompts } from '@/store/composer-queue'
import {
  openGatewayForAgent,
  openGatewayForProfile,
  requestGatewayForAgent,
  retainGatewayForAgent
} from '@/store/gateway'
import { $gatewaySwitching } from '@/store/gateway-switch'
import { $pinnedSessionIds } from '@/store/layout'
import { clearNotifications, notify, notifyError } from '@/store/notifications'
import {
  $activeGatewayProfile,
  $gatewaySwapTarget,
  $newChatProfile,
  $profiles,
  $showAllProfiles,
  type AgentProfileRoute,
  ensureGatewayAgent,
  ensureGatewayProfile,
  normalizeProfileKey,
  resolveNewChatOwnerRoute
} from '@/store/profile'
import {
  $projectScope,
  beginSessionMutation,
  endSessionMutation,
  resolveNewSessionCwd,
  tombstoneSessions,
  untombstoneSessions
} from '@/store/projects'
import { setApprovalRequest } from '@/store/prompts'
import { clearStoredTranscriptReadOnly, markStoredTranscriptReadOnly } from '@/store/read-only-transcript'
import {
  $activeSessionStoredIdRotation,
  $connection,
  $currentCwd,
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $messages,
  $newChatWorkspaceTarget,
  $sessions,
  $yoloActive,
  getSessionOwnerHint,
  type NewChatWorkspaceTarget,
  resolveComposerSessionKey,
  sessionPinId,
  setActiveSessionId,
  setActiveSessionStoredIdRotation,
  setAwaitingResponse,
  setBusy,
  setCurrentBranch,
  setCurrentCwd,
  setCurrentCwdTransient,
  setCurrentServiceTier,
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
import { isSessionOwnerResolutionError } from '@/store/session-owner-resolution'
import {
  requestForSessionProfile,
  type SessionOwnerScope,
  type SessionProfileRoute
} from '@/store/session-request-router'
import {
  $sessionTiles,
  closeSessionTile,
  dropSessionState,
  holdSessionOwnerUntilForeground,
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
import { restoreSessionTodosFromSnapshot } from '@/store/todos'
import {
  dropTranscriptTail,
  dropTranscriptTailEverywhere,
  loadTranscriptTail,
  saveTranscriptTail
} from '@/store/transcript-tail-cache'
import { isWatchWindow } from '@/store/windows'
import type { SessionCreateResponse, SessionMessage, SessionResumeResponse, UsageStats } from '@/types/hermes'

import { navigateToWorkspacePage, NEW_CHAT_ROUTE, sessionRoute, SETTINGS_ROUTE } from '../../../routes'
import type { ClientSessionState, SidebarNavItem } from '../../../types'
import { sessionContextDrift } from '../session-context-drift'
import { singleFlightSessionResume } from '../use-prompt-actions/single-flight-resume'

import { pendingClarifyToolPayload, restorePendingClarifyFromSnapshot } from './restore-pending-clarify'
import {
  createPersistedDisplayTranscriptProvenance,
  hasPersistedDisplayTranscriptProvenance,
  suppressTranscriptForView,
  withoutTranscriptProvenance
} from './transcript-provenance'
import {
  appendLiveSessionProjection,
  applyRuntimeInfo,
  applyStoredSessionPreviewRuntimeInfo,
  type BranchMessage,
  chatMessageArraysEquivalent,
  dedupeInflightUserAgainstTranscript,
  dropListedSession,
  findListedSession,
  goneSessionVerdict,
  isSessionGoneError,
  overlayConcurrentMessageChanges,
  patchSessionWorkspace,
  preserveLocalPendingTurnMessages,
  reconcileResumeMessages,
  removeRepresentedLocalLiveProjection,
  resolveResumedBusy,
  resolveSessionProfile,
  resolveStoredSession,
  restoreListedSession,
  selectBranchMessages,
  sessionMatchesStoredId,
  sessionShouldHaveTranscript,
  toBranchMessages,
  upsertOptimisticSession
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

// Stored ids created in THIS renderer run. A brand-new session lives only in the
// gateway's in-memory map until its first turn persists a state.db row — so if a
// respawning/flapping backend drops it, both resume RPC and the REST transcript
// 404 even though the user just made it. We must NOT treat that as "gone" (which
// yanks them to a fresh draft — the "new sessions clear themselves" bug); the
// bounded retry rebinds it when the backend returns. Boot-into-a-stale-last-id
// (NOT in this set) still legitimately drops to a draft.
const createdThisRun = new Set<string>()

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
  liveProjection?: Pick<SessionResumeResponse, 'inflight' | 'queued' | 'session_id'>
): ChatMessage[] {
  const withLiveProjection = liveProjection
    ? appendLiveSessionProjection(authoritativeMessages, liveProjection)
    : authoritativeMessages

  const reconciled = reconcileResumeMessages(withLiveProjection, previousMessages)
  const withPendingTurn = preserveLocalPendingTurnMessages(reconciled, previousMessages)

  return preserveLocalAssistantErrors(withPendingTurn, previousMessages)
}

function reconcileAuthoritativeMessages(
  authoritativeMessages: SessionResumeResponse['messages'],
  previousMessages: ChatMessage[],
  liveProjection?: Pick<SessionResumeResponse, 'inflight' | 'queued' | 'session_id'>
): ChatMessage[] {
  return reconcileAuthoritativeChatMessages(toChatMessages(authoritativeMessages), previousMessages, liveProjection)
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
// profile to None). The sticky UI model/effort/fast ride as per-session overrides,
// never the profile default (that lives in Settings → Model).
async function desktopSessionCreateParams(
  cwd: string,
  capturedRoute = resolveNewChatOwnerRoute()
): Promise<Record<string, unknown>> {
  // Treat Send as the linearization point for the visible selector state. The
  // profile handshake below can yield long enough for background config/model
  // refreshes to finish; reading atoms afterward would silently create the
  // session with a different selection than the one the user submitted.
  const selection = {
    effort: $currentReasoningEffort.get().trim(),
    fast: $currentFastMode.get(),
    model: $currentModel.get().trim(),
    provider: $currentProvider.get().trim()
  }

  const profile = capturedRoute?.profile || $newChatProfile.get() || normalizeProfileKey($activeGatewayProfile.get())

  if (capturedRoute) {
    await ensureGatewayAgent(capturedRoute.connectionId, profile)
  } else {
    await ensureGatewayProfile(profile)
  }

  return {
    cols: 96,
    source: 'desktop',
    ...(cwd && { cwd }),
    ...(profile ? { profile: capturedRoute?.targetProfile || profile } : {}),
    ...(selection.model
      ? { model: selection.model, ...(selection.provider ? { provider: selection.provider } : {}) }
      : {}),
    ...(selection.effort ? { reasoning_effort: selection.effort } : {}),
    fast: selection.fast
  }
}

interface FreshSessionDraftOptions {
  preserveRoute?: boolean
  replaceRoute?: boolean
  workspaceTarget?: NewChatWorkspaceTarget
}

function restorePendingApproval(response: SessionResumeResponse, sessionId: string): boolean {
  const pending = response.pending_approval

  if (!pending) {
    return false
  }

  setApprovalRequest({
    allowPermanent: pending.allow_permanent !== false,
    choices: pending.choices,
    command: pending.command ?? '',
    description: pending.description ?? 'dangerous command',
    requestId: typeof pending.request_id === 'string' ? pending.request_id : undefined,
    sessionId,
    smartDenied: pending.smart_denied === true
  })

  return true
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

  // Follow auto-compression's stored-id rotation only while the exact runtime,
  // selection, and route intent still belong to the rotating conversation.
  // The previous implementation carried only the next stored id and navigated
  // unconditionally; a fast A → B → C switch could therefore be overwritten
  // by A's delayed session.info event and visibly jump back to A.
  const storedIdRotation = useStore($activeSessionStoredIdRotation)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (!storedIdRotation) {
      return
    }

    // Consume the event even when it is stale. Rotation is an edge, not durable
    // state; replaying it after a later remount/selection would steal focus.
    setActiveSessionStoredIdRotation(current => (current === storedIdRotation ? null : current))

    const selectedStoredSessionId = selectedStoredSessionIdRef.current
    const routedStoredSessionId = getRoutedStoredSessionId()

    if (
      activeSessionIdRef.current !== storedIdRotation.runtimeSessionId ||
      selectedStoredSessionId !== storedIdRotation.previousStoredSessionId ||
      (routedStoredSessionId !== null && routedStoredSessionId !== storedIdRotation.previousStoredSessionId)
    ) {
      return
    }

    // Park unsent draft/queue on the durable lineage key (not the new tip).
    // ChatBar scopes composer state on resolveComposerSessionKey(); migrating
    // onto the tip while the composer is still bound to the root can lose newer
    // live editor text on a brief remount. If the new tip row is not in
    // $sessions yet, resolveComposerSessionKey falls back to the tip id — prefer
    // the previous id (usually the lineage root) in that gap.
    const previousId = storedIdRotation.previousStoredSessionId
    const nextId = storedIdRotation.nextStoredSessionId
    const sessions = $sessions.get()
    const resolvedNext = resolveComposerSessionKey(nextId, sessions)

    const durableKey =
      resolvedNext && resolvedNext !== nextId
        ? resolvedNext
        : (resolveComposerSessionKey(previousId, sessions) ?? previousId)

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
  }, [activeSessionIdRef, getRoutedStoredSessionId, navigate, selectedStoredSessionIdRef, storedIdRotation])

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
      // refreshCurrentModel). Only $currentServiceTier (a live-session mirror)
      // is cleared.
      setCurrentServiceTier('')
      setYoloActive(false)
      setNewChatWorkspaceTarget(hasWorkspaceTarget ? workspaceTarget : undefined)

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
    async (preview: string | null = null): Promise<string | null> => {
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
        const params = await desktopSessionCreateParams(cwd, capturedRoute)

        // Lease the owner socket for the whole create → owner-publication
        // sequence (#93602 primitive). The per-request lease inside
        // requestGatewayForAgent ends when session.create returns; the
        // foreground hold below takes over from that point until the created
        // chat is selected. Between the two, nothing may close the socket
        // that just minted the runtime.
        const releaseCreateLease = capturedRoute
          ? await retainGatewayForAgent(capturedRoute.connectionId, capturedRoute.profile)
          : () => undefined

        let created: SessionCreateResponse
        let stored: null | string

        try {
          created = capturedRoute
            ? await requestGatewayForAgent<SessionCreateResponse>(
                capturedRoute.connectionId,
                capturedRoute.profile,
                'session.create',
                params
              )
            : await requestGateway<SessionCreateResponse>('session.create', params)

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
          createdThisRun.add(stored)
          // Seed the sidebar preview with the user's first message so the row
          // reads meaningfully while the turn is in flight, instead of flashing
          // "Untitled session" until the turn persists and auto-title runs. The
          // server later returns its own preview/title and supersedes this.
          // The row carries the create route's exact owner (backend profile +
          // connection), never the ambient profile — see upsertOptimisticSession.
          upsertOptimisticSession(created, stored, null, preview?.trim() || null, null, undefined, capturedRoute)
          navigate(sessionRoute(stored), { replace: true })
          // Other windows (e.g. the main window when this is the pop-out) can't
          // see this session until they re-pull the shared list.
          broadcastSessionsChanged()
        }

        setFreshDraftReady(false)
        setNewChatWorkspaceTarget(undefined)
        setActiveSessionId(created.session_id)
        setSelectedStoredSessionId(stored)
        setSessionStartedAt(Date.now())
        const yoloArmed = $yoloActive.get()
        const runtimeInfo = applyRuntimeInfo(created.info)

        if (runtimeInfo) {
          updateSessionState(created.session_id, state => ({ ...state, ...runtimeInfo }), stored)
        }

        // User may have armed YOLO on the new-chat draft before the runtime
        // session existed — apply it to the freshly created session.
        if (yoloArmed) {
          await setSessionYolo(requestGateway, created.session_id, true).catch(() => undefined)
        }

        return created.session_id
      } finally {
        window.setTimeout(() => {
          creatingSessionRef.current = false
        }, 0)
      }
    },
    [
      activeSessionIdRef,
      creatingSessionRef,
      ensureSessionState,
      getRouteToken,
      navigate,
      requestGateway,
      resetViewSync,
      selectedStoredSessionIdRef,
      updateSessionState
    ]
  )

  const selectSidebarItem = useCallback(
    (item: SidebarNavItem) => {
      if (item.action === 'new-session') {
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
        cwd?: null | string
        listed?: boolean
        route?: AgentProfileRoute | null
        workspaceScope?: SessionTileWorkspaceScope
      }
    ) => {
      const listed = options?.listed ?? true

      try {
        // Fresh tile → the caller's workspace when one was named (the sidebar
        // "+" on a project/worktree lane), explicit null means Home/detached,
        // else the resolved new-session cwd (project scope → configured default).
        // `options?.cwd || resolve…` is wrong for Home: null is falsy and used
        // to fall through into the last project folder while main chat was
        // occupied (openTab path for "New session in Home").
        const capturedRoute = options?.route === undefined ? resolveNewChatOwnerRoute() : options.route
        const workspaceScope = options?.workspaceScope ?? { workspaceMode: 'sessions' }

        const cwd =
          options?.cwd === null ? '' : typeof options?.cwd === 'string' ? options.cwd.trim() : resolveNewSessionCwd()

        const params = {
          ...(await desktopSessionCreateParams(cwd, capturedRoute)),
          ...(workspaceScope.workspaceMode === 'bots' ? { hidden: true } : {})
        }

        // Same lease chain as createBackendSessionForSend: owner socket held
        // across the create, then the foreground hold carries it until the
        // tile is mounted ($sessionTiles names the owner from then on).
        const releaseCreateLease = capturedRoute
          ? await retainGatewayForAgent(capturedRoute.connectionId, capturedRoute.profile)
          : () => undefined

        let created: SessionCreateResponse
        let stored: string | undefined

        try {
          created = capturedRoute
            ? await requestGatewayForAgent<SessionCreateResponse>(
                capturedRoute.connectionId,
                capturedRoute.profile,
                'session.create',
                params
              )
            : await requestGateway<SessionCreateResponse>('session.create', params)

          stored = created.stored_session_id

          if (stored && capturedRoute) {
            // Same ownership transition as createBackendSessionForSend: the
            // route that minted the session is its exact owner from this
            // moment on, and its socket stays pinned until the tile mounts.
            setSessionOwnerHint(stored, capturedRoute)
            holdSessionOwnerUntilForeground(stored, capturedRoute)
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

        createdThisRun.add(stored)

        // Seed the per-runtime cache so the tile renders immediately without a
        // redundant resume. Only add the row to the SIDEBAR when `listed` — an
        // unlisted (draft) tab stays out of the session list until its first
        // turn persists and a refresh surfaces it.
        if (listed) {
          upsertOptimisticSession(created, stored, null, null, null, undefined, capturedRoute)
        }

        // A tile lives in its OWN worktree, so it must not run the full
        // foreground composer publish. A CENTER tile is the focused surface,
        // though, and the Files pane still keys off the global `$currentCwd` —
        // so the right rail kept showing the previous session's tree when a
        // Project "+" created a session while the main chat was occupied
        // (#76696). Split/side tiles deliberately stay isolated.
        const runtimeInfo = applyRuntimeInfo(created.info, { foreground: false })
        updateSessionState(created.session_id, state => (runtimeInfo ? { ...state, ...runtimeInfo } : state), stored)

        openSessionTile(stored, dir, undefined, undefined, workspaceScope)
        patchSessionTile(stored, { runtimeId: created.session_id })

        if (dir === 'center' && runtimeInfo?.cwd) {
          setCurrentCwdTransient(runtimeInfo.cwd)
          setWorkspaceCwdOwner(stored)
        }

        revealTreePane(`session-tile:${stored}`)

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
      const requestId = resumeRequestRef.current + 1
      resumeRequestRef.current = requestId
      const resumedSameSelectedSession = selectedStoredSessionIdRef.current === storedSessionId
      const resumeStartMessages = resumedSameSelectedSession ? $messages.get() : []

      const isCurrentResume = () =>
        resumeRequestRef.current === requestId && selectedStoredSessionIdRef.current === storedSessionId

      // Paint the click before the profile-resolve / gateway-swap awaits below,
      // so there's zero dead air: highlight the row instantly (the sidebar reads
      // $selectedStoredSessionId) and, for a cold target, drop the previous
      // transcript so the thread shows its loader instead of the old session
      // lingering until resume lands. A warm-cached target keeps its transcript —
      // the cached fast-path repaints it this same tick. Setting the ref here is
      // also what use-route-resume's self-heal assumes ("set synchronously at
      // resume entry").
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

      if (!takeWarmCache()) {
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
      const ownerRoute = capturedOwner || getSessionOwnerHint(storedSessionId)
      // A connection switch clears/reloads the session rows before this path
      // runs, so an untagged row belongs to the connection that supplied the
      // current list. Capture that source before the async metadata lookup. If
      // we reduce it to the profile string `default`, requestForSessionProfile
      // resolves the local default socket and sends an SSH session id to the
      // wrong machine ("resume failed: session not found").
      const ambientConnection = $connection.get()

      const ambientConnectionId =
        ambientConnection?.mode === 'remote' ? ambientConnection.connectionId?.trim() || '' : ''

      const storedForProfile = await resolveStoredSession(storedSessionId, ownerRoute)
      const sessionProfile = storedForProfile?.profile

      if (resumeRequestRef.current !== requestId) {
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

      // All-profiles / plugin navigation must not steal chrome API-home:
      // dial the owning backend without moving $activeGatewayProfile.
      if ($showAllProfiles.get()) {
        if (resolvedConnectionId) {
          await openGatewayForAgent(resolvedConnectionId, ownerRoute?.profile || sessionProfile || 'default')
        } else if (sessionProfile) {
          await openGatewayForProfile(normalizeProfileKey(sessionProfile))
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

      const sessionRestScope = resolvedConnectionId
        ? {
            connectionId: resolvedConnectionId,
            profile: ownerRoute?.targetProfile || ownerRoute?.profile || sessionProfile || 'default'
          }
        : storedForProfile?.connection_id
          ? {
              connectionId: storedForProfile.connection_id,
              profile: sessionProfile || 'default'
            }
          : sessionProfile

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
          // wait on session.activate (#71254). Unproven cache entries (no
          // persisted-display provenance) stay off the view until REST
          // authority lands — a compressed runtime tail is legal in cache and
          // is exactly the session-switch flicker (#73646). Proven caches and
          // same-session re-resumes still paint immediately. The persisted
          // refresh itself still starts after activate reattaches the live
          // transport, so a turn finishing between snapshot and reattach
          // cannot leave a stale partial on screen.
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
          syncSessionStateToView(
            cachedRuntimeId,
            suppressTranscriptForView(cachedViewState, suppressUnprovenWarmTranscript)
          )
          setCurrentCwdTransient(cachedViewState.cwd)
          // The warm cache IS this conversation's own workspace truth, so the
          // switch is already re-homed here. This claim cannot wait for
          // `session.activate`: its missing-RPC compat branch returns before
          // `applyRuntimeInfo` runs, which would leave the workspace marked
          // un-owned for the life of the session (#71254).
          setWorkspaceCwdOwner(storedSessionId)
          setCurrentBranch(cachedViewState.branch)
          setSessionStartedAt(Date.now())

          try {
            let activated: SessionResumeResponse | null = null
            const activateStartedAt = Date.now() / 1000
            const activateBaselineState = sessionStateByRuntimeIdRef.current.get(cachedRuntimeId) ?? cachedViewState
            const clarifyRequestIdAtActivateStart = $clarifyRequests.get()[cachedRuntimeId]?.requestId

            try {
              activated = await requestForSession<SessionResumeResponse>('session.activate', {
                session_id: cachedRuntimeId,
                cols: 96,
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
                setCurrentUsage(current => ({ ...current, ...usage }))
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
                    (clarifyAuthoritativelyAbsent ? false : state.needsInput),
                  // Adopting someone else's turn: we'll stream its reply
                  // without ever having received its prompt, so the settle
                  // path must not take the "I saw it all" shortcut.
                  adoptedRunningTurn: state.adoptedRunningTurn || running,
                  turnStartedAt: running ? (activatedTurnStartedAt ?? state.turnStartedAt ?? Date.now()) : null
                }),
                storedSessionId
              )

              busyRef.current = running
              setBusy(running)
              setAwaitingResponse(running && !pendingClarify)
              syncSessionStateToView(
                cachedRuntimeId,
                suppressTranscriptForView(activatedLivenessState, suppressUnprovenWarmTranscript)
              )

              // session.activate is the ordering barrier for reconnect recovery:
              // it atomically rebinds a running turn before returning. If the
              // turn is already terminal, this post-barrier REST read sees its
              // durable final row; if it is still running, later deltas/finish
              // events arrive on the newly attached transport. Hydration below
              // reconciles only messages, so those events also retain liveness
              // authority while the request is pending.
              const persistedTranscriptPromise = shouldRefreshPersistedTranscript
                ? getLatestSessionMessages(storedSessionId, sessionRestScope).catch(() => null)
                : null

              // The persisted REST transcript is the display authority: a live
              // runtime may carry only the agent's compressed context projection,
              // which is intentionally smaller than the user-visible conversation.
              // Reconcile its in-flight/queued tail onto the complete transcript
              // instead of replacing durable history while the turn is running.
              let acceptedPersistedDisplayTranscript = false

              if (persistedTranscriptPromise) {
                const persisted = await persistedTranscriptPromise

                if (!isCurrentResume()) {
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
                  const persistedMessages = graftRefreshedTailOntoBackfill(
                    toChatMessages(persisted.messages),
                    cachedViewState.messages
                  )

                  const runtimeMessages = toChatMessages(activated.messages)
                  const previousMessages = removeRepresentedLocalLiveProjection(cachedViewState.messages, activated)

                  const liveProjection = dedupeInflightUserAgainstTranscript(
                    persistedMessages,
                    runtimeMessages,
                    activated
                  )

                  activatedMessages = reconcileAuthoritativeChatMessages(
                    persistedMessages,
                    previousMessages,
                    liveProjection
                  )
                }
              }

              const currentMessages = sessionStateByRuntimeIdRef.current.get(cachedRuntimeId)?.messages

              if (currentMessages) {
                activatedMessages = overlayConcurrentMessageChanges(
                  activatedMessages,
                  cachedViewState.messages,
                  currentMessages
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

              const visibleActivatedMessages =
                pendingClarifyProjection?.messages ?? clearedClarifyProjection?.messages ?? activatedMessages

              releaseTranscriptView()

              const activatedState = updateSessionState(
                cachedRuntimeId,
                state => ({
                  ...state,
                  messages: visibleActivatedMessages,
                  transcriptProvenance:
                    acceptedPersistedDisplayTranscript || hasValidProvenance
                      ? (expectedProvenance ?? undefined)
                      : undefined,
                  ...(pendingClarifyProjection
                    ? {
                        awaitingResponse: false,
                        sawAssistantPayload: true,
                        streamId: pendingClarifyProjection.streamId
                      }
                    : {}),
                  ...(clearedClarifyProjection
                    ? {
                        streamId: state.busy ? (clearedClarifyProjection.streamId ?? state.streamId) : null
                      }
                    : {})
                }),
                storedSessionId
              )

              syncSessionStateToView(cachedRuntimeId, activatedState)
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
      if (!resumedSameSelectedSession && $messages.get().length > 0) {
        setMessages([])
      }

      // Instant paint from the durable tail cache: a cold resume (fresh app
      // launch, reaped/respawned backend) otherwise shows a loader until the
      // REST prefetch lands — which on a cold multi-profile boot waits behind
      // a backend spawn. Painting the persisted tail here makes the wake
      // visually complete at ~0ms (and satisfies the paint-first hydration
      // wait). The paint is DISPLAY-ONLY: reconciliation below must treat the
      // view as empty (see cachedTailPaint), because grafting the REST tail
      // onto a stale cached tail would duplicate or misorder rows — the
      // authoritative transcript REPLACES the cached paint when it lands.
      // Same-selected re-resumes skip it — their transcript is already live.
      let cachedTailPaint: ChatMessage[] | null = null

      if (!resumedSameSelectedSession && $messages.get().length === 0) {
        const cachedTail = loadTranscriptTail(storedSessionId, sessionRestScope)

        if (cachedTail && selectedStoredSessionIdRef.current === storedSessionId) {
          cachedTailPaint = cachedTail
          setMessages(cachedTail)
        }
      }

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
      setSessionStartedAt(Date.now())

      const stored =
        $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId)) ?? storedForProfile

      applyStoredSessionPreviewRuntimeInfo(stored, storedSessionId)

      if (stored) {
        applyStoredUsage(stored)
      }

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
          requestForSession<SessionResumeResponse>('session.resume', {
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

        // Keep both requests concurrent, but do not paint the REST result until
        // the runtime resume has also settled. An eager prefetch paint followed
        // by the runtime projection rebuilds large transcripts during resume.
        let prefetchedResult: { messages: SessionMessage[]; session_id?: string } | null = null

        try {
          if (prefetchPromise) {
            prefetchedResult = await prefetchPromise
          }
        } catch {
          // Non-fatal: gateway resume below can still hydrate the session.
        }

        const resumed = await resumePromise

        if (!isCurrentResume()) {
          return
        }

        if (prefetchedResult) {
          const previousMessages = resumedSameSelectedSession
            ? preserveLocalPendingTurnMessages(viewMessagesForReconcile(), resumeStartMessages)
            : viewMessagesForReconcile()

          // Tail page + previously backfilled prefix (same-session re-resume).
          const graftedPrefetch = graftRefreshedTailOntoBackfill(
            toChatMessages(prefetchedResult.messages),
            previousMessages
          )

          prefetchedTranscriptMessages = graftedPrefetch
          localSnapshot = reconcileAuthoritativeChatMessages(graftedPrefetch, previousMessages)
          prefetchApplied = true
          prefetchedStoredSessionId = prefetchedResult.session_id || storedSessionId
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
                liveProjection
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
        if (sessionShouldHaveTranscript(stored) && preferredMessages.length === 0) {
          // Roll back a provisional cached-tail paint and drop its entry: the
          // authoritative sources say this session has no transcript, so the
          // cache no longer reflects backend truth and must not survive to
          // mislead the retry (or the next wake).
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

        const visibleMessagesForView =
          pendingClarifyProjection?.messages ?? clearedClarifyProjection?.messages ?? messagesForView

        updateSessionState(
          resumed.session_id,
          state => ({
            ...state,
            ...(runtimeInfo ?? {}),
            messages: visibleMessagesForView,
            busy: resumedRunning,
            awaitingResponse: resumedRunning && !recoveredInFlightTail,
            // Backend reported this turn running at resume time — live proof.
            turnLive: state.turnLive || resumedRunning,
            needsInput:
              pendingApproval || Boolean(pendingClarify) || (clarifyAuthoritativelyAbsent ? false : state.needsInput),
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
            ...(pendingClarifyProjection
              ? {
                  awaitingResponse: false,
                  sawAssistantPayload: true,
                  streamId: pendingClarifyProjection.streamId
                }
              : {}),
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

          setMessages(fallbackRecovery.messages)
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
            createdThisRun: createdThisRun.has(storedSessionId),
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
  const forkBranch = useCallback(
    async (
      branchMessages: BranchMessage[],
      sourceSessionId: null | string,
      parentStoredId: null | string,
      cwd?: string,
      profile?: null | string,
      branchCount?: number
    ): Promise<boolean> => {
      creatingSessionRef.current = true

      try {
        // A branch belongs to its parent's OWNING profile. Swapping the live
        // gateway first AND passing `profile` on the create mirrors
        // desktopSessionCreateParams/resumeSession: in app-global remote mode
        // one backend serves every profile, so an omitted profile silently
        // lands the branch on the launch (default) profile — the "session
        // jumps between profiles after branching" bug. The swap also makes
        // upsertOptimisticSession's $activeGatewayProfile stamp correct.
        await ensureGatewayProfile(profile)

        // No title: the backend auto-names the branch from its parent's lineage.
        const branched = sourceSessionId
          ? await requestGateway<SessionCreateResponse>('session.branch', {
              session_id: sourceSessionId,
              ...(branchCount !== undefined ? { count: branchCount } : {})
            })
          : await requestGateway<SessionCreateResponse>('session.create', {
              cols: 96,
              source: 'desktop',
              ...(cwd && { cwd }),
              ...(profile ? { profile } : {}),
              messages: branchMessages.map(({ content, role }) => ({ content, role })),
              ...(parentStoredId && { parent_session_id: parentStoredId })
            })

        const responseBranchMessages =
          sourceSessionId && branched.messages?.length ? toBranchMessages(toChatMessages(branched.messages)) : []

        const effectiveBranchMessages = responseBranchMessages.length ? responseBranchMessages : branchMessages
        const routedSessionId = branched.stored_session_id ?? branched.session_id
        const preview = effectiveBranchMessages.map(({ content }) => content).find(Boolean) ?? null
        // Draft until submit: nest under the parent at the parent's recency so it
        // doesn't bubble to the top until a real message lands (backend persists
        // + auto-names it then). The selected row survives refreshes (sessionsToKeep).
        const rows = $sessions.get()
        const parent = parentStoredId ? rows.find(session => sessionMatchesStoredId(session, parentStoredId)) : null

        const siblings = parentStoredId
          ? rows.filter(session => session.parent_session_id?.trim() === parentStoredId).length
          : 0

        setFreshDraftReady(false)
        upsertOptimisticSession(
          branched,
          routedSessionId,
          copy.branchTitle(siblings + 1).toLowerCase(),
          preview,
          parentStoredId,
          parent ? parent.last_active || parent.started_at : undefined
        )
        ensureSessionState(branched.session_id, routedSessionId)
        updateSessionState(
          branched.session_id,
          state => ({
            ...state,
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
          await resumeSession(routedSessionId)
        } else {
          openSessionTile(routedSessionId, 'center')
          patchSessionTile(routedSessionId, { runtimeId: branched.session_id })
          revealTreePane(`session-tile:${routedSessionId}`)
        }

        broadcastSessionsChanged()

        return true
      } catch (err) {
        notifyError(err, copy.branchFailed)

        return false
      } finally {
        window.setTimeout(() => {
          creatingSessionRef.current = false
        }, 0)
      }
    },
    [
      copy,
      creatingSessionRef,
      ensureSessionState,
      requestGateway,
      resumeSession,
      selectedStoredSessionIdRef,
      updateSessionState
    ]
  )

  // Branch the open chat — optionally from a specific message — off its live transcript.
  const branchCurrentSession = useCallback(
    async (messageId?: string): Promise<boolean> => {
      if (!activeSessionIdRef.current) {
        notify({ kind: 'warning', title: copy.nothingToBranch, message: copy.branchNeedsChat })

        return false
      }

      if (busyRef.current) {
        notify({ kind: 'warning', title: copy.sessionBusy, message: copy.branchStopCurrent })

        return false
      }

      const startingActiveSessionId = activeSessionIdRef.current
      const messages = $messages.get()
      const storedSessionId = selectedStoredSessionIdRef.current
      const startingRouteToken = getRouteToken()
      const startingCwd = $currentCwd.get().trim()

      // The live atom may be a compacted model projection. Read the durable
      // display projection before choosing the branch prefix so a whole-chat
      // branch does not inherit only the summary/tail. If the backend is
      // temporarily unavailable, retain the local snapshot and let the branch
      // RPC make its own authoritative read.
      let authoritativeMessages: ChatMessage[] | null = null
      const profile = await resolveSessionProfile(storedSessionId)

      if (storedSessionId) {
        try {
          const persisted = await getAllSessionMessages(storedSessionId, profile)
          const hydrated = toChatMessages(persisted.messages)

          if (hydrated.length) {
            authoritativeMessages = hydrated
          }
        } catch {
          // The branch RPC has a backend-side display projection fallback.
        }
      }

      const drift = sessionContextDrift({
        startRouteToken: startingRouteToken,
        nowRouteToken: getRouteToken(),
        startSelectedStoredId: storedSessionId,
        nowSelectedStoredId: selectedStoredSessionIdRef.current
      })

      const runtimeChanged = activeSessionIdRef.current !== startingActiveSessionId
      const selectionChanged = selectedStoredSessionIdRef.current !== storedSessionId

      if (drift || runtimeChanged || selectionChanged) {
        console.warn('[branch-drift-abort]', drift ?? 'runtime-or-selection-changed', {
          phase: 'transcript-hydration'
        })

        return false
      }

      const branchMessages = selectBranchMessages(messages, authoritativeMessages, messageId)

      if (!branchMessages.length) {
        notify({ kind: 'warning', title: copy.nothingToBranch, message: copy.branchNoText })

        return false
      }

      clearNotifications()

      // The open chat's owning profile, NOT the picker's / launch profile —
      // /profile only retargets new chats, so a branch of an existing thread
      // must stay on that thread's backend (cache hit for an open session).
      return forkBranch(
        branchMessages,
        startingActiveSessionId,
        storedSessionId,
        startingCwd,
        profile,
        messageId ? branchMessages.length : undefined
      )
    },
    [activeSessionIdRef, busyRef, copy, forkBranch, getRouteToken, selectedStoredSessionIdRef]
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
      const stored =
        $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId)) ??
        (sessionProfile ? undefined : await resolveStoredSession(storedSessionId))

      const profile = sessionProfile ?? stored?.profile

      try {
        await ensureGatewayProfile(profile)
        const { messages } = await getAllSessionMessages(storedSessionId, profile)
        const branchMessages = toBranchMessages(toChatMessages(messages))

        if (!branchMessages.length) {
          notify({ kind: 'warning', title: copy.nothingToBranch, message: copy.branchNoText })

          return false
        }

        return await forkBranch(branchMessages, null, stored?.id ?? storedSessionId, stored?.cwd?.trim(), profile)
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

      const wasSelected = selectedStoredSessionId === storedSessionId
      const closingRuntimeId = wasSelected ? activeSessionId : null
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

        if (closingRuntimeId) {
          clearQueuedPrompts(closingRuntimeId)
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
      activeSessionId,
      activeSessionIdRef,
      copy,
      navigate,
      requestGateway,
      runtimeIdByStoredSessionIdRef,
      selectedStoredSessionId,
      selectedStoredSessionIdRef,
      sessionStateByRuntimeIdRef,
      startFreshSessionDraft
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

      const wasSelected = selectedStoredSessionId === storedSessionId
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
    [copy, runtimeIdByStoredSessionIdRef, selectedStoredSessionId, sessionStateByRuntimeIdRef, startFreshSessionDraft]
  )

  return {
    archiveSession,
    branchCurrentSession,
    branchStoredSession,
    closeSettings,
    createBackendSessionForSend,
    openNewSessionTile,
    openSettings,
    removeSession,
    resumeSession,
    selectSidebarItem,
    startFreshSessionDraft
  }
}
