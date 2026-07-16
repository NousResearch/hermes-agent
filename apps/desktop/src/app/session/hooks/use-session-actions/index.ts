import type { MutableRefObject } from 'react'
import { useCallback, useRef } from 'react'
import type { NavigateFunction } from 'react-router-dom'

import { revealTreePane } from '@/components/pane-shell/tree/store'
import { deleteSession, getSessionMessages, setSessionArchived } from '@/hermes'
import { useI18n } from '@/i18n'
import { preserveLocalAssistantErrors, toChatMessages } from '@/lib/chat-messages'
import { sessionIdentityKey, sessionMatchesIdentity } from '@/lib/session-identity'
import { setSessionYolo } from '@/lib/yolo-session'
import { clearQueuedPrompts } from '@/store/composer-queue'
import { $pinnedSessionIds } from '@/store/layout'
import { clearNotifications, notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile, $newChatProfile, ensureGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { resolveNewSessionCwd, tombstoneSessions, untombstoneSessions } from '@/store/projects'
import {
  $currentCwd,
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $messages,
  $newChatWorkspaceTarget,
  $sessions,
  $yoloActive,
  type NewChatWorkspaceTarget,
  sessionPinId,
  setActiveSessionId,
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
  setSessions,
  setSessionStartedAt,
  setSessionsTotal,
  setTurnStartedAt,
  setYoloActive
} from '@/store/session'
import {
  discardSessionTile,
  dropSessionState,
  openSessionTile,
  patchSessionTile,
  publishSessionState,
  type TileDock
} from '@/store/session-states'
import { broadcastSessionsChanged } from '@/store/session-sync'
import { isWatchWindow } from '@/store/windows'
import type { SessionCreateResponse, SessionResumeResponse, UsageStats } from '@/types/hermes'

import { NEW_CHAT_ROUTE, sessionRoute, SETTINGS_ROUTE } from '../../../routes'
import type { ClientSessionState, SidebarNavItem } from '../../../types'

import {
  applyRuntimeInfo,
  applyStoredSessionPreviewRuntimeInfo,
  type BranchMessage,
  chatMessageArraysEquivalent,
  isSessionGoneError,
  patchSessionWorkspace,
  reconcileResumeMessages,
  resolveStoredSession,
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
  ensureSessionState: (
    sessionId: string,
    storedSessionId?: string | null,
    storedSessionProfile?: null | string
  ) => ClientSessionState
  getRouteToken: () => string
  navigate: NavigateFunction
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
    storedSessionId?: string | null,
    storedSessionProfile?: null | string
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
async function desktopSessionCreateParams(cwd: string): Promise<{ params: Record<string, unknown>; profile: string }> {
  const profile = $newChatProfile.get() ?? normalizeProfileKey($activeGatewayProfile.get())
  await ensureGatewayProfile(profile)

  const model = $currentModel.get().trim()
  const provider = $currentProvider.get().trim()
  const effort = $currentReasoningEffort.get().trim()

  return {
    profile,
    params: {
      cols: 96,
      source: 'desktop',
      ...(cwd && { cwd }),
      ...(profile ? { profile } : {}),
      ...(model ? { model, ...(provider ? { provider } : {}) } : {}),
      ...(effort ? { reasoning_effort: effort } : {}),
      ...($currentFastMode.get() ? { fast: true } : {})
    }
  }
}

interface FreshSessionDraftOptions {
  replaceRoute?: boolean
  workspaceTarget?: NewChatWorkspaceTarget
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
  navigate,
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

  const startFreshSessionDraft = useCallback(
    (options: boolean | FreshSessionDraftOptions = false) => {
      const draftOptions = typeof options === 'boolean' ? { replaceRoute: options } : options
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
      navigate(NEW_CHAT_ROUTE, { replace: replaceRoute })
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
        setCurrentCwd(resolveNewSessionCwd())
      } else if (workspaceTarget === null) {
        setCurrentCwdTransient('')
      } else if (typeof workspaceTarget === 'string') {
        setCurrentCwd(workspaceTarget)
      }

      setCurrentBranch('')
      // Never clear the composer here — ChatBar's per-thread draft swap owns it.
      setFreshDraftReady(true)
    },
    [activeSessionIdRef, busyRef, navigate, resetViewSync, selectedStoredSessionIdRef]
  )

  const createBackendSessionForSend = useCallback(
    async (preview: string | null = null): Promise<string | null> => {
      const startingActiveSessionId = activeSessionIdRef.current
      const startingStoredSessionId = selectedStoredSessionIdRef.current
      const startingRouteToken = getRouteToken()

      creatingSessionRef.current = true

      try {
        // An explicit one-shot workspace target (null → detached, string → that
        // folder) wins; otherwise the live cwd, then the project-aware default
        // (resolveNewSessionCwd — a project's new session keeps its repo cwd).
        const workspaceTarget = $newChatWorkspaceTarget.get()

        const cwd =
          workspaceTarget === null
            ? ''
            : typeof workspaceTarget === 'string'
              ? workspaceTarget.trim()
              : $currentCwd.get().trim() || resolveNewSessionCwd()

        const { params, profile: newChatProfile } = await desktopSessionCreateParams(cwd)
        const created = await requestGateway<SessionCreateResponse>('session.create', params)
        const stored = created.stored_session_id ?? null

        if (
          activeSessionIdRef.current !== startingActiveSessionId ||
          selectedStoredSessionIdRef.current !== startingStoredSessionId ||
          getRouteToken() !== startingRouteToken
        ) {
          await requestGateway('session.close', { session_id: created.session_id }).catch(() => undefined)

          return null
        }

        resetViewSync()
        activeSessionIdRef.current = created.session_id
        selectedStoredSessionIdRef.current = stored
        ensureSessionState(created.session_id, stored, newChatProfile)

        if (stored) {
          createdThisRun.add(stored)
          // Seed the sidebar preview with the user's first message so the row
          // reads meaningfully while the turn is in flight, instead of flashing
          // "Untitled session" until the turn persists and auto-title runs. The
          // server later returns its own preview/title and supersedes this.
          upsertOptimisticSession(created, stored, null, preview?.trim() || null, null, undefined, newChatProfile)
          navigate(sessionRoute(stored, newChatProfile), { replace: true })
          // Other windows (e.g. the main window when this is the pop-out) can't
          // see this session until they re-pull the shared list.
          broadcastSessionsChanged()
        }

        setFreshDraftReady(false)
        setNewChatWorkspaceTarget(undefined)
        setActiveSessionId(created.session_id)
        setSelectedStoredSessionId(stored, newChatProfile)
        setSessionStartedAt(Date.now())
        const yoloArmed = $yoloActive.get()
        const runtimeInfo = applyRuntimeInfo(created.info)

        if (runtimeInfo) {
          updateSessionState(created.session_id, state => ({ ...state, ...runtimeInfo }), stored, newChatProfile)
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
        startFreshSessionDraft()

        return
      }

      if (item.route) {
        navigate(item.route)
      }
    },
    [navigate, startFreshSessionDraft]
  )

  /** Create a fresh session and open it as a tile — leaves the primary chat alone.
   *  Used by the New session row's "Open in split" menu (and any future
   *  "new chat beside" affordance). */
  const openNewSessionTile = useCallback(
    async (dir: TileDock = 'right') => {
      try {
        // Fresh tile → the resolved new-session cwd (project/default), not the
        // primary composer's live cwd.
        const { params, profile: newChatProfile } = await desktopSessionCreateParams(resolveNewSessionCwd().trim())
        const created = await requestGateway<SessionCreateResponse>('session.create', params)
        const stored = created.stored_session_id

        if (!stored) {
          await requestGateway('session.close', { session_id: created.session_id }).catch(() => undefined)
          notify({ kind: 'error', title: copy.sessionUnavailable, message: copy.createSessionFailed })

          return
        }

        createdThisRun.add(stored)
        // Seed the sidebar + per-runtime cache, but DON'T steal the primary
        // selection — this session lives in the tile. Prime it with the create
        // runtime so the tile skips a redundant resume.
        upsertOptimisticSession(created, stored, null, null, null, undefined, newChatProfile)
        const runtimeInfo = applyRuntimeInfo(created.info)
        updateSessionState(
          created.session_id,
          state => (runtimeInfo ? { ...state, ...runtimeInfo } : state),
          stored,
          newChatProfile
        )

        openSessionTile(stored, dir, undefined, undefined, newChatProfile)
        patchSessionTile(stored, { runtimeId: created.session_id })
        revealTreePane(`session-tile:${stored}`)
        broadcastSessionsChanged()
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
      navigate(sessionRoute(selectedStoredSessionId, $activeGatewayProfile.get()))

      return
    }

    navigate(NEW_CHAT_ROUTE)
  }, [navigate, selectedStoredSessionId])

  const resumeSession = useCallback(
    async (storedSessionId: string, replaceRoute = false, profileHint?: null | string) => {
      const requestId = resumeRequestRef.current + 1
      resumeRequestRef.current = requestId

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
      setSelectedStoredSessionId(storedSessionId, profileHint)
      selectedStoredSessionIdRef.current = storedSessionId
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
      // The session.usage 404 guard below only catches a fully-DEAD id — a
      // recycled-live id 200s, so an unchecked hit paints the wrong transcript
      // under the current route (the "open chat A, chat B loads" bug). On a
      // mismatch the mapping is cross-wired: purge both sides and report a miss
      // so the caller falls through to a full resume that rebinds a correct id.
      const takeWarmCache = (profile?: null | string): { runtimeId: string; state: ClientSessionState } | null => {
        const cacheKey = sessionIdentityKey(storedSessionId, profile)
        const runtimeId = runtimeIdByStoredSessionIdRef.current.get(cacheKey)
        const state = runtimeId ? sessionStateByRuntimeIdRef.current.get(runtimeId) : undefined

        if (!runtimeId || !state) {
          return null
        }

        if (
          state.storedSessionId !== storedSessionId ||
          normalizeProfileKey(state.storedSessionProfile) !== normalizeProfileKey(profile)
        ) {
          runtimeIdByStoredSessionIdRef.current.delete(cacheKey)
          sessionStateByRuntimeIdRef.current.delete(runtimeId)
          dropSessionState(runtimeId)

          return null
        }

        return { runtimeId, state }
      }

      if (!takeWarmCache(profileHint ?? $activeGatewayProfile.get())) {
        setActiveSessionId(null)
        activeSessionIdRef.current = null
        setMessages([])
      }

      // Swap the single live gateway to this session's profile before any
      // gateway call (no-op when it's already on that profile / single-profile).
      // resolveStoredSession finds the row by id (cheap), so an uncached pasted
      // id loads as fast as a sidebar click instead of hanging on a list scan.
      const storedForProfile = await resolveStoredSession(storedSessionId, profileHint)

      const sessionProfile = storedForProfile?.profile
        ? normalizeProfileKey(storedForProfile.profile)
        : profileHint
          ? normalizeProfileKey(profileHint)
          : undefined

      if (resumeRequestRef.current !== requestId) {
        return
      }

      await ensureGatewayProfile(sessionProfile)

      // Re-check after the profile-resolve / gateway-swap awaits above: the
      // cache may have changed, and takeWarmCache re-validates belongs-to and
      // purges a cross-wired mapping before we trust the fast-path.
      const warmHit = takeWarmCache(sessionProfile ?? $activeGatewayProfile.get())

      if (warmHit) {
        const cachedRuntimeId = warmHit.runtimeId
        const cachedState = warmHit.state

        const stored =
          $sessions
            .get()
            .find(
              session =>
                sessionMatchesStoredId(session, storedSessionId) &&
                (!sessionProfile || normalizeProfileKey(session.profile) === sessionProfile)
            ) ?? storedForProfile

        const cachedViewState =
          !cachedState.model && stored?.model != null
            ? {
                ...cachedState,
                model: stored.model || ''
              }
            : cachedState

        if (cachedViewState !== cachedState) {
          sessionStateByRuntimeIdRef.current.set(cachedRuntimeId, cachedViewState)
          publishSessionState(cachedRuntimeId, cachedViewState)
        }

        if (sessionShouldHaveTranscript(stored) && cachedViewState.messages.length === 0) {
          runtimeIdByStoredSessionIdRef.current.delete(sessionIdentityKey(storedSessionId, sessionProfile))
          sessionStateByRuntimeIdRef.current.delete(cachedRuntimeId)
          dropSessionState(cachedRuntimeId)
        } else {
          setFreshDraftReady(false)
          clearNotifications()
          setSelectedStoredSessionId(storedSessionId, sessionProfile)
          selectedStoredSessionIdRef.current = storedSessionId
          setActiveSessionId(cachedRuntimeId)
          activeSessionIdRef.current = cachedRuntimeId
          syncSessionStateToView(cachedRuntimeId, cachedViewState)
          setCurrentCwd(cachedViewState.cwd)
          setCurrentBranch(cachedViewState.branch)
          setSessionStartedAt(Date.now())

          try {
            const usage = await requestGateway<UsageStats>('session.usage', { session_id: cachedRuntimeId })

            if (!isCurrentResume()) {
              return
            }

            if (usage) {
              setCurrentUsage(current => ({ ...current, ...usage }))
            }

            return
          } catch {
            // The cached runtime id was minted by a prior backend instance. A
            // pooled profile backend that gets idle-reaped (pruneSecondaryGateways)
            // and respawned across a profile swap mints fresh ids, so this mapping
            // now 404s ("session not found"). Drop it and fall through to a full
            // resume that rebinds a live runtime id.
            if (!isCurrentResume()) {
              return
            }

            runtimeIdByStoredSessionIdRef.current.delete(sessionIdentityKey(storedSessionId, sessionProfile))
            sessionStateByRuntimeIdRef.current.delete(cachedRuntimeId)
            dropSessionState(cachedRuntimeId)
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
      if ($messages.get().length > 0) {
        setMessages([])
      }

      busyRef.current = true
      setBusy(true)
      setAwaitingResponse(false)
      clearNotifications()
      setSelectedStoredSessionId(storedSessionId, sessionProfile)
      selectedStoredSessionIdRef.current = storedSessionId
      setSessionStartedAt(Date.now())

      const stored =
        $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId)) ?? storedForProfile

      applyStoredSessionPreviewRuntimeInfo(stored)

      if (stored) {
        applyStoredUsage(stored)
      }

      let resumedRunning = false

      try {
        const watchWindow = isWatchWindow()
        let localSnapshot = $messages.get()

        // REST transcript prefetch and the gateway resume RPC are independent
        // — run them concurrently so a big session's wall time is
        // max(prefetch, resume) instead of their sum. The prefetch paints the
        // transcript as soon as it lands; the RPC binds the runtime id.
        // Watch windows skip the prefetch — lazy resume attaches the live mirror.
        const prefetchPromise = watchWindow ? null : getSessionMessages(storedSessionId, sessionProfile)

        const resumePromise = requestGateway<SessionResumeResponse>('session.resume', {
          session_id: storedSessionId,
          cols: 96,
          source: 'desktop',
          // Watch windows attach lazily (live mirror). Every other cold resume
          // gets the gateway's default deferred build: the RPC returns the
          // transcript immediately instead of blocking the switch on _make_agent
          // (MCP discovery / prompt build), and the agent pre-warms in the
          // background while the prefetch above paints the transcript.
          ...(watchWindow ? { lazy: true } : {}),
          ...(sessionProfile ? { profile: sessionProfile } : {})
        })

        // The rejection is consumed by the `await` below; this guard only
        // keeps it from surfacing as unhandled while the prefetch settles.
        resumePromise.catch(() => undefined)

        try {
          if (prefetchPromise) {
            const storedMessages = await prefetchPromise

            if (isCurrentResume()) {
              localSnapshot = preserveLocalAssistantErrors(toChatMessages(storedMessages.messages), $messages.get())

              if (!chatMessageArraysEquivalent($messages.get(), localSnapshot)) {
                setMessages(localSnapshot)
              }
            }
          }
        } catch {
          // Non-fatal: gateway resume below can still hydrate the session.
        }

        const resumed = await resumePromise

        if (!isCurrentResume()) {
          return
        }

        const currentMessages = $messages.get()

        // Keep the local snapshot when resume would only reshuffle runtime
        // projection. When the REST prefetch already hydrated the transcript,
        // skip converting/reconciling the resume payload entirely — on a
        // 1000+-message session that second conversion plus the deep
        // equivalence compare costs over a second of main-thread time.
        const preferredMessages =
          localSnapshot.length > 0
            ? localSnapshot
            : (() => {
                const resumedMessages = preserveLocalAssistantErrors(
                  reconcileResumeMessages(toChatMessages(resumed.messages), currentMessages),
                  currentMessages
                )

                return chatMessageArraysEquivalent(currentMessages, resumedMessages) ? currentMessages : resumedMessages
              })()

        // Prefetch-hit fast path: `preferredMessages` IS the live `$messages`
        // array (already error-merged when `localSnapshot` was built), so reuse
        // the ref instead of rebuilding a throwaway transcript+Map every switch.
        const messagesForView =
          preferredMessages === currentMessages
            ? currentMessages
            : preserveLocalAssistantErrors(preferredMessages, currentMessages)

        if (sessionShouldHaveTranscript(stored) && messagesForView.length === 0) {
          setActiveSessionId(null)
          activeSessionIdRef.current = null
          setResumeFailedSessionId(storedSessionId)
          resumedRunning = false

          return
        }

        setActiveSessionId(resumed.session_id)
        activeSessionIdRef.current = resumed.session_id
        const runtimeInfo = applyRuntimeInfo(resumed.info)

        patchSessionWorkspace(storedSessionId, runtimeInfo?.cwd, sessionProfile)

        resumedRunning = Boolean((resumed as { running?: boolean }).running)

        updateSessionState(
          resumed.session_id,
          state => ({
            ...state,
            ...(runtimeInfo ?? {}),
            messages: messagesForView,
            busy: resumedRunning,
            awaitingResponse: resumedRunning
          }),
          storedSessionId,
          sessionProfile
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
          const fallback = await getSessionMessages(storedSessionId, sessionProfile)

          if (!isCurrentResume()) {
            return
          }

          setMessages(preserveLocalAssistantErrors(toChatMessages(fallback.messages), $messages.get()))
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

        // The session is genuinely gone (deleted, or a stale id from a wiped /
        // rotated backend): the resume RPC and the authoritative REST transcript
        // both 404. There's nothing to recover — silently drop to a fresh draft
        // instead of toasting an error and hot-looping the bounded retry on a
        // permanently-dead id. (Booting straight into a no-longer-existent
        // last-session id is the common trigger.)
        if ($messages.get().length === 0 && isSessionGoneError(fallbackError)) {
          // A session created THIS run isn't gone — its backend just flapped
          // before the turn-less session persisted. Keep the empty view and arm
          // the bounded retry to rebind, rather than yanking to a fresh draft.
          // Only a stale id from a PRIOR run drops to a draft.
          if (createdThisRun.has(storedSessionId)) {
            setResumeFailedSessionId(storedSessionId)

            return
          }

          startFreshSessionDraft(true)

          return
        }

        if ($messages.get().length === 0) {
          // Arm the self-heal ONLY when the window is still empty: the gateway
          // resume rejected AND the REST fallback failed to paint a transcript.
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
          setAwaitingResponse(resumedRunning)
        }
      }
    },
    [
      activeSessionIdRef,
      busyRef,
      copy,
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
  // `parentStoredId` so it nests under its parent, then make it the active chat.
  const forkBranch = useCallback(
    async (branchMessages: BranchMessage[], parentStoredId: null | string, cwd?: string): Promise<boolean> => {
      creatingSessionRef.current = true

      try {
        // No title: the backend auto-names the branch from its parent's lineage.
        const branched = await requestGateway<SessionCreateResponse>('session.create', {
          cols: 96,
          source: 'desktop',
          ...(cwd && { cwd }),
          messages: branchMessages.map(({ content, role }) => ({ content, role })),
          ...(parentStoredId && { parent_session_id: parentStoredId })
        })

        const routedSessionId = branched.stored_session_id ?? branched.session_id
        const preview = branchMessages.map(({ content }) => content).find(Boolean) ?? null
        // Draft until submit: nest under the parent at the parent's recency so it
        // doesn't bubble to the top until a real message lands (backend persists
        // + auto-names it then). The selected row survives refreshes (sessionsToKeep).
        const rows = $sessions.get()
        const branchProfile = $activeGatewayProfile.get()

        const parent = parentStoredId
          ? rows.find(session => sessionMatchesIdentity(session, parentStoredId, branchProfile))
          : null

        const siblings = parentStoredId
          ? rows.filter(
              session =>
                normalizeProfileKey(session.profile) === normalizeProfileKey(branchProfile) &&
                session.parent_session_id?.trim() === parentStoredId
            ).length
          : 0

        setFreshDraftReady(false)
        upsertOptimisticSession(
          branched,
          routedSessionId,
          copy.branchTitle(siblings + 1).toLowerCase(),
          preview,
          parentStoredId,
          parent ? parent.last_active || parent.started_at : undefined,
          branchProfile
        )
        ensureSessionState(branched.session_id, routedSessionId, branchProfile)
        setActiveSessionId(branched.session_id)
        activeSessionIdRef.current = branched.session_id
        updateSessionState(
          branched.session_id,
          state => ({
            ...state,
            messages: branchMessages.map(({ source }) => source),
            busy: false,
            awaitingResponse: false
          }),
          routedSessionId,
          branchProfile
        )
        setSelectedStoredSessionId(routedSessionId, branchProfile)
        selectedStoredSessionIdRef.current = routedSessionId
        navigate(sessionRoute(routedSessionId, branchProfile))

        const runtimeInfo = applyRuntimeInfo(branched.info)
        patchSessionWorkspace(routedSessionId, runtimeInfo?.cwd, branchProfile)

        if (runtimeInfo) {
          updateSessionState(
            branched.session_id,
            state => ({ ...state, ...runtimeInfo }),
            routedSessionId,
            branchProfile
          )
        }

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
      activeSessionIdRef,
      copy,
      creatingSessionRef,
      ensureSessionState,
      navigate,
      requestGateway,
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

      const messages = $messages.get()

      const at = messageId
        ? messages.findIndex(message => message.id === messageId)
        : messages.findLastIndex(message => message.role === 'assistant' || message.role === 'user')

      const start = at >= 0 ? at : Math.max(messages.length - 1, 0)
      const end = at >= 0 ? at + 1 : messages.length
      const branchMessages = toBranchMessages(messages.slice(start, end))

      if (!branchMessages.length) {
        notify({ kind: 'warning', title: copy.nothingToBranch, message: copy.branchNoText })

        return false
      }

      clearNotifications()

      return forkBranch(branchMessages, selectedStoredSessionIdRef.current, $currentCwd.get().trim())
    },
    [activeSessionIdRef, busyRef, copy, forkBranch, selectedStoredSessionIdRef]
  )

  // Branch any listed session, not just the open one. Reads the target's stored
  // transcript directly (no resume/active-session dependency), so it works on
  // right-click and nests under its parent.
  const branchStoredSession = useCallback(
    async (storedSessionId: string, sessionProfile?: string | null): Promise<boolean> => {
      clearNotifications()

      const stored = $sessions
        .get()
        .find(session =>
          sessionProfile === undefined
            ? sessionMatchesStoredId(session, storedSessionId)
            : sessionMatchesIdentity(session, storedSessionId, sessionProfile)
        )

      const profile = sessionProfile ?? stored?.profile

      try {
        await ensureGatewayProfile(profile)
        const { messages } = await getSessionMessages(storedSessionId, profile)
        const branchMessages = toBranchMessages(toChatMessages(messages))

        if (!branchMessages.length) {
          notify({ kind: 'warning', title: copy.nothingToBranch, message: copy.branchNoText })

          return false
        }

        return await forkBranch(branchMessages, stored?.id ?? storedSessionId, stored?.cwd?.trim())
      } catch (err) {
        notifyError(err, copy.branchFailed)

        return false
      }
    },
    [copy, forkBranch]
  )

  const removeSession = useCallback(
    async (storedSessionId: string, sessionProfile?: null | string) => {
      clearNotifications()

      const removed = $sessions
        .get()
        .find(session =>
          sessionProfile === undefined
            ? sessionMatchesStoredId(session, storedSessionId)
            : sessionMatchesIdentity(session, storedSessionId, sessionProfile)
        )

      const profile = normalizeProfileKey(sessionProfile ?? removed?.profile ?? $activeGatewayProfile.get())

      const matchesRemoved = (session: Parameters<typeof sessionMatchesStoredId>[0]) =>
        sessionMatchesIdentity(session, storedSessionId, profile)

      const wasSelected =
        selectedStoredSessionId === storedSessionId && normalizeProfileKey($activeGatewayProfile.get()) === profile

      const closingRuntimeId = wasSelected ? activeSessionId : null
      const previousMessages = $messages.get()
      const previousPinned = $pinnedSessionIds.get()
      // Pins are keyed on the durable lineage-root id; the stored id may be the
      // live tip after compression. Drop both so the pin can't linger.
      const removedPinId = removed ? sessionPinId(removed) : sessionIdentityKey(storedSessionId, profile)

      setSessions(prev => prev.filter(session => !matchesRemoved(session)))
      // Evict from the project tree's optimistic layer too (the backend snapshot
      // still lists it until its next refresh), so grouped + flat views drop the
      // row in lockstep.
      tombstoneSessions([storedSessionId, removed?.id, removed?._lineage_root_id], profile)
      // Keep $sessionsTotal in sync so the sidebar's "Load N more" footer
      // doesn't keep claiming the removed row is still on the server.
      setSessionsTotal(prev => Math.max(0, prev - 1))
      $pinnedSessionIds.set(previousPinned.filter(id => id !== removedPinId))

      // Tear down before awaiting so the route effect can't resume the
      // doomed session via the stale /<sid> URL.
      if (wasSelected) {
        startFreshSessionDraft(true)
      }

      try {
        if (closingRuntimeId) {
          await requestGateway('session.close', { session_id: closingRuntimeId }).catch(() => undefined)
        }

        await deleteSession(storedSessionId, profile)
        clearQueuedPrompts(storedSessionId)

        if (closingRuntimeId) {
          clearQueuedPrompts(closingRuntimeId)
        }

        // A tiled copy of this session must not outlive it: collapse the pane
        // and evict its mirrored runtime state so nothing submits to (or renders)
        // a deleted session.
        const tileCacheKey = sessionIdentityKey(storedSessionId, profile)
        const tiledRuntimeId = runtimeIdByStoredSessionIdRef.current.get(tileCacheKey)
        discardSessionTile(storedSessionId, profile)

        if (tiledRuntimeId) {
          runtimeIdByStoredSessionIdRef.current.delete(tileCacheKey)
          sessionStateByRuntimeIdRef.current.delete(tiledRuntimeId)
          dropSessionState(tiledRuntimeId)
        }
      } catch (err) {
        if (removed) {
          setSessions(prev => [removed, ...prev])
          setSessionsTotal(prev => prev + 1)
        }

        untombstoneSessions([storedSessionId, removed?.id, removed?._lineage_root_id], profile)
        $pinnedSessionIds.set(previousPinned)

        if (wasSelected) {
          setFreshDraftReady(false)
          setSelectedStoredSessionId(storedSessionId, profile)
          selectedStoredSessionIdRef.current = storedSessionId
          const stored = $sessions.get().find(matchesRemoved)

          if (stored) {
            applyStoredUsage(stored)
          }

          setMessages(previousMessages)
          navigate(sessionRoute(storedSessionId, profile), { replace: true })

          if (closingRuntimeId) {
            setActiveSessionId(closingRuntimeId)
            activeSessionIdRef.current = closingRuntimeId
          }
        }

        notifyError(err, copy.deleteFailed)
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
    async (storedSessionId: string, sessionProfile?: null | string) => {
      clearNotifications()

      const archived = $sessions
        .get()
        .find(session =>
          sessionProfile === undefined
            ? sessionMatchesStoredId(session, storedSessionId)
            : sessionMatchesIdentity(session, storedSessionId, sessionProfile)
        )

      const profile = normalizeProfileKey(sessionProfile ?? archived?.profile ?? $activeGatewayProfile.get())

      const matchesArchived = (session: Parameters<typeof sessionMatchesStoredId>[0]) =>
        sessionMatchesIdentity(session, storedSessionId, profile)

      const wasSelected =
        selectedStoredSessionId === storedSessionId && normalizeProfileKey($activeGatewayProfile.get()) === profile

      const previousPinned = $pinnedSessionIds.get()
      // Pins are keyed on the durable lineage-root id; the stored id may be the
      // live tip after compression. Drop both so the pin can't linger.
      const archivedPinId = archived ? sessionPinId(archived) : sessionIdentityKey(storedSessionId, profile)

      // Soft-hide: drop from the sidebar immediately, keep the data.
      setSessions(prev => prev.filter(session => !matchesArchived(session)))
      tombstoneSessions([storedSessionId, archived?.id, archived?._lineage_root_id], profile)
      // Archived sessions are hidden by the listSessions(min_messages=1) query
      // on the next refresh, so they count as "removed" for the load-more
      // footer math.
      setSessionsTotal(prev => Math.max(0, prev - 1))
      $pinnedSessionIds.set(previousPinned.filter(id => id !== archivedPinId))

      if (wasSelected) {
        startFreshSessionDraft(true)
      }

      try {
        await setSessionArchived(storedSessionId, true, profile)
        // A sidebar refresh can race the optimistic removal while the PATCH is
        // in flight and briefly reinsert the still-unarchived backend row. Win
        // that race after the mutation succeeds so right-click → Archive does
        // not appear to do nothing until the next full refresh.
        setSessions(prev => prev.filter(session => !matchesArchived(session)))
        $pinnedSessionIds.set($pinnedSessionIds.get().filter(id => id !== archivedPinId))
        // An archived session is hidden from the sidebar; its tile must go too.
        const tileCacheKey = sessionIdentityKey(storedSessionId, profile)
        const tiledRuntimeId = runtimeIdByStoredSessionIdRef.current.get(tileCacheKey)
        discardSessionTile(storedSessionId, profile)

        if (tiledRuntimeId) {
          runtimeIdByStoredSessionIdRef.current.delete(tileCacheKey)
          sessionStateByRuntimeIdRef.current.delete(tiledRuntimeId)
          dropSessionState(tiledRuntimeId)
        }

        notify({ durationMs: 2_000, kind: 'success', message: copy.archived })
      } catch (err) {
        if (archived) {
          setSessions(prev => [archived, ...prev.filter(session => !matchesArchived(session))])
          setSessionsTotal(prev => prev + 1)
        }

        untombstoneSessions([storedSessionId, archived?.id, archived?._lineage_root_id], profile)
        $pinnedSessionIds.set(previousPinned)
        notifyError(err, copy.archiveFailed)
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
