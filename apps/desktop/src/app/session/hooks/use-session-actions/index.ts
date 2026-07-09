import type { MutableRefObject } from 'react'
import { useCallback, useRef } from 'react'
import type { NavigateFunction } from 'react-router-dom'

import { deleteSession, getSessionMessages, setSessionArchived } from '@/hermes'
import { useI18n } from '@/i18n'
import { preserveLocalAssistantErrors, toChatMessages } from '@/lib/chat-messages'
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
  $sessions,
  $yoloActive,
  sessionPinId,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setCurrentBranch,
  setCurrentCwd,
  setCurrentServiceTier,
  setCurrentUsage,
  setFreshDraftReady,
  setIntroSeed,
  setMessages,
  setResumeExhaustedSessionId,
  setResumeFailedSessionId,
  setSelectedStoredSessionId,
  setSessions,
  setSessionStartedAt,
  setSessionsTotal,
  setTurnStartedAt,
  setYoloActive,
  workspaceCwdForNewSession
} from '@/store/session'
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
  ensureSessionState: (sessionId: string, storedSessionId?: string | null) => ClientSessionState
  getRouteToken: () => string
  navigate: NavigateFunction
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
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

export function useSessionActions({
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
}: SessionActionsOptions) {
  const { t } = useI18n()
  const copy = t.desktop
  const resumeRequestRef = useRef(0)

  const startFreshSessionDraft = useCallback(
    (replaceRoute = false) => {
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
      // In a project → the repo's default-branch (main worktree) checkout; not in
      // a project → detached. So cmd-n "knows" the project instead of inheriting
      // whatever linked worktree the last session drifted into.
      setCurrentCwd(resolveNewSessionCwd())
      setCurrentBranch('')
      // Never clear the composer here — ChatBar's per-thread draft swap owns it.
      setFreshDraftReady(true)
    },
    [activeSessionIdRef, busyRef, navigate, selectedStoredSessionIdRef]
  )

  const createBackendSessionForSend = useCallback(
    async (preview: string | null = null): Promise<string | null> => {
      const startingActiveSessionId = activeSessionIdRef.current
      const startingStoredSessionId = selectedStoredSessionIdRef.current
      const startingRouteToken = getRouteToken()

      creatingSessionRef.current = true

      try {
        // A plain new session (top "New Session", /new, keybind) leaves
        // $newChatProfile null to mean "use the live context"; the per-profile
        // "+" sets it explicitly. Resolve null to the active gateway profile so
        // session.create always carries it: in global-remote mode one backend
        // serves every profile, so an omitted profile param silently lands the
        // chat on the launch (default) profile — the "rubberbands back to
        // default" bug. This is a no-op for single-profile/local-pooled users:
        // a backend resolves its own launch profile to None (_profile_home).
        const newChatProfile = $newChatProfile.get() ?? normalizeProfileKey($activeGatewayProfile.get())
        await ensureGatewayProfile(newChatProfile)
        const cwd = $currentCwd.get().trim() || workspaceCwdForNewSession()
        // The composer's model/effort/fast is sticky UI state ($currentModel,
        // $currentProvider, $currentReasoningEffort, $currentFastMode). Ship it
        // with every session.create so the new chat opens on whatever the picker
        // shows — applied as per-session overrides, never written to the profile
        // default (that lives in Settings → Model).
        const uiModel = $currentModel.get().trim()
        const uiProvider = $currentProvider.get().trim()
        const uiEffort = $currentReasoningEffort.get().trim()
        const uiFast = $currentFastMode.get()

        const created = await requestGateway<SessionCreateResponse>('session.create', {
          cols: 96,
          source: 'desktop',
          ...(cwd && { cwd }),
          ...(newChatProfile ? { profile: newChatProfile } : {}),
          ...(uiModel ? { model: uiModel, ...(uiProvider ? { provider: uiProvider } : {}) } : {}),
          ...(uiEffort ? { reasoning_effort: uiEffort } : {}),
          ...(uiFast ? { fast: true } : {})
        })

        const stored = created.stored_session_id ?? null

        if (
          activeSessionIdRef.current !== startingActiveSessionId ||
          selectedStoredSessionIdRef.current !== startingStoredSessionId ||
          getRouteToken() !== startingRouteToken
        ) {
          await requestGateway('session.close', { session_id: created.session_id }).catch(() => undefined)

          return null
        }

        activeSessionIdRef.current = created.session_id
        selectedStoredSessionIdRef.current = stored
        ensureSessionState(created.session_id, stored)

        if (stored) {
          // Seed the sidebar preview with the user's first message so the row
          // reads meaningfully while the turn is in flight, instead of flashing
          // "Untitled session" until the turn persists and auto-title runs. The
          // server later returns its own preview/title and supersedes this.
          upsertOptimisticSession(created, stored, null, preview?.trim() || null)
          navigate(sessionRoute(stored), { replace: true })
          // Other windows (e.g. the main window when this is the pop-out) can't
          // see this session until they re-pull the shared list.
          broadcastSessionsChanged()
        }

        setFreshDraftReady(false)
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
    async (storedSessionId: string, replaceRoute = false) => {
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
      setSelectedStoredSessionId(storedSessionId)
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
      const takeWarmCache = (): { runtimeId: string; state: ClientSessionState } | null => {
        const runtimeId = runtimeIdByStoredSessionIdRef.current.get(storedSessionId)
        const state = runtimeId ? sessionStateByRuntimeIdRef.current.get(runtimeId) : undefined

        if (!runtimeId || !state) {
          return null
        }

        if (state.storedSessionId !== storedSessionId) {
          runtimeIdByStoredSessionIdRef.current.delete(storedSessionId)
          sessionStateByRuntimeIdRef.current.delete(runtimeId)

          return null
        }

        return { runtimeId, state }
      }

      if (!takeWarmCache()) {
        setActiveSessionId(null)
        activeSessionIdRef.current = null
        setMessages([])
      }

      // Swap the single live gateway to this session's profile before any
      // gateway call (no-op when it's already on that profile / single-profile).
      // resolveStoredSession finds the row by id (cheap), so an uncached pasted
      // id loads as fast as a sidebar click instead of hanging on a list scan.
      const storedForProfile = await resolveStoredSession(storedSessionId)
      const sessionProfile = storedForProfile?.profile

      if (resumeRequestRef.current !== requestId) {
        return
      }

      await ensureGatewayProfile(sessionProfile)

      // Re-check after the profile-resolve / gateway-swap awaits above: the
      // cache may have changed, and takeWarmCache re-validates belongs-to and
      // purges a cross-wired mapping before we trust the fast-path.
      const warmHit = takeWarmCache()

      if (warmHit) {
        const cachedRuntimeId = warmHit.runtimeId
        const cachedState = warmHit.state

        const stored =
          $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId)) ?? storedForProfile

        const cachedViewState =
          !cachedState.model && stored?.model != null
            ? {
                ...cachedState,
                model: stored.model || ''
              }
            : cachedState

        if (cachedViewState !== cachedState) {
          sessionStateByRuntimeIdRef.current.set(cachedRuntimeId, cachedViewState)
        }

        if (sessionShouldHaveTranscript(stored) && cachedViewState.messages.length === 0) {
          runtimeIdByStoredSessionIdRef.current.delete(storedSessionId)
          sessionStateByRuntimeIdRef.current.delete(cachedRuntimeId)
        } else {
          setFreshDraftReady(false)
          clearNotifications()
          setSelectedStoredSessionId(storedSessionId)
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

            runtimeIdByStoredSessionIdRef.current.delete(storedSessionId)
            sessionStateByRuntimeIdRef.current.delete(cachedRuntimeId)
          }
        }
      }

      setFreshDraftReady(false)
      setActiveSessionId(null)
      activeSessionIdRef.current = null
      busyRef.current = true
      setBusy(true)
      setAwaitingResponse(false)
      clearNotifications()
      setSelectedStoredSessionId(storedSessionId)
      selectedStoredSessionIdRef.current = storedSessionId
      setSessionStartedAt(Date.now())

      const stored =
        $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId)) ?? storedForProfile

      applyStoredSessionPreviewRuntimeInfo(stored)

      if (stored) {
        setCurrentUsage(current => ({
          ...current,
          input: stored.input_tokens || 0,
          output: stored.output_tokens || 0,
          total: (stored.input_tokens || 0) + (stored.output_tokens || 0)
        }))
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

        patchSessionWorkspace(storedSessionId, runtimeInfo?.cwd)

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
          storedSessionId
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
          routedSessionId
        )
        setSelectedStoredSessionId(routedSessionId)
        selectedStoredSessionIdRef.current = routedSessionId
        navigate(sessionRoute(routedSessionId))

        const runtimeInfo = applyRuntimeInfo(branched.info)
        patchSessionWorkspace(routedSessionId, runtimeInfo?.cwd)

        if (runtimeInfo) {
          updateSessionState(branched.session_id, state => ({ ...state, ...runtimeInfo }), routedSessionId)
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

      const stored = $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))
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
    async (storedSessionId: string) => {
      clearNotifications()

      const removed = $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))
      const wasSelected = selectedStoredSessionId === storedSessionId
      const closingRuntimeId = wasSelected ? activeSessionId : null
      const previousMessages = $messages.get()
      const previousPinned = $pinnedSessionIds.get()
      // Pins are keyed on the durable lineage-root id; the stored id may be the
      // live tip after compression. Drop both so the pin can't linger.
      const removedPinId = removed ? sessionPinId(removed) : storedSessionId

      setSessions(prev => prev.filter(session => !sessionMatchesStoredId(session, storedSessionId)))
      // Evict from the project tree's optimistic layer too (the backend snapshot
      // still lists it until its next refresh), so grouped + flat views drop the
      // row in lockstep.
      tombstoneSessions([storedSessionId, removed?.id, removed?._lineage_root_id])
      // Keep $sessionsTotal in sync so the sidebar's "Load N more" footer
      // doesn't keep claiming the removed row is still on the server.
      setSessionsTotal(prev => Math.max(0, prev - 1))
      $pinnedSessionIds.set(previousPinned.filter(id => id !== storedSessionId && id !== removedPinId))

      // Tear down before awaiting so the route effect can't resume the
      // doomed session via the stale /<sid> URL.
      if (wasSelected) {
        startFreshSessionDraft(true)
      }

      try {
        if (closingRuntimeId) {
          await requestGateway('session.close', { session_id: closingRuntimeId }).catch(() => undefined)
        }

        await deleteSession(storedSessionId, removed?.profile)
        clearQueuedPrompts(storedSessionId)

        if (closingRuntimeId) {
          clearQueuedPrompts(closingRuntimeId)
        }
      } catch (err) {
        if (removed) {
          setSessions(prev => [removed, ...prev])
          setSessionsTotal(prev => prev + 1)
        }

        untombstoneSessions([storedSessionId, removed?.id, removed?._lineage_root_id])
        $pinnedSessionIds.set(previousPinned)

        if (wasSelected) {
          setFreshDraftReady(false)
          setSelectedStoredSessionId(storedSessionId)
          selectedStoredSessionIdRef.current = storedSessionId
          const stored = $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))

          if (stored) {
            setCurrentUsage(current => ({
              ...current,
              input: stored.input_tokens || 0,
              output: stored.output_tokens || 0,
              total: (stored.input_tokens || 0) + (stored.output_tokens || 0)
            }))
          }

          setMessages(previousMessages)
          navigate(sessionRoute(storedSessionId), { replace: true })

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
      selectedStoredSessionId,
      selectedStoredSessionIdRef,
      startFreshSessionDraft
    ]
  )

  const archiveSession = useCallback(
    async (storedSessionId: string) => {
      clearNotifications()

      const archived = $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))
      const wasSelected = selectedStoredSessionId === storedSessionId
      const previousPinned = $pinnedSessionIds.get()
      // Pins are keyed on the durable lineage-root id; the stored id may be the
      // live tip after compression. Drop both so the pin can't linger.
      const archivedPinId = archived ? sessionPinId(archived) : storedSessionId

      // Soft-hide: drop from the sidebar immediately, keep the data.
      setSessions(prev => prev.filter(session => !sessionMatchesStoredId(session, storedSessionId)))
      tombstoneSessions([storedSessionId, archived?.id, archived?._lineage_root_id])
      // Archived sessions are hidden by the listSessions(min_messages=1) query
      // on the next refresh, so they count as "removed" for the load-more
      // footer math.
      setSessionsTotal(prev => Math.max(0, prev - 1))
      $pinnedSessionIds.set(previousPinned.filter(id => id !== storedSessionId && id !== archivedPinId))

      if (wasSelected) {
        startFreshSessionDraft(true)
      }

      try {
        await setSessionArchived(storedSessionId, true, archived?.profile)
        // A sidebar refresh can race the optimistic removal while the PATCH is
        // in flight and briefly reinsert the still-unarchived backend row. Win
        // that race after the mutation succeeds so right-click → Archive does
        // not appear to do nothing until the next full refresh.
        setSessions(prev => prev.filter(session => !sessionMatchesStoredId(session, storedSessionId)))
        $pinnedSessionIds.set($pinnedSessionIds.get().filter(id => id !== storedSessionId && id !== archivedPinId))
        notify({ durationMs: 2_000, kind: 'success', message: copy.archived })
      } catch (err) {
        if (archived) {
          setSessions(prev => [archived, ...prev.filter(session => !sessionMatchesStoredId(session, storedSessionId))])
          setSessionsTotal(prev => prev + 1)
        }

        untombstoneSessions([storedSessionId, archived?.id, archived?._lineage_root_id])
        $pinnedSessionIds.set(previousPinned)
        notifyError(err, copy.archiveFailed)
      }
    },
    [copy, selectedStoredSessionId, startFreshSessionDraft]
  )

  return {
    archiveSession,
    branchCurrentSession,
    branchStoredSession,
    closeSettings,
    createBackendSessionForSend,
    openSettings,
    removeSession,
    resumeSession,
    selectSidebarItem,
    startFreshSessionDraft
  }
}
