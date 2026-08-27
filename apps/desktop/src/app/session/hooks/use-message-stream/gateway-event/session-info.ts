import { normalizePersonalityValue } from '@/lib/chat-runtime'
import { modelOptionsQueryKey } from '@/lib/model-options'
import { reconcileApprovalModeForProfile } from '@/store/approval-mode'
import { requestDesktopOnboardingForCredentialWarning } from '@/store/onboarding'
import { followActiveSessionCwd } from '@/store/projects'
import {
  $activeSessionId,
  $currentCwd,
  $currentModel,
  $currentProvider,
  $selectedStoredSessionId,
  $sessions,
  sessionMatchesStoredId,
  setActiveSessionId,
  setCurrentBranch,
  setCurrentCwdTransient,
  setCurrentFastMode,
  setCurrentPersonality,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setCurrentUsage,
  setSessions,
  setTerminalBackend,
  setWorkspaceCwdOwner,
  setYoloActive
} from '@/store/session'
import { reportInstallMethodWarning } from '@/store/updates'

import { finalizeInterruptedMessages } from '../../use-prompt-actions/rewind'
import { hasSessionInfoStatePatch, PRE_TURN_LIVE_SETTLE_GRACE_MS, sessionInfoStatePatch } from '../utils'

import type { GatewayEventContext } from './types'

/**
 * Whether a `session.info` payload's `stored_session_id` may be treated as the
 * selected conversation's, so its cwd can be claimed for it (#71254).
 *
 * Absent is not the same as different: the backend omits the id on a
 * not-yet-built (`lazy`) session, and refusing there would leave the workspace
 * marked un-owned for the rest of the conversation. Matching goes through the
 * lineage (`sessionMatchesStoredId`) so a compression-rotated tip and the root
 * a pinned-row selection may hold still read as one conversation.
 */
function sessionInfoDescribesSelectedSession(storedSessionId: string | undefined): boolean {
  const infoStoredSessionId = storedSessionId?.trim() || null
  const selected = $selectedStoredSessionId.get() ?? null

  if (!infoStoredSessionId) {
    return true
  }

  // A named session cannot describe a fresh draft. Treating a null selection as
  // a wildcard let a background tile's `session.info` rehome the draft to the
  // tile's workspace.
  if (!selected) {
    return false
  }

  if (infoStoredSessionId === selected) {
    return true
  }

  // Either id may be the live tip or the lineage root, so ask whether ONE row
  // answers to both rather than assuming which side rotated.
  return $sessions
    .get()
    .some(session => sessionMatchesStoredId(session, infoStoredSessionId) && sessionMatchesStoredId(session, selected))
}

/**
 * Adopt a rebuilt runtime back into the open pane (#93942 scenario B).
 *
 * A mid-conversation model/provider switch rebuilds the agent runtime: the
 * new runtime emits `session.info` (and every later event) under a NEW
 * explicit session_id while the pane still holds the dead one as its active
 * id — so `isActiveEvent` is false for the same conversation and the view
 * stops receiving live updates until a full resume. When an incoming
 * `session.info` lineage-matches the selected conversation but carries a
 * different runtime id, and the OLD runtime shows no live turn (not busy,
 * not streaming), re-bind: adopt the new id as the active session id,
 * keeping the durable selection untouched. A live turn on the old runtime
 * (overlap window during a manual switch) refuses the adoption.
 */
function maybeRebindPaneToRebuiltRuntime(ctx: GatewayEventContext): boolean {
  const { deps, explicitSid, isActiveEvent, payload } = ctx

  if (!explicitSid || isActiveEvent || typeof payload?.stored_session_id !== 'string') {
    return false
  }

  const selected = $selectedStoredSessionId.get()

  if (!selected || !sessionInfoDescribesSelectedSession(payload.stored_session_id)) {
    return false
  }

  const activeId = $activeSessionId.get()
  const oldState = activeId ? deps.sessionStateByRuntimeIdRef.current.get(activeId) : undefined

  // Only a dead old runtime may be adopted over: hijacking a streaming turn
  // would split one conversation's events across two panes.
  if (oldState?.busy || oldState?.awaitingResponse || oldState?.streamId) {
    return false
  }

  setActiveSessionId(explicitSid)
  ctx.deps.activeSessionIdRef.current = explicitSid

  return true
}

/** session.info / session.usage / session.title. */
export function handleSessionInfoEvent(ctx: GatewayEventContext): boolean {
  const { deps, event, payload, sessionId, explicitSid, isActiveEvent, occurredAt, fromActiveSource } = ctx

  const {
    activeGatewayProfile,
    activeSessionIdRef,
    hydrateFromStoredSession,
    lastCwdInfoSessionRef,
    queryClient,
    scheduleSessionsRefresh,
    sessionStateByRuntimeIdRef,
    updateSessionState
  } = deps

  if (event.type === 'session.info') {
    // A rebuilt runtime (mid-conversation model/provider switch) speaks under
    // a NEW session_id. Before scoping anything by isActiveEvent, check
    // whether this event is the rebuilt runtime announcing itself for the
    // conversation already on screen — if so, re-bind the pane so every
    // subsequent isActiveEvent gate keeps matching (#93942 scenario B).
    const rebound = maybeRebindPaneToRebuiltRuntime(ctx)

    // Apply session-scoped fields when the event targets the active
    // session, OR when it's a global broadcast and we have no session.
    const apply = (explicitSid ? isActiveEvent : !activeSessionIdRef.current) || rebound
    const statePatch = sessionInfoStatePatch(payload)
    const hasStatePatch = hasSessionInfoStatePatch(statePatch)
    const modelChanged = typeof payload?.model === 'string'
    const providerChanged = typeof payload?.provider === 'string'
    const runningChanged = typeof payload?.running === 'boolean'
    // The backend stamps model/provider (as strings) on EVERY session.info,
    // so the presence flags above are true on every heartbeat/turn edge —
    // fine for the cheap atom writes below (nanostores skips identical
    // values), but they also drove queryClient.invalidateQueries, refetching
    // the model-options provider catalog once or twice per turn for a model
    // that never changed. Only a genuine VALUE change (vs the session's own
    // cached runtime state, captured before the state patch below applies;
    // composer atoms as the fallback for an uncached session) invalidates.
    const knownState = sessionId ? sessionStateByRuntimeIdRef.current.get(sessionId) : undefined
    const modelValueChanged = modelChanged && payload!.model !== (knownState?.model ?? $currentModel.get())

    const providerValueChanged =
      providerChanged && payload!.provider !== (knownState?.provider ?? $currentProvider.get())

    // Config is profile-scoped, but session.info also arrives for background
    // sessions. Only an active-session event from the currently active
    // gateway may reconcile the foreground cache. Requiring the renderer's
    // source tag prevents an event queued before a profile swap from being
    // attributed to the newly active profile.
    if (isActiveEvent && typeof payload?.approval_mode === 'string' && event.profile && fromActiveSource()) {
      reconcileApprovalModeForProfile(event.profile, payload.approval_mode)
    }

    if (apply) {
      // Do not call setCurrentModel / setCurrentProvider here. Composer
      // model/provider is sticky UI state (localStorage + manual picks).
      // Periodic session.info heartbeats often carry the profile default
      // (or a stale session model) and would silently revert the dropdown.
      // Active-session model/provider still flows through the session state
      // cache via updateSessionState → syncRuntimeMetadataToView below.

      if (typeof payload?.cwd === 'string' && sessionInfoDescribesSelectedSession(payload.stored_session_id)) {
        // The active session's agent can relocate itself (new repo/worktree
        // via the terminal). When the SAME active session's cwd actually
        // moves, follow it — refresh the project tree + scope so the sidebar
        // tracks the live thread. A fresh selection (different session id)
        // is a switch, not a move, so it refreshes data without yanking scope.
        const cwdMoved = payload.cwd !== $currentCwd.get()
        const sameSession = !!sessionId && sessionId === lastCwdInfoSessionRef.current

        lastCwdInfoSessionRef.current = sessionId
        setCurrentCwdTransient(payload.cwd)

        // The backend just confirmed the selected conversation's real
        // workspace, so it owns the path we wrote. Without the claim the
        // marker keeps naming whoever held it before — including the
        // released state a detached resume leaves behind — and the primary
        // workspace-derived surfaces stay hidden against a folder the
        // backend has confirmed (#71254).
        setWorkspaceCwdOwner($selectedStoredSessionId.get())

        if (cwdMoved && sameSession) {
          void followActiveSessionCwd(payload.cwd)
        }
      }

      if (typeof payload?.branch === 'string') {
        setCurrentBranch(payload.branch)
      }

      if (typeof payload?.terminal_backend === 'string') {
        setTerminalBackend(payload.terminal_backend)
      }

      if (typeof payload?.personality === 'string') {
        setCurrentPersonality(normalizePersonalityValue(payload.personality))
      }

      if (typeof payload?.reasoning_effort === 'string') {
        setCurrentReasoningEffort(payload.reasoning_effort)
      }

      if (typeof payload?.service_tier === 'string') {
        setCurrentServiceTier(payload.service_tier)
      }

      if (typeof payload?.fast === 'boolean') {
        setCurrentFastMode(payload.fast)
      }

      if (typeof payload?.yolo === 'boolean') {
        setYoloActive(payload.yolo)
      }
    }

    if (sessionId && hasStatePatch) {
      updateSessionState(
        sessionId,
        state => ({
          ...state,
          ...statePatch,
          branch: statePatch.branch ?? state.branch,
          cwd: statePatch.cwd ?? state.cwd
        }),
        payload?.stored_session_id || undefined
      )
    }

    // The running→busy transition must reach EVERY session, not just the
    // active one. The `apply` gate above correctly scopes view-only side
    // effects (setCurrentCwd, etc.) to the focused chat,
    // but the per-session busy state is what drives the sidebar working
    // indicator — a background session's turn start/finish must update
    // its dot without the user opening it. updateSessionState only
    // mutates the per-runtime cache entry, and syncSessionStateToView
    // guards the view publish to the active session, so this is safe.
    if (runningChanged && sessionId) {
      // Set when THIS event released a turn that ended without ever
      // producing an assistant payload, so the catch-up side effects below
      // run on that edge only. The updater is invoked exactly once,
      // synchronously, by updateSessionState.
      let recoveredWithoutPayload = false

      const nextState = updateSessionState(
        sessionId,
        state => {
          const busy = Boolean(payload!.running)

          if (state.busy === busy && (busy || !state.awaitingResponse)) {
            return state
          }

          if (busy) {
            // Don't re-arm busy from a stale session.info if the user
            // just clicked Stop (interrupted=true). The backend's
            // cooperative interrupt may not have propagated yet, so
            // running is still true in the heartbeat. The turn's
            // finally block will emit running=false to clear busy.
            if (state.interrupted) {
              return state
            }

            // Prefer the gateway-reported turn_started_at so the timer
            // survives session switches and session.info heartbeats.
            const gatewayTurnStartedAt =
              typeof payload!.turn_started_at === 'number' && payload!.turn_started_at > 0
                ? payload!.turn_started_at * 1000
                : null

            return {
              ...state,
              busy,
              // running=true from the backend is turn-live proof, same as
              // message.start (e.g. resuming an already-running session
              // that never replays its start event).
              turnLive: true,
              turnStartedAt: state.turnStartedAt ?? gatewayTurnStartedAt ?? Date.now()
            }
          }

          // The turn has not started backend-side yet. submit arms
          // busy/awaitingResponse optimistically, so a running=false
          // heartbeat that lands in the gap before the turn spins up is a
          // pre-start report, not a finished turn — settling on it would
          // drop the spinner and re-open the send guard mid-flight.
          // turnLive is stamped only once the backend reports the turn
          // live (message.start, the running=true edge, or a resumed
          // in-flight turn) and is cleared by every settle, so false
          // here is exactly "no turn has been reported running yet".
          // (turnStartedAt can't discriminate — it is optimistically
          // seeded at submit so the visible timer starts at Enter.)
          //
          // BOUNDED (#86795): an armed turn that never goes live — a
          // restore/edit whose rewind was refused after the optimistic
          // arm, a submit response lost to a gateway bounce, a terminal
          // error event that never arrived — would otherwise hold this
          // gate forever. busy then latches until app restart:
          // isTargetSessionBusy refuses every send, the composer queues
          // each message, and the queue drain (gated on busy→false) never
          // fires. turnStartedAt is seeded at the optimistic arm, so its
          // age bounds the hold; past the grace window (or with no clock
          // at all) the gateway's running=false is authoritative and the
          // settle below releases the session.
          const armedAt = state.turnStartedAt

          const withinPreStartGrace =
            typeof armedAt === 'number' && Date.now() - armedAt < PRE_TURN_LIVE_SETTLE_GRACE_MS

          if (state.awaitingResponse && !state.sawAssistantPayload && !state.turnLive && withinPreStartGrace) {
            return state
          }

          // Past that gate the turn DID start and the backend now reports it
          // finished. When no assistant payload ever arrived (gateway crash
          // mid-stream, provider error before the first delta, agent-build
          // failure) message.complete never fires, so this is the only event
          // that can release the session. Bailing here instead left
          // awaitingResponse/busy latched until app restart (#46517): the
          // per-session busy flag is authoritative for isTargetSessionBusy,
          // so submitPrompt and the slash dispatcher silently returned false
          // and the session accepted no further input.
          recoveredWithoutPayload = state.awaitingResponse && !state.sawAssistantPayload

          return {
            ...state,
            awaitingResponse: false,
            busy,
            // The turn is over but its streaming bubble may still say
            // pending — running=false from the agent loop's finally block
            // is the ONLY settle signal when message.complete never
            // arrives (turn crash, reconnect gap). Left pending, that
            // bubble shows a thinking indicator forever, stranded
            // mid-transcript once the next user message lands after it.
            // finalizeInterruptedMessages un-pends kept text and drops
            // empty placeholders; on the normal path message.complete
            // already settled everything and this is a no-op.
            messages: finalizeInterruptedMessages(state.messages, state.streamId, occurredAt),
            pendingBranchGroup: null,
            streamId: null,
            turnStartedAt: null,
            turnLive: false
          }
        },
        payload?.stored_session_id || undefined
      )

      if (recoveredWithoutPayload) {
        // Stays unscoped, like the settle above: a background session's
        // sidebar row has to drop its working dot without the user opening
        // it. This fires on the recovery edge only — once awaitingResponse
        // is false the `state.busy === busy` guard above short-circuits
        // every later heartbeat — so it costs one coalesced refresh per
        // broken turn, not one per tick.
        scheduleSessionsRefresh()

        // The transcript catch-up IS scoped. The stream died, but the turn
        // itself may have completed and been persisted, so refetch stored
        // history for the session actually on screen; a background session
        // reads its history when the user opens it, and hydrating every one
        // of them here would fan a REST call out per idle session.
        if (isActiveEvent) {
          void hydrateFromStoredSession(3, nextState.storedSessionId, sessionId)
        }
      }
    }

    if (payload?.usage && (!explicitSid || isActiveEvent)) {
      setCurrentUsage(current => ({ ...current, ...payload.usage }))
    }

    requestDesktopOnboardingForCredentialWarning(payload?.credential_warning)

    if (apply) {
      reportInstallMethodWarning(payload?.install_warning)
      // Config refetch is only meaningful for the foreground context —
      // everything refreshHermesConfig applies is either active-session
      // guarded or a composer/global pref. Background sessions' heartbeats
      // used to trigger it too (two REST calls each, every turn).
      ctx.scheduleConfigRefresh()
    }

    if (modelValueChanged || providerValueChanged) {
      void queryClient.invalidateQueries({
        queryKey: explicitSid && sessionId ? modelOptionsQueryKey(activeGatewayProfile, sessionId) : ['model-options']
      })
    }

    return true
  }

  if (event.type === 'session.usage') {
    // Live usage tick emitted while a turn is mid-flight (see tui_gateway
    // _start_usage_ticker) so the status-bar context window tracks growth
    // during the turn instead of only jumping at message.complete.
    if (payload?.usage && sessionId) {
      // Per-session twin first: a focused secondary tile reads this cache,
      // while the primary-only global mirrors the active session.
      updateSessionState(sessionId, state => ({
        ...state,
        usage: { calls: 0, input: 0, output: 0, total: 0, ...state.usage, ...payload.usage }
      }))

      if (isActiveEvent) {
        setCurrentUsage(current => ({ ...current, ...payload.usage }))
      }
    }

    return true
  }

  if (event.type === 'session.title') {
    // Live auto-title push (titler runs async, after the turn's refresh).
    const storedId = typeof payload?.session_id === 'string' ? payload.session_id : ''
    const nextTitle = typeof payload?.title === 'string' ? payload.title.trim() : ''

    if (storedId && nextTitle) {
      setSessions(prev => prev.map(s => (sessionMatchesStoredId(s, storedId) ? { ...s, title: nextTitle } : s)))
    }

    return true
  }

  return false
}
