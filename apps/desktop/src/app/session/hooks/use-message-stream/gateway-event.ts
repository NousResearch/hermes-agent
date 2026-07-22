import type { HermesSkin } from '@hermes/shared/skin'
import type { QueryClient } from '@tanstack/react-query'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { writeAgentTerminalChunk } from '@/app/right-sidebar/terminal/agent-terminal-stream'
import { readActiveTerminal } from '@/app/right-sidebar/terminal/buffer'
import { closeAgentTerminalByProc } from '@/app/right-sidebar/terminal/terminals'
import { burstVibeHearts } from '@/components/chat/vibe-hearts'
import { translateNow } from '@/i18n'
import { type GatewayEventPayload, textPart } from '@/lib/chat-messages'
import { coerceGatewayText, coerceThinkingText, normalizePersonalityValue } from '@/lib/chat-runtime'
import { playCompletionSound } from '@/lib/completion-sound'
import { resolveGatewayEventSessionId } from '@/lib/gateway-events'
import { triggerHaptic } from '@/lib/haptics'
import { isProviderSetupErrorMessage } from '@/lib/provider-setup-errors'
import { reconcileApprovalModeForProfile } from '@/store/approval-mode'
import { clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { setSessionCompacting } from '@/store/compaction'
import { refreshBackgroundProcesses } from '@/store/composer-status'
import { $gateway } from '@/store/gateway'
import { dispatchNativeNotification } from '@/store/native-notifications'
import { notify } from '@/store/notifications'
import { requestDesktopOnboarding } from '@/store/onboarding'
import { flashPetActivity, markPetUnread, setPetActivity } from '@/store/pet'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { followActiveSessionCwd } from '@/store/projects'
import { clearAllPrompts, setApprovalRequest, setSecretRequest, setSudoRequest } from '@/store/prompts'
import {
  $currentCwd,
  $currentModel,
  $currentProvider,
  sessionMatchesStoredId,
  setCurrentBranch,
  setCurrentCwd,
  setCurrentFastMode,
  setCurrentPersonality,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setCurrentUsage,
  setSessions,
  setTurnStartedAt,
  setYoloActive
} from '@/store/session'
import { clearSessionSubagents, pruneDelegateFallbackSubagents, upsertSubagent } from '@/store/subagents'
import { clearActiveSessionTodos } from '@/store/todos'
import { recordToolDiff } from '@/store/tool-diffs'
import { reportInstallMethodWarning } from '@/store/updates'
import { notifyWorkspaceChanged, toolChangedPath, toolMayMutateFiles } from '@/store/workspace-events'
// Leaf import (not the `@/themes` barrel) to avoid pulling the ThemeProvider
// module graph into the gateway event hot path.
import { ingestBackendSkin } from '@/themes/backend-sync'
import type { RpcEvent } from '@/types/hermes'

import type { ClientSessionState } from '../../../types'

import { hasSessionInfoStatePatch, sessionInfoStatePatch, SUBAGENT_EVENT_TYPES, toTodoPayload } from './utils'

const COMPACTION_RESUME_EVENT_TYPES = new Set([
  'message.delta',
  'message.interim',
  'thinking.delta',
  'reasoning.delta',
  'reasoning.available',
  'moa.reference',
  'moa.aggregating',
  'tool.start',
  'tool.progress',
  'tool.generating',
  'tool.complete'
])

interface GatewayEventDeps {
  activeSessionIdRef: MutableRefObject<string | null>
  compactedTurnRef: MutableRefObject<Set<string>>
  lastCwdInfoSessionRef: MutableRefObject<string | null>
  nativeSubagentSessionsRef: MutableRefObject<Set<string>>
  appendAssistantDelta: (sessionId: string, delta: string) => void
  appendReasoningDelta: (sessionId: string, delta: string, replace?: boolean) => void
  completeAssistantMessage: (sessionId: string, text: string, responsePreviewed?: boolean) => void
  failAssistantMessage: (sessionId: string, errorMessage: string) => void
  flushQueuedDeltas: (sessionId?: string) => void
  finalizeInterimAssistantMessage: (sessionId: string, text: string) => void
  queryClient: QueryClient
  refreshHermesConfig: () => Promise<void>
  sessionInterrupted: (sessionId: string) => boolean
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
  upsertToolCall: (
    sessionId: string,
    payload: GatewayEventPayload | undefined,
    phase: 'running' | 'complete',
    sourceEventType?: string
  ) => void
}

/** The gateway-event dispatcher, extracted from useMessageStream. */
export function useGatewayEventHandler(deps: GatewayEventDeps) {
  const {
    appendAssistantDelta,
    appendReasoningDelta,
    activeSessionIdRef,
    compactedTurnRef,
    lastCwdInfoSessionRef,
    nativeSubagentSessionsRef,
    completeAssistantMessage,
    failAssistantMessage,
    flushQueuedDeltas,
    finalizeInterimAssistantMessage,
    queryClient,
    refreshHermesConfig,
    sessionInterrupted,
    sessionStateByRuntimeIdRef,
    updateSessionState,
    upsertToolCall
  } = deps

  const unscopedStreamSessionIdRef = useRef<string | null>(null)

  // session.info arrives in bursts (agent build ready + turn end + title /
  // MCP / compress edges within the same second). Each used to fire its own
  // refreshHermesConfig — two REST calls (config + defaults) per event, per
  // turn, including for BACKGROUND sessions whose values the fetch can't even
  // apply. Coalesce to one trailing fetch per burst; the caller gates on
  // `apply` so background traffic doesn't schedule anything.
  const configRefreshTimerRef = useRef<null | number>(null)

  const scheduleConfigRefresh = useCallback(() => {
    if (configRefreshTimerRef.current !== null) {
      return
    }

    if (typeof window === 'undefined') {
      void refreshHermesConfig()

      return
    }

    configRefreshTimerRef.current = window.setTimeout(() => {
      configRefreshTimerRef.current = null
      void refreshHermesConfig()
    }, 300)
  }, [refreshHermesConfig])

  useEffect(
    () => () => {
      if (configRefreshTimerRef.current !== null && typeof window !== 'undefined') {
        window.clearTimeout(configRefreshTimerRef.current)
        configRefreshTimerRef.current = null
      }
    },
    []
  )

  return useCallback(
    (event: RpcEvent) => {
      const payload = event.payload as GatewayEventPayload | undefined
      const explicitSid = event.session_id || ''

      const route = resolveGatewayEventSessionId({
        activeSessionId: activeSessionIdRef.current,
        eventType: event.type,
        explicitSessionId: explicitSid,
        unscopedStreamSessionId: unscopedStreamSessionIdRef.current
      })

      unscopedStreamSessionIdRef.current = route.nextUnscopedStreamSessionId

      if (route.drop) {
        return
      }

      const sessionId = route.sessionId
      const isActiveEvent = !!sessionId && sessionId === activeSessionIdRef.current

      // Mid-turn compaction does not emit another message.start. The first
      // model output or tool event proves summarization has finished and the
      // turn has resumed, so retire the phase label without waiting for the
      // whole turn to complete.
      if (sessionId && COMPACTION_RESUME_EVENT_TYPES.has(event.type) && compactedTurnRef.current.has(sessionId)) {
        setSessionCompacting(sessionId, false)
      }

      if (event.type === 'gateway.ready') {
        // Seed the active skin into the desktop theme registry without applying,
        // so a fresh connect never overrides the user's persisted desktop theme.
        ingestBackendSkin((payload as { skin?: HermesSkin } | undefined)?.skin, { apply: false })

        return
      } else if (event.type === 'skin.changed') {
        // A runtime skin switch (Hermes activating an authored skin, or `/skin`
        // on another surface). Only the active profile's change repaints.
        const fromActiveProfile =
          !event.profile || normalizeProfileKey(event.profile) === normalizeProfileKey($activeGatewayProfile.get())

        if (fromActiveProfile) {
          ingestBackendSkin(payload as HermesSkin | undefined, { apply: true })
        }

        return
      } else if (event.type === 'session.info') {
        // Apply session-scoped fields when the event targets the active
        // session, OR when it's a global broadcast and we have no session.
        const apply = explicitSid ? isActiveEvent : !activeSessionIdRef.current
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
        if (
          isActiveEvent &&
          typeof payload?.approval_mode === 'string' &&
          event.profile &&
          normalizeProfileKey(event.profile) === normalizeProfileKey($activeGatewayProfile.get())
        ) {
          reconcileApprovalModeForProfile(event.profile, payload.approval_mode)
        }

        if (apply) {
          // Do not call setCurrentModel / setCurrentProvider here. Composer
          // model/provider is sticky UI state (localStorage + manual picks).
          // Periodic session.info heartbeats often carry the profile default
          // (or a stale session model) and would silently revert the dropdown.
          // Active-session model/provider still flows through the session state
          // cache via updateSessionState → syncRuntimeMetadataToView below.

          if (typeof payload?.cwd === 'string') {
            // The active session's agent can relocate itself (new repo/worktree
            // via the terminal). When the SAME active session's cwd actually
            // moves, follow it — refresh the project tree + scope so the sidebar
            // tracks the live thread. A fresh selection (different session id)
            // is a switch, not a move, so it refreshes data without yanking scope.
            const cwdMoved = payload.cwd !== $currentCwd.get()
            const sameSession = !!sessionId && sessionId === lastCwdInfoSessionRef.current

            lastCwdInfoSessionRef.current = sessionId
            setCurrentCwd(payload.cwd)

            if (cwdMoved && sameSession) {
              void followActiveSessionCwd(payload.cwd)
            }
          }

          if (typeof payload?.branch === 'string') {
            setCurrentBranch(payload.branch)
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
          updateSessionState(
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

                return {
                  ...state,
                  busy,
                  turnStartedAt: state.turnStartedAt ?? Date.now()
                }
              }

              if (state.awaitingResponse && !state.sawAssistantPayload) {
                return state
              }

              return {
                ...state,
                awaitingResponse: false,
                busy,
                pendingBranchGroup: null,
                streamId: null,
                turnStartedAt: null
              }
            },
            payload?.stored_session_id || undefined
          )
        }

        if (payload?.usage && (!explicitSid || isActiveEvent)) {
          setCurrentUsage(current => ({ ...current, ...payload.usage }))
        }

        if (typeof payload?.credential_warning === 'string' && payload.credential_warning) {
          requestDesktopOnboarding(payload.credential_warning)
        }

        if (apply) {
          reportInstallMethodWarning(payload?.install_warning)
          // Config refetch is only meaningful for the foreground context —
          // everything refreshHermesConfig applies is either active-session
          // guarded or a composer/global pref. Background sessions' heartbeats
          // used to trigger it too (two REST calls each, every turn).
          scheduleConfigRefresh()
        }

        if (modelValueChanged || providerValueChanged) {
          void queryClient.invalidateQueries({
            queryKey: explicitSid && sessionId ? ['model-options', sessionId] : ['model-options']
          })
        }
      } else if (event.type === 'message.start') {
        if (!sessionId) {
          return
        }

        flushQueuedDeltas(sessionId)
        clearSessionSubagents(sessionId)
        setSessionCompacting(sessionId, false)
        compactedTurnRef.current.delete(sessionId)
        nativeSubagentSessionsRef.current.delete(sessionId)

        if (isActiveEvent) {
          triggerHaptic('streamStart')
        }

        updateSessionState(sessionId, state => {
          // If the user clicked Stop (cancelRun set interrupted=true), don't
          // let a stale message.start from a chained turn (goal follow-up,
          // completion drain) or an in-flight LLM response re-arm busy.
          // The interrupt is user intent — the backend's cooperative cancel
          // may not have propagated yet, so its events are stale. The turn's
          // finally block will emit session.info with running=false to clear
          // busy for real once the agent loop actually exits.
          if (state.interrupted) {
            return state
          }

          return {
            ...state,
            busy: true,
            awaitingResponse: true,
            sawAssistantPayload: false,
            interrupted: false,
            interimBoundaryPending: false,
            turnStartedAt: Date.now()
          }
        })

        if (isActiveEvent) {
          setTurnStartedAt(Date.now())
        }
      } else if (event.type === 'message.delta') {
        if (sessionId) {
          appendAssistantDelta(sessionId, coerceGatewayText(payload?.text))
        }
      } else if (event.type === 'message.interim') {
        // The agent emitted interim assistant commentary (text alongside tool
        // calls, or the attempted final answer before a verify-on-stop nudge).
        // Finalize it as its own sealed bubble so message.complete doesn't wipe
        // it — the text was already streamed via message.delta and is visible.
        if (sessionId) {
          flushQueuedDeltas(sessionId)
          const text = coerceGatewayText(payload?.text)

          if (text) {
            finalizeInterimAssistantMessage(sessionId, text)
          }
        }
      } else if (event.type === 'thinking.delta') {
        // thinking.delta carries the kawaii spinner status (face + verb from
        // KawaiiSpinner), not real reasoning. The bottom-of-thread loading
        // indicator already covers that UX, so we ignore these events to
        // avoid a duplicative "Thinking" disclosure showing spinner text.
      } else if (event.type === 'reaction') {
        // Core-detected affection (ily / <3 / good bot) on the user's message.
        // Play hearts only for the visible session so background turns stay quiet.
        if (isActiveEvent && (payload?.kind ?? 'vibe') === 'vibe') {
          burstVibeHearts()
        }
      } else if (event.type === 'reasoning.delta') {
        if (sessionId) {
          appendReasoningDelta(sessionId, coerceThinkingText(payload?.text))
        }

        if (isActiveEvent) {
          setPetActivity({ reasoning: true })
        }
      } else if (event.type === 'reasoning.available') {
        if (sessionId) {
          appendReasoningDelta(sessionId, coerceThinkingText(payload?.text), true)
        }

        if (isActiveEvent) {
          setPetActivity({ reasoning: true })
        }
      } else if (event.type === 'moa.reference') {
        // MoA reference-model output — surface as a labelled thinking chunk
        // (tagged with the source model) before the aggregator's response, so
        // the mixture-of-agents process is visible. Reuses the reasoning
        // disclosure rather than introducing a parallel surface.
        if (sessionId) {
          const label = coerceGatewayText(payload?.label) || 'reference'
          const idx = typeof payload?.index === 'number' ? payload.index : undefined
          const cnt = typeof payload?.count === 'number' ? payload.count : undefined
          const header = idx && cnt ? `◇ Reference ${idx}/${cnt} — ${label}` : `◇ Reference — ${label}`
          const body = coerceThinkingText(payload?.text)
          appendReasoningDelta(sessionId, `${header}\n${body}\n\n`, true)
        }

        if (isActiveEvent) {
          setPetActivity({ reasoning: true })
        }
      } else if (event.type === 'moa.aggregating') {
        // Status transition only; the aggregator's reply arrives via the normal
        // message stream. No reasoning/transcript mutation here.
        if (isActiveEvent) {
          setPetActivity({ reasoning: true })
        }
      } else if (event.type === 'message.complete') {
        if (!sessionId) {
          return
        }

        // Turn ended — drop any blocking prompt still open for THIS session
        // (e.g. interrupted, or the approval already resolved). Scoped to the
        // session so a background turn finishing can't wipe the active chat's
        // prompt, and vice versa.
        clearAllPrompts(sessionId)
        clearClarifyRequest(undefined, sessionId)
        // Turn ended without a final `todo` update — drop a still-unfinished
        // list so "Tasks N/M" doesn't stay pinned above the composer with the
        // last item stuck pending/in_progress. Finished lists keep their linger.
        clearActiveSessionTodos(sessionId)
        setSessionCompacting(sessionId, false)

        flushQueuedDeltas(sessionId)

        // Keyed by session so only one window beeps when several are open.
        playCompletionSound(sessionId)

        const finalText = coerceGatewayText(payload?.text) || coerceGatewayText(payload?.rendered)
        completeAssistantMessage(sessionId, finalText, payload?.response_previewed)

        if (isActiveEvent) {
          setTurnStartedAt(null)

          // Pet beat: a finished turn always celebrates — go straight to the
          // jump, never linger on the run/reason pose. One atom update (clears
          // toolRunning/reasoning AND sets celebrate together) so no stray "run"
          // frame leaks to the sprite — including the popped-out overlay, which
          // mirrors each activity change. The jump runs ~2 loops, then settles.
          flashPetActivity({ celebrate: true, reasoning: false, toolRunning: false }, 2200)

          // Light up the pet's mail icon if the user wasn't looking when the turn
          // finished — a glanceable "new message" hint on the popped-out overlay.
          // Cleared when they open the app via the mail icon or refocus the window.
          if (typeof document !== 'undefined' && !document.hasFocus()) {
            markPetUnread()
          }
        }

        if (payload?.usage) {
          // Per-session twin FIRST (the statusbar reads it for focused tiles);
          // the primary-only global mirrors the ACTIVE session — ungated it
          // let a background tile's turn overwrite the primary's count.
          updateSessionState(sessionId, state => ({
            ...state,
            usage: { calls: 0, input: 0, output: 0, total: 0, ...state.usage, ...payload.usage }
          }))

          if (isActiveEvent) {
            setCurrentUsage(current => ({ ...current, ...payload.usage }))
          }
        }
      } else if (event.type === 'session.title') {
        // Live auto-title push (titler runs async, after the turn's refresh).
        const storedId = typeof payload?.session_id === 'string' ? payload.session_id : ''
        const nextTitle = typeof payload?.title === 'string' ? payload.title.trim() : ''

        if (storedId && nextTitle) {
          setSessions(prev => prev.map(s => (sessionMatchesStoredId(s, storedId) ? { ...s, title: nextTitle } : s)))
        }
      } else if (event.type === 'tool.start' || event.type === 'tool.progress' || event.type === 'tool.generating') {
        if (!sessionId) {
          return
        }

        flushQueuedDeltas(sessionId)
        upsertToolCall(sessionId, toTodoPayload(payload) ?? payload, 'running', event.type)

        if (isActiveEvent) {
          setPetActivity({ reasoning: false, toolRunning: true })
        }
      } else if (event.type === 'tool.complete') {
        if (sessionId) {
          flushQueuedDeltas(sessionId)
          upsertToolCall(sessionId, toTodoPayload(payload) ?? payload, 'complete', event.type)

          if (isActiveEvent) {
            setPetActivity({ toolRunning: false })
          }

          // A pending clarify blocks the turn, so the first tool.complete after
          // one is the clarify resolving — drop the "needs input" flag here so
          // the sidebar indicator clears as soon as it's answered, not only at
          // message.complete.
          updateSessionState(sessionId, state => (state.needsInput ? { ...state, needsInput: false } : state))

          // terminal/process tool calls are the only things that spawn or reap
          // background processes — sync the composer status stack right after.
          if (!sessionInterrupted(sessionId) && (payload?.name === 'terminal' || payload?.name === 'process')) {
            void refreshBackgroundProcesses(sessionId)
          }
        }

        if (typeof payload?.inline_diff === 'string' && payload.inline_diff.trim()) {
          recordToolDiff(payload.tool_id || payload.name || '', payload.inline_diff)
        }

        // A file-mutating tool just finished — nudge the git-mirroring surfaces
        // (coding rail, review pane, file tree) to refresh. Event-driven, not
        // polled: fires exactly when the agent touches the tree.
        if (payload && toolMayMutateFiles(payload)) {
          notifyWorkspaceChanged(toolChangedPath(payload))
        }
      } else if (SUBAGENT_EVENT_TYPES.has(event.type)) {
        if (sessionId && payload && !sessionInterrupted(sessionId)) {
          if (!nativeSubagentSessionsRef.current.has(sessionId)) {
            pruneDelegateFallbackSubagents(sessionId)
          }

          nativeSubagentSessionsRef.current.add(sessionId)
          upsertSubagent(
            sessionId,
            payload as Record<string, unknown>,
            event.type === 'subagent.spawn_requested' || event.type === 'subagent.start',
            event.type
          )
        }
      } else if (event.type === 'clarify.request') {
        // Surface the clarify tool's overlay. The Python side is blocked on
        // `clarify.respond`, so without this handler the agent would hang
        // forever (see tools/clarify_tool.py + tui_gateway/server.py:_block).
        //
        // Store the request for whichever session raised it — even a background
        // one. clarify.request is a one-shot event; if we dropped it for an
        // unfocused session, that session would block on `clarify.respond`
        // indefinitely and re-focusing it could never recover (the event is
        // gone). Parking it per-session lets the user answer once they switch
        // over; the inline ClarifyTool reads the active session's entry.
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''
        const question = typeof payload?.question === 'string' ? payload.question : ''

        if (requestId && question) {
          setClarifyRequest({
            requestId,
            question,
            choices: Array.isArray(payload?.choices) ? payload!.choices!.filter(c => typeof c === 'string') : null,
            sessionId: sessionId ?? null
          })

          // The transcript only renders the active session, so a background
          // clarify is otherwise invisible (the row just keeps spinning like
          // it's working). Flag the session so the sidebar shows a persistent
          // "needs input" indicator on its row — works for the active session
          // too, and survives alt-tab / window blur (unlike a toast).
          if (sessionId) {
            updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
          }

          dispatchNativeNotification({
            body: question,
            kind: 'input',
            sessionId,
            title: translateNow('notifications.native.inputTitle')
          })
        }
      } else if (event.type === 'approval.request') {
        // Dangerous-command / execute_code approval. The Python side is blocked
        // in _await_gateway_decision() until approval.respond lands; without
        // this the agent stalls until its 5-min timeout and the tool is BLOCKED.
        // Park it per-session (like clarify) so a *background* profile's turn can
        // raise it and wait — the sidebar flags "needs input" and the inline bar
        // surfaces once the user focuses that chat.
        const command = typeof payload?.command === 'string' ? payload.command : ''
        const description = typeof payload?.description === 'string' ? payload.description : 'dangerous command'

        setApprovalRequest({
          // false only when a tirith warning forbids it; backend omits the field otherwise.
          allowPermanent: payload?.allow_permanent !== false,
          choices: Array.isArray(payload?.choices)
            ? payload.choices.filter(choice => typeof choice === 'string')
            : undefined,
          command,
          description,
          sessionId: sessionId ?? null,
          smartDenied: payload?.smart_denied === true
        })

        if (sessionId) {
          updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
        }

        dispatchNativeNotification({
          actions: [
            { id: 'approve', text: translateNow('notifications.native.approveAction') },
            { id: 'reject', text: translateNow('notifications.native.rejectAction') }
          ],
          body: command || description,
          kind: 'approval',
          sessionId,
          title: translateNow('notifications.native.approvalTitle')
        })
      } else if (event.type === 'sudo.request') {
        // Sudo password capture (tools/terminal_tool.py). Blocked on
        // sudo.respond {request_id, password}.
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

        if (requestId) {
          setSudoRequest({ requestId, sessionId: sessionId ?? null })

          if (sessionId) {
            updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
          }

          dispatchNativeNotification({
            body: translateNow('notifications.native.inputBody'),
            kind: 'input',
            sessionId,
            title: translateNow('notifications.native.inputTitle')
          })
        }
      } else if (event.type === 'secret.request') {
        // Skill credential capture (tools/skills_tool.py). Blocked on
        // secret.respond {request_id, value}.
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

        if (requestId) {
          const envVar = typeof payload?.env_var === 'string' ? payload.env_var : ''
          const promptText = typeof payload?.prompt === 'string' ? payload.prompt : ''

          setSecretRequest({
            requestId,
            envVar,
            prompt: promptText,
            sessionId: sessionId ?? null
          })

          if (sessionId) {
            updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
          }

          dispatchNativeNotification({
            body: promptText || envVar || translateNow('notifications.native.inputBody'),
            kind: 'input',
            sessionId,
            title: translateNow('notifications.native.inputTitle')
          })
        }
      } else if (event.type === 'terminal.read.request') {
        // read_terminal tool: serialize the renderer's xterm buffer and answer
        // immediately (Python blocks on the respond). Empty text = no live pane.
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

        if (requestId) {
          const start = typeof payload?.start === 'number' ? payload.start : undefined
          const count = typeof payload?.count === 'number' ? payload.count : undefined
          const result = readActiveTerminal({ start, count })

          void $gateway.get()?.request('terminal.read.respond', {
            request_id: requestId,
            text: result ? JSON.stringify(result) : ''
          })
        }
      } else if (event.type === 'agent.terminal.output') {
        // Live chunk from a background process → its read-only agent terminal tab.
        writeAgentTerminalChunk(payload?.process_id ?? '', payload?.chunk ?? '')
      } else if (event.type === 'terminal.close') {
        // Agent closed its own read-only tab via the desktop-gated close_terminal tool.
        // The process is untouched — this only drops the view.
        closeAgentTerminalByProc(payload?.process_id ?? '')
      } else if (event.type === 'status.update') {
        if (sessionId && payload?.kind === 'compacting') {
          setSessionCompacting(sessionId, true)
          compactedTurnRef.current.add(sessionId)
        } else if (sessionId && payload?.kind === 'process') {
          // The gateway's notification poller announces background process
          // completions / watch matches here — re-sync the status stack.
          void refreshBackgroundProcesses(sessionId)
        }
      } else if (event.type === 'review.summary') {
        // Self-improvement background review saved something to memory/skills
        // and emitted a persistent summary (Python formats it as
        // "💾 Self-improvement review: …"). The CLI prints this via
        // prompt_toolkit and the Ink TUI renders it as a system line; the
        // desktop has neither, so without this handler the skill/memory
        // change happens silently. Surface it as a persistent system message
        // in the transcript so the user is always informed — it must not be a
        // transient toast that can be missed.
        const text = coerceGatewayText(payload?.text).trim()

        if (text && sessionId) {
          flushQueuedDeltas(sessionId)
          updateSessionState(sessionId, state => ({
            ...state,
            messages: [
              ...state.messages,
              {
                id: `review-summary-${Date.now()}`,
                role: 'system',
                parts: [textPart(text)],
                timestamp: Math.floor(Date.now() / 1000)
              }
            ]
          }))
        }
      } else if (event.type === 'error') {
        const errorMessage = payload?.message || 'Hermes reported an error'
        const looksLikeProviderSetup = isProviderSetupErrorMessage(errorMessage)

        // A turn that errors out has also ended — drop any open blocking prompt
        // for this session so an approval/sudo/secret overlay can't linger past
        // the failed turn (same intent as the message.complete clear).
        if (sessionId) {
          clearAllPrompts(sessionId)
          clearClarifyRequest(undefined, sessionId)
          clearActiveSessionTodos(sessionId)
          setSessionCompacting(sessionId, false)
          compactedTurnRef.current.delete(sessionId)
        }

        if (isActiveEvent) {
          setPetActivity({ reasoning: false, toolRunning: false })
          flashPetActivity({ error: true })
        }

        dispatchNativeNotification({
          body: errorMessage,
          kind: 'turnError',
          sessionId,
          title: translateNow('notifications.native.turnErrorTitle')
        })

        if (looksLikeProviderSetup) {
          requestDesktopOnboarding(errorMessage)
        } else {
          // Toast globally, not just when the failing thread is focused: a
          // turn-ending error (e.g. out of funds) blocks every thread, so the
          // inline error alone is too easy to miss. The stable id collapses the
          // same error from multiple blocked threads into one toast.
          notify({
            id: `gateway-error:${errorMessage}`,
            kind: 'error',
            title: 'Hermes error',
            message: errorMessage
          })
        }

        if (sessionId) {
          flushQueuedDeltas(sessionId)
          failAssistantMessage(sessionId, errorMessage)
        }

        if (isActiveEvent) {
          setTurnStartedAt(null)
        }
      }
    },
    [
      appendAssistantDelta,
      appendReasoningDelta,
      activeSessionIdRef,
      compactedTurnRef,
      completeAssistantMessage,
      failAssistantMessage,
      finalizeInterimAssistantMessage,
      flushQueuedDeltas,
      lastCwdInfoSessionRef,
      nativeSubagentSessionsRef,
      queryClient,
      scheduleConfigRefresh,
      sessionInterrupted,
      sessionStateByRuntimeIdRef,
      updateSessionState,
      upsertToolCall
    ]
  )
}
