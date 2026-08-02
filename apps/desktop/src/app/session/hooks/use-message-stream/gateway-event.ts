import type { BillingBlock } from '@hermes/shared'
import type { HermesSkin } from '@hermes/shared/skin'
import type { QueryClient } from '@tanstack/react-query'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { readActivePreview } from '@/app/chat/right-rail/preview-reader'
import { writeAgentTerminalChunk } from '@/app/right-sidebar/terminal/agent-terminal-stream'
import { readActiveTerminal } from '@/app/right-sidebar/terminal/buffer'
import { closeAgentTerminalByProc } from '@/app/right-sidebar/terminal/terminals'
import { burstVibeHearts } from '@/components/chat/vibe-hearts'
import { translateNow } from '@/i18n'
import { type GatewayEventPayload, textPart } from '@/lib/chat-messages'
import { coerceGatewayText, coerceThinkingText, normalizePersonalityValue } from '@/lib/chat-runtime'
import { playCompletionSound } from '@/lib/completion-sound'
import { approvalReplaySessionId, resolveGatewayEventSessionId } from '@/lib/gateway-events'
import { triggerHaptic } from '@/lib/haptics'
import { modelOptionsQueryKey } from '@/lib/model-options'
import { isProviderSetupErrorMessage } from '@/lib/provider-setup-errors'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { type AgentNoticePayload, clearAgentNotice, nativeNoticeInput, showAgentNotice } from '@/store/agent-notices'
import { reconcileApprovalModeForProfile } from '@/store/approval-mode'
import { billingCtaLabel, clearBillingBlock, runBillingRecovery, setBillingBlock } from '@/store/billing-block'
import { clearClarifyRequest, normalizeChoices, setClarifyRequest, warnDroppedChoices } from '@/store/clarify'
import { setSessionCompacting } from '@/store/compaction'
import { refreshBackgroundProcesses } from '@/store/composer-status'
import { $gateway } from '@/store/gateway'
import { applyGoalStatusText } from '@/store/goals'
import {
  notifyCronChanged,
  notifyPairingChanged,
  notifyPetChanged,
  notifyPlatformsChanged,
  notifySessionsChanged,
  type PetChangeMeta,
  setChangeEventsAvailable
} from '@/store/live-sync'
import { setMcpSetupRequest } from '@/store/mcp-setup'
import { dispatchNativeNotification } from '@/store/native-notifications'
import { isDiskFullErrorMessage, notify, notifyError } from '@/store/notifications'
import { requestDesktopOnboarding, requestDesktopOnboardingForCredentialWarning } from '@/store/onboarding'
import { revealDesktopPane } from '@/store/pane-focus'
import { flashPetActivity, markPetUnread, setPetActivity } from '@/store/pet'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { followActiveSessionCwd } from '@/store/projects'
import {
  clearAllPrompts,
  receiveApprovalRequest,
  replayPendingApproval,
  setSecretRequest,
  setSudoRequest
} from '@/store/prompts'
import { recordAgentReaction } from '@/store/reactions-local'
import {
  $currentCwd,
  $currentModel,
  $currentProvider,
  $selectedStoredSessionId,
  $sessions,
  sessionMatchesStoredId,
  setCurrentBranch,
  setCurrentCwdTransient,
  setCurrentFastMode,
  setCurrentPersonality,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setCurrentUsage,
  setMessages,
  setSessions,
  setTerminalBackend,
  setTurnStartedAt,
  setWorkspaceCwdOwner,
  setYoloActive
} from '@/store/session'
import { dropSessionState } from '@/store/session-states'
import { pruneDelegateFallbackSubagents, pruneFinishedSessionSubagents, upsertSubagent } from '@/store/subagents'
import { reportMcpToolResult } from '@/store/suggestion-providers/repair'
import { invalidateSkillSuggestionIndex } from '@/store/suggestion-providers/skill'
import { requestScrollToBottom } from '@/store/thread-scroll'
import { clearActiveSessionTodos } from '@/store/todos'
import { recordToolDiff } from '@/store/tool-diffs'
import { setSessionDraftingTool } from '@/store/tool-drafting'
import { reportInstallMethodWarning } from '@/store/updates'
import { notifyWorkspaceChanged, toolChangedPath, toolMayMutateFiles } from '@/store/workspace-events'
// Leaf import (not the `@/themes` barrel) to avoid pulling the ThemeProvider
// module graph into the gateway event hot path.
import { ingestBackendSkin } from '@/themes/backend-sync'
import type { RpcEvent } from '@/types/hermes'

import type { ClientSessionState } from '../../../types'
import { finalizeInterruptedMessages } from '../use-prompt-actions/rewind'

import { hasSessionInfoStatePatch, sessionInfoStatePatch, SUBAGENT_EVENT_TYPES, toTodoPayload } from './utils'

function firstBillingLine(text: string): string {
  return (text || '').split('\n')[0]?.trim() ?? ''
}

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
 * A turn failed on a billing wall (out of credits / payment required). The
 * gateway forwards the structured descriptor built by `agent/billing_links.py`;
 * we cache it per-session (drives the in-chat banner) AND raise one sticky,
 * billing-specific toast — never the generic "Hermes error" — with a smart CTA
 * (Nous → in-app Settings → Billing, other providers → their billing page).
 */
function surfaceBillingBlock(sessionId: string, raw: unknown): void {
  if (!raw || typeof raw !== 'object') {
    return
  }

  const block = raw as BillingBlock

  if (typeof block.provider !== 'string') {
    return
  }

  setBillingBlock(sessionId, block)

  const ctaCopy = {
    addCredits: translateNow('billingBlock.addCredits'),
    openBilling: translateNow('billingBlock.openBilling')
  }

  notify({
    // Collapse repeat walls from the same provider into one toast.
    id: `billing-block:${block.provider}`,
    kind: 'warning',
    icon: 'credit-card',
    title: block.is_nous
      ? translateNow('billingBlock.titleNous')
      : translateNow('billingBlock.titleProvider', block.provider_label),
    message: firstBillingLine(block.message) || translateNow('billingBlock.fallbackMessage'),
    // Sticky: a credit wall blocks every turn until resolved.
    durationMs: 0,
    action: { label: billingCtaLabel(block, ctaCopy), onClick: () => runBillingRecovery(block) }
  })
}

/**
 * Events that retire a "drafting a tool call" claim.
 *
 * `tool.generating` opens the claim and nothing closes it — a draft can be
 * abandoned without ever reaching `tool.start`, so enumerating the ways one
 * *ends* left the label on screen for the rest of the turn. Inverted: the
 * claim only covers what the model is emitting right now, and any other output
 * from the session proves it moved on. Same rule the TUI applies to its
 * transient trail lines (`turnController.pruneTransient`).
 */
const DRAFT_SUPERSEDING_EVENT_TYPES = new Set([
  'error',
  'message.complete',
  'message.delta',
  'message.start',
  'reasoning.delta',
  'thinking.delta',
  'tool.complete',
  'tool.progress',
  'tool.start'
])

const COMPACTION_RESUME_EVENT_TYPES = new Set([
  'message.delta',
  'message.interim',
  'thinking.delta',
  'reasoning.delta',
  'reasoning.available',
  'moa.reference',
  'moa.aggregating',
  'moa.progress',
  'moa.phase',
  'tool.start',
  'tool.progress',
  'tool.generating',
  'tool.complete'
])

interface GatewayEventDeps {
  activeGatewayProfile: string
  activeSessionIdRef: MutableRefObject<string | null>
  compactedTurnRef: MutableRefObject<Set<string>>
  lastCwdInfoSessionRef: MutableRefObject<string | null>
  nativeSubagentSessionsRef: MutableRefObject<Set<string>>
  appendAssistantDelta: (sessionId: string, delta: string) => void
  appendReasoningDelta: (sessionId: string, delta: string, replace?: boolean) => void
  completeAssistantMessage: (
    sessionId: string,
    text: string,
    responsePreviewed?: boolean,
    failure?: { error: string; partial: boolean }
  ) => void
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
    activeGatewayProfile,
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

      const replaySessionId = approvalReplaySessionId(event.type, activeSessionIdRef.current, sessionId)
      if (replaySessionId) {
        void replayPendingApproval($gateway.get(), replaySessionId).catch(() => undefined)
      }

      // Mid-turn compaction does not emit another message.start. The first
      // model output or tool event proves summarization has finished and the
      // turn has resumed, so retire the phase label without waiting for the
      // whole turn to complete.
      if (sessionId && COMPACTION_RESUME_EVENT_TYPES.has(event.type) && compactedTurnRef.current.has(sessionId)) {
        setSessionCompacting(sessionId, false)
      }

      if (sessionId && DRAFT_SUPERSEDING_EVENT_TYPES.has(event.type)) {
        setSessionDraftingTool(sessionId, '')
      }

      if (event.type === 'gateway.ready') {
        // Seed the active skin into the desktop theme registry without applying,
        // so a fresh connect never overrides the user's persisted desktop theme.
        ingestBackendSkin((payload as { skin?: HermesSkin } | undefined)?.skin, { apply: false })
        // Backends with the change watcher broadcast pet/cron/sessions change
        // events; consumers demote their legacy polls to slow backstops.
        setChangeEventsAvailable(Boolean((payload as { change_events?: boolean } | undefined)?.change_events))

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
      } else if (
        event.type === 'pet.changed' ||
        event.type === 'cron.changed' ||
        event.type === 'sessions.changed' ||
        event.type === 'platforms.changed' ||
        event.type === 'pairing.changed'
      ) {
        // Change-watcher broadcasts (server._broadcast_watched_changes): the
        // backend's on-disk signature moved. Route to the live-sync ticks the
        // former pollers now subscribe to. Only the active profile's changes
        // apply — background profile sockets watch their own homes.
        const fromActiveChangeProfile =
          !event.profile || normalizeProfileKey(event.profile) === normalizeProfileKey($activeGatewayProfile.get())

        if (fromActiveChangeProfile) {
          if (event.type === 'pet.changed') {
            notifyPetChanged(payload as PetChangeMeta | undefined)
          } else if (event.type === 'cron.changed') {
            notifyCronChanged()
          } else if (event.type === 'platforms.changed') {
            notifyPlatformsChanged()
          } else if (event.type === 'pairing.changed') {
            notifyPairingChanged()
          } else {
            notifySessionsChanged()
          }
        }

        return
      } else if (event.type === 'session.reclaimed') {
        // The backend reclaimed a live session we may still be holding (idle
        // TTL, LRU cap, or the WS-orphan reap). Without this the runtime id
        // stays cached until something fails against it, which reads as the
        // session vanishing rather than being reclaimed. Drop the cached state
        // now — the stored row is untouched, so the sidebar keeps the
        // conversation and reopening it resumes from the DB.
        const reclaimedRuntimeId = String((payload as { session_id?: string } | undefined)?.session_id ?? '')

        if (reclaimedRuntimeId) {
          dropSessionState(reclaimedRuntimeId)
        }

        // The row's ended_at moved, so refresh the lists that render it.
        notifySessionsChanged()

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
                // The turn is over but its streaming bubble may still say
                // pending — running=false from the agent loop's finally block
                // is the ONLY settle signal when message.complete never
                // arrives (turn crash, reconnect gap). Left pending, that
                // bubble shows a thinking indicator forever, stranded
                // mid-transcript once the next user message lands after it.
                // finalizeInterruptedMessages un-pends kept text and drops
                // empty placeholders; on the normal path message.complete
                // already settled everything and this is a no-op.
                messages: finalizeInterruptedMessages(state.messages, state.streamId),
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

        requestDesktopOnboardingForCredentialWarning(payload?.credential_warning)

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
            queryKey:
              explicitSid && sessionId ? modelOptionsQueryKey(activeGatewayProfile, sessionId) : ['model-options']
          })
        }
      } else if (event.type === 'message.start') {
        if (!sessionId) {
          return
        }

        flushQueuedDeltas(sessionId)
        pruneFinishedSessionSubagents(sessionId)
        setSessionCompacting(sessionId, false)
        compactedTurnRef.current.delete(sessionId)
        nativeSubagentSessionsRef.current.delete(sessionId)
        // A fresh turn on this session optimistically clears its billing wall;
        // if credits are still exhausted the next failure re-raises it.
        clearBillingBlock(sessionId)

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
          const text = `${header}\n${body}\n\n`

          if (idx === undefined || idx <= 1) {
            // First reference: clear any stale reasoning left over from
            // before this turn's references start, same as before.
            appendReasoningDelta(sessionId, text, true)
          } else {
            // Later references must accumulate, not replace — otherwise
            // each new reference wipes out the ones already shown (#64658).
            // Queue-then-flush (rather than the streamed/batched queue path)
            // applies it immediately, since each reference arrives as one
            // complete block rather than incremental tokens. reasoning.delta
            // cannot be mid-flight here: MoAChatCompletions.reference_callback
            // (agent/moa_loop.py) fires "moa.reference" once per reference's
            // already-complete text, with no concurrent token stream for the
            // reference-gathering phase, so there is no in-flight delta to
            // collide with in the shared queue bucket.
            appendReasoningDelta(sessionId, text, false)
            flushQueuedDeltas(sessionId)
          }
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
      } else if (event.type === 'moa.progress') {
        // Live reference fan-out progress ("refs k/n") — surfaced in the same
        // reasoning disclosure the references land in. These lines arrive
        // BEFORE any moa.reference event (references are only emitted once the
        // whole fan-out completes), and the first moa.reference replaces the
        // block, so the progress trail is self-cleaning.
        if (sessionId && typeof payload?.refs_done === 'number' && typeof payload?.refs_total === 'number') {
          const label = coerceGatewayText(payload?.label)

          const line = label
            ? `◇ MoA refs ${payload.refs_done}/${payload.refs_total} — ${label}\n`
            : `◇ MoA refs ${payload.refs_done}/${payload.refs_total}\n`

          appendReasoningDelta(sessionId, line, payload.refs_done <= 1)
          flushQueuedDeltas(sessionId)
        }

        if (isActiveEvent) {
          setPetActivity({ reasoning: true })
        }
      } else if (event.type === 'moa.phase') {
        // Phase transition — currently only phase="aggregator" (fan-out done,
        // aggregator acting). Append a one-line marker; the first
        // moa.reference that follows replaces the whole block.
        if (sessionId && payload?.phase === 'aggregator') {
          appendReasoningDelta(sessionId, '◇ MoA aggregating…\n', false)
          flushQueuedDeltas(sessionId)
        }

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

        // Terminal error frames (status "error") carry the failure in
        // structured fields: `error` is the message, and `partial` marks
        // `text` as streamed output to keep rather than the error string.
        const failure =
          payload?.status === 'error'
            ? {
                error: coerceGatewayText(payload.error).trim() || finalText || 'Hermes reported an error',
                partial: Boolean(payload.partial)
              }
            : undefined

        completeAssistantMessage(sessionId, finalText, payload?.response_previewed, failure)

        // Structured billing wall forwarded by the gateway (out of credits /
        // payment required) — cache it + raise a billing-specific toast.
        if (payload?.billing) {
          surfaceBillingBlock(sessionId, payload.billing)
        }

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
      } else if (event.type === 'tool.generating') {
        // Announced while the model is still emitting the call's JSON, so it
        // carries a name and nothing else — no id, no args. Materializing a row
        // from it strands an argless placeholder whenever the bubble is sealed
        // before the real `tool.start` arrives, because the two can no longer be
        // reconciled across the boundary. It's a status, so say it as one.
        // A stopped turn can still emit a frame or two before the backend
        // notices, and naming a tool we will never run leaves the label up
        // until something else retires it. `mutateStream` drops late tool rows
        // on the same condition; the status line has to agree with it.
        if (!sessionId || sessionInterrupted(sessionId)) {
          return
        }

        setSessionDraftingTool(sessionId, typeof payload?.name === 'string' ? payload.name : '')

        if (isActiveEvent) {
          setPetActivity({ reasoning: false, toolRunning: true })
        }
      } else if (event.type === 'tool.start' || event.type === 'tool.progress') {
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

        // The agent just created/deleted/renamed a skill, which adds or removes
        // its `/name` command. Drop the composer's cached `/` list so the new
        // skill is offerable now rather than after the hour-long TTL — and the
        // skill-suggestion provider's index with it.
        if (payload?.name === 'skill_manage') {
          invalidateSlashCompletions()
          invalidateSkillSuggestionIndex()
        }

        // MCP tool outcomes feed the connection-repair suggestion provider:
        // an auth/connection-shaped failure offers a reconnect pill; a later
        // success against the same server withdraws it.
        if (sessionId && typeof payload?.name === 'string' && payload.name.startsWith('mcp__')) {
          reportMcpToolResult(
            sessionId,
            payload.name,
            Boolean(payload.error),
            [payload.error, payload.result].filter(part => typeof part === 'string').join(' ')
          )
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
        const rawChoices = payload?.choices
        const choices = normalizeChoices(rawChoices)
        const multiSelect = payload?.multi_select === true

        if (requestId && question) {
          if (rawChoices != null && choices.length === 0) {
            warnDroppedChoices('gateway', question, rawChoices)
          }

          setClarifyRequest({
            requestId,
            question,
            choices: choices.length > 0 ? choices : null,
            multiSelect,
            sessionId: sessionId ?? null
          })

          if (sessionId) {
            // `clarify.request` is the blocking event the Python side waits on,
            // while the inline UI normally mounts from the earlier `tool.start`
            // row. If that row was missed (stream reconnect / hydration race) the
            // sidebar still says "needs input" but there is nowhere to render the
            // choices. Upsert a stable pending clarify tool row from the request
            // itself so the prompt stays answerable; a real tool.start/complete
            // with the same request id merges rather than duplicates.
            upsertToolCall(
              sessionId,
              {
                args: { choices, ...(multiSelect ? { multi_select: true } : {}), question },
                name: 'clarify',
                tool_id: requestId
              },
              'running'
            )

            // The transcript only renders the active session, so a background
            // clarify is otherwise invisible (the row just keeps spinning like
            // it's working). Flag the session so the sidebar shows a persistent
            // "needs input" indicator on its row — works for the active session
            // too, and survives alt-tab / window blur (unlike a toast).
            updateSessionState(sessionId, state => ({ ...state, needsInput: true }))

            if (sessionId === activeSessionIdRef.current) {
              requestScrollToBottom()
            }
          }

          dispatchNativeNotification({
            body: question,
            kind: 'input',
            sessionId,
            title: translateNow('notifications.native.inputTitle')
          })
        }
      } else if (event.type === 'mcp.setup.request') {
        // setup_mcp tool (desktop GUI): the agent proposed an MCP server and
        // the Python side is blocked on mcp.setup.respond. Park the request
        // per-session (like clarify) and upsert a stable pending tool row so
        // the inline consent card has somewhere to render even when the
        // tool.start event was missed (stream reconnect / hydration race).
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''
        const server = typeof payload?.server === 'string' ? payload.server : ''
        const rawAction = typeof payload?.action === 'string' ? payload.action : 'install'
        const action = rawAction === 'enable' || rawAction === 'authorize' ? rawAction : 'install'
        const reason = typeof payload?.reason === 'string' ? payload.reason : ''

        if (requestId && server) {
          setMcpSetupRequest({ action, reason, requestId, server, sessionId: sessionId ?? null })

          if (sessionId) {
            upsertToolCall(
              sessionId,
              { args: { action, reason, server }, name: 'setup_mcp', tool_id: requestId },
              'running'
            )
            updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
          }

          dispatchNativeNotification({
            body: reason || server,
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

        void receiveApprovalRequest($gateway.get(), {
          // false only when a tirith warning forbids it; backend omits the field otherwise.
          allowPermanent: payload?.allow_permanent !== false,
          choices: Array.isArray(payload?.choices)
            ? payload.choices.filter(choice => typeof choice === 'string')
            : undefined,
          command,
          description,
          requestId: typeof payload?.request_id === 'string' ? payload.request_id : undefined,
          sessionId: sessionId ?? null,
          smartDenied: payload?.smart_denied === true
        }).catch(() => undefined)

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
      } else if (event.type === 'preview.read.request') {
        // read_preview tool: serialize the active preview tab (a Browser
        // webview's page text is async) and answer. Empty text = nothing open.
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

        if (requestId) {
          const start = typeof payload?.start === 'number' ? payload.start : undefined
          const count = typeof payload?.count === 'number' ? payload.count : undefined

          void readActivePreview({ count, start }).then(result => {
            void $gateway.get()?.request('preview.read.respond', {
              request_id: requestId,
              text: result ? JSON.stringify(result) : ''
            })
          })
        }
      } else if (event.type === 'window.read.request') {
        // read_window_below tool: main owns native window enumeration, so ask
        // it over IPC and answer. Empty text = unavailable (no bridge, or
        // enumeration unsupported on this system e.g. Wayland).
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

        if (requestId) {
          const read = window.hermesDesktop?.readWindowBelow

          const answer = (result: unknown) =>
            $gateway.get()?.request('window.read.respond', {
              request_id: requestId,
              text: result ? JSON.stringify(result) : ''
            })

          // .catch: ipcRenderer.invoke rejects on an older shell without the
          // handler or a main-side throw — without an empty answer the tool
          // would stall its full 30s timeout.
          void Promise.resolve(read ? read() : null).then(answer, () => answer(null))
        }
      } else if (event.type === 'agent.terminal.output') {
        // Live chunk from a background process → its read-only agent terminal tab.
        writeAgentTerminalChunk(payload?.process_id ?? '', payload?.chunk ?? '')
      } else if (event.type === 'terminal.close') {
        // Agent closed its own read-only tab via the desktop-gated close_terminal tool.
        // The process is untouched — this only drops the view.
        closeAgentTerminalByProc(payload?.process_id ?? '')
      } else if (event.type === 'pane.reveal') {
        // Agent revealed a pane via the desktop-gated focus_pane tool, in
        // response to an explicit user request. Active session only — a
        // background turn must never move the user's focus (desktop AGENTS.md:
        // offer, don't hijack).
        if (isActiveEvent) {
          revealDesktopPane(payload?.pane ?? '')
        }
      } else if (event.type === 'message.reaction') {
        // The agent reacted to a message via the desktop-gated
        // react_to_message tool. Already persisted — this only paints it now
        // instead of at the next resume. Fresh ChatMessage object per change:
        // the runtime repository caches normalized ThreadMessages in a WeakMap
        // keyed by ChatMessage identity.
        const reactedRowId = payload?.row_id

        if (typeof reactedRowId === 'number') {
          const nextReactions = Array.isArray(payload?.reactions) ? payload.reactions : []
          const reactedRole = payload?.role === 'assistant' ? 'assistant' : 'user'

          setMessages(messages => {
            // Preferred leg: the message already knows its durable row id
            // (rehydrated transcript, or a live row that has round-tripped).
            const byRowId = messages.find(message => message.rowId === reactedRowId)

            if (byRowId) {
              // Overlay survives the end-of-turn resume, which rebuilds from
              // in-memory history that doesn't carry this mid-turn DB write.
              recordAgentReaction(reactedRowId, nextReactions)

              return messages.map(message =>
                message.rowId === reactedRowId ? { ...message, reactions: nextReactions } : message
              )
            }

            // Live leg: the targeted message is still optimistic (no rowId —
            // it hasn't round-tripped through a resume). The agent's default
            // target is the newest message of that role, so stamp the reaction
            // AND the now-known row id onto it. Without this the event matches
            // nothing and the reaction only appears after a reload.
            const lastIndex = messages.findLastIndex(
              message => message.role === reactedRole && message.rowId === undefined
            )

            if (lastIndex === -1) {
              return messages
            }

            recordAgentReaction(reactedRowId, nextReactions)

            return messages.map((message, index) =>
              index === lastIndex ? { ...message, rowId: reactedRowId, reactions: nextReactions } : message
            )
          })
        }
      } else if (event.type === 'status.update') {
        if (sessionId && payload?.kind === 'compacting') {
          setSessionCompacting(sessionId, true)
          compactedTurnRef.current.add(sessionId)
        } else if (sessionId && payload?.kind === 'compacted') {
          setSessionCompacting(sessionId, false)
          compactedTurnRef.current.delete(sessionId)
        } else if (sessionId && payload?.kind === 'process') {
          // The gateway's notification poller announces background process
          // completions / watch matches here — re-sync the status stack.
          void refreshBackgroundProcesses(sessionId)
        } else if (sessionId && payload?.kind === 'goal') {
          applyGoalStatusText(sessionId, coerceGatewayText(payload?.text))
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
        //
        // Typed here with the `review:` marker (same convention as `steer:` /
        // `slash:`) so SystemMessage can paint it as the memory-write row it
        // is instead of sniffing the backend's prose. The leading 💾 goes with
        // it — the row draws its own glyph.
        const text = coerceGatewayText(payload?.text)
          .trim()
          .replace(/^[^\p{L}\p{N}]+/u, '')

        if (text && sessionId) {
          flushQueuedDeltas(sessionId)
          updateSessionState(sessionId, state => ({
            ...state,
            messages: [
              ...state.messages,
              {
                id: `review-summary-${Date.now()}`,
                role: 'system',
                parts: [textPart(`review:${text}`)],
                timestamp: Math.floor(Date.now() / 1000)
              }
            ]
          }))
        }
      } else if (event.type === 'notification.show') {
        // Driver-agnostic agent notice (credits usage/grant/depleted/restored
        // from `agent/credits_tracker.py`). The Ink TUI renders these in its
        // status bar; the desktop renders them as toasts. The notice key doubles
        // as the toast id, so the escalating 50→75→90 credits line replaces in
        // place instead of stacking. Account-wide signal — shown regardless of
        // which session is focused.
        const notice = event.payload as AgentNoticePayload | undefined

        showAgentNotice(notice)

        // The urgent pair (access paused / restored) also breaks through as a
        // native OS notification when Hermes is backgrounded; dispatch is gated
        // by the user's notification prefs + backgrounded check.
        const native = nativeNoticeInput(notice, translateNow('notifications.native.creditsTitle'))

        if (native) {
          dispatchNativeNotification(native)
        }

        // A credits crossing moves the account balance. Settings → Billing polls
        // `billing.state` every 30s; nudge it so the page reflects the crossing
        // immediately instead of up to 30s late.
        if (notice?.key?.startsWith('credits.')) {
          void queryClient.invalidateQueries({ queryKey: ['billing', 'state'] })
        }
      } else if (event.type === 'notification.clear') {
        // Key-matched dismissal (e.g. credits restored clears the depleted
        // notice). notify() keys the toast by the notice key, so this maps
        // straight to dismissNotification(key).
        clearAgentNotice((event.payload as AgentNoticePayload | undefined)?.key)
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
        } else if (isDiskFullErrorMessage(errorMessage)) {
          notifyError(new Error(errorMessage), translateNow('notifications.errors.diskFull'))
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
      activeGatewayProfile,
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
