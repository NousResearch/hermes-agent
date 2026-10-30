import { type BillingBlock, mergeUsage } from '@hermes/shared'

import { noteTurnSettled } from '@/app/session/hooks/use-session-actions/settled-turns'
import { burstVibeHearts } from '@/components/chat/vibe-hearts'
import { translateNow } from '@/i18n'
import type { GatewayEventPayload } from '@/lib/chat-messages'
import { coerceGatewayText, coerceThinkingText } from '@/lib/chat-runtime'
import { playCompletionSound } from '@/lib/completion-sound'
import { parseErrorSurface } from '@/lib/error-surface'
import { triggerHaptic } from '@/lib/haptics'
import { billingCtaLabel, clearBillingBlock, runBillingRecovery, setBillingBlock } from '@/store/billing-block'
import { clearSettledClarifyRequest } from '@/store/clarify'
import { setSessionCompacting } from '@/store/compaction'
import { noteFreeTierTurnComplete } from '@/store/free-tier-sign-in'
import { notify } from '@/store/notifications'
import { flashPetActivity, markPetUnread, setPetActivity } from '@/store/pet'
import { clearAllPrompts } from '@/store/prompts'
import { providerWaitText, setSessionProviderWait } from '@/store/provider-wait'
import { setCurrentUsage, setTurnStartedAt } from '@/store/session'
import { refreshSupportedSessionControlAfterTurn } from '@/store/session-control'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $sessionStates } from '@/store/session-states'
import { lightDesktopMessagingCompletion } from '@/store/session-unread'
import { pruneFinishedSessionSubagents } from '@/store/subagents'
import { clearActiveSessionTodos } from '@/store/todos'

import { segmentStamp } from '../segment-identity'

import type { GatewayEventContext } from './types'

function firstBillingLine(text: string): string {
  return (text || '').split('\n')[0]?.trim() ?? ''
}

/**
 * A turn failed on a billing wall (out of credits / payment required). The
 * gateway forwards the structured descriptor built by `agent/billing_links.py`;
 * we cache it per-session, which drives the out-of-credits banner above the
 * composer. A wall in a session that is NOT on screen also raises one sticky,
 * billing-specific toast — never the generic "Hermes error" — with a smart CTA
 * (Nous → in-app Settings → Billing, other providers → their billing page); on
 * screen the banner alone carries it (one event, one surface).
 */
function surfaceBillingBlock(sessionId: string, raw: unknown, toast: boolean): void {
  if (!raw || typeof raw !== 'object') {
    return
  }

  const block = raw as BillingBlock

  if (typeof block.provider !== 'string') {
    return
  }

  setBillingBlock(sessionId, block)

  if (!toast) {
    return
  }

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

/** Terminal error frames (status "error") carry the failure in structured
 *  fields: `error` is the message, `partial` marks `text` as streamed output to
 *  keep rather than the error string, and `error_surface` (newer gateways)
 *  names the failing layer for the card. */
function turnFailure(payload: GatewayEventPayload | undefined, finalText: string) {
  if (payload?.status !== 'error') {
    return undefined
  }

  return {
    error: coerceGatewayText(payload.error).trim() || finalText || 'Hermes reported an error',
    partial: Boolean(payload.partial),
    surface: parseErrorSurface(payload.error_surface)
  }
}

function appendMoaReference(ctx: GatewayEventContext, sessionId: string): void {
  const { deps, payload, occurredAt } = ctx
  const label = coerceGatewayText(payload?.label) || 'reference'
  const idx = typeof payload?.index === 'number' ? payload.index : undefined
  const cnt = typeof payload?.count === 'number' ? payload.count : undefined
  const header = idx && cnt ? `◇ Reference ${idx}/${cnt} — ${label}` : `◇ Reference — ${label}`
  const body = coerceThinkingText(payload?.text)
  const text = `${header}\n${body}\n\n`

  if (idx === undefined || idx <= 1) {
    // First reference: clear any stale reasoning left over from
    // before this turn's references start, same as before.
    deps.appendReasoningDelta(sessionId, text, true, occurredAt)

    return
  }

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
  deps.appendReasoningDelta(sessionId, text, false, occurredAt)
  deps.flushQueuedDeltas(sessionId)
}

/** The message/reasoning/MoA streaming family: message.start → deltas →
 *  interim → complete, thinking/reasoning deltas, moa.* progress, reaction. */
/** The stamped completion of a messaging turn Desktop sent decides its unread dot. */
function lightOwnMessagingTurn(runtimeId: string, admissionId: string | undefined, profile?: string) {
  const storedId = $sessionStates.get()[runtimeId]?.storedSessionId ?? runtimeId
  lightDesktopMessagingCompletion(storedId, admissionId, storedId === $focusedStoredSessionId.get(), profile)
}

/** message.start: arm the session for a new turn, unless it re-sends the current one. */
function handleMessageStart(ctx: GatewayEventContext): boolean {
  const { deps, payload, sessionId, isActiveEvent } = ctx
  const { compactedTurnRef, flushQueuedDeltas, dropQueuedDeltas, nativeSubagentSessionsRef } = deps
  const { sessionStateByRuntimeIdRef, updateSessionState } = deps

  if (!sessionId) {
    return true
  }

  const turnId = typeof payload?.turn_id === 'string' ? payload.turn_id : ''

  // A re-sent start of the turn this session already has (reconnect replay,
  // a superseded attempt) is not a new occurrence: resetting here would let
  // the turn's re-sent segment paint as a second copy (#144609).
  if (turnId && sessionStateByRuntimeIdRef.current.get(sessionId)?.streamTurnId === turnId) {
    return true
  }

  // Turn-boundary orphan drop (#119543): when no turn is live, anything
  // still queued belongs to a turn that already ended (a delta reordered
  // behind its own complete or heartbeat). Flushing it would seed a bubble
  // the new turn inherits, painting a stale duplicate of the previous
  // reply. A still-live previous turn (steer) keeps the flush: those bytes
  // are real output of the bubble on screen.
  if (sessionStateByRuntimeIdRef.current.get(sessionId)?.turnLive) {
    flushQueuedDeltas(sessionId)
  } else {
    dropQueuedDeltas(sessionId)
  }

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

  // Submit→accept latency: seedOptimistic armed the clock at Enter; this
  // event is the backend accepting the turn. Debug-only visibility into
  // how long the arm actually took (the "no progress box for seconds"
  // complaint) — reads the pre-update cache, costs nothing when clean.
  const seededAt = sessionStateByRuntimeIdRef.current.get(sessionId)?.turnStartedAt

  if (typeof seededAt === 'number') {
    console.debug('[turn-accept-latency]', { sessionId, ms: Date.now() - seededAt })
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
      // Backend accepted the turn — the no-payload settle gate below may
      // now treat a running=false heartbeat as a real turn end.
      turnLive: true,
      // A new turn is a new occurrence: the previous turn's late terminal
      // frame (#119569) can no longer claim its heartbeat-settled bubble.
      heartbeatSettledStreamId: null,
      streamTurnId: turnId || null,
      // Keep the submit-time seed (submit.ts seedOptimistic) — resetting
      // here would hide the submit→accept round trip from the timer.
      // Backend-originated turns (queue drain elsewhere, goal follow-up)
      // have no seed and arm here.
      turnStartedAt: state.turnStartedAt ?? Date.now()
    }
  })

  if (isActiveEvent) {
    // Belt-and-suspenders mirror of the ACTIVE session's per-session
    // clock (the load-bearing mirror is the view-sync flush in
    // use-session-state-cache). Mirror the seeded value, not Date.now():
    // resetting to accept-time here would visibly snap the timer back
    // after the submit-time seed above already started it.
    setTurnStartedAt(sessionStateByRuntimeIdRef.current.get(sessionId)?.turnStartedAt ?? Date.now())
  }

  return true
}

// health: allow CC -- usage-window limit: one guard so the error card, not a second billing toast, says it; this event switch was over target on main
export function handleMessageStreamEvent(ctx: GatewayEventContext): boolean {
  const { deps, event, payload, sessionId, isActiveEvent, occurredAt } = ctx

  const {
    appendAssistantDelta,
    appendReasoningDelta,
    compactedTurnRef,
    completeAssistantMessage,
    finalizeInterimAssistantMessage,
    flushQueuedDeltas,
    dropQueuedDeltas,
    nativeSubagentSessionsRef,
    sessionStateByRuntimeIdRef,
    updateSessionState
  } = deps

  if (event.type === 'message.start') {
    return handleMessageStart(ctx)
  }

  if (event.type === 'message.delta') {
    if (sessionId) {
      appendAssistantDelta(sessionId, coerceGatewayText(payload?.text), occurredAt, segmentStamp(payload))
    }

    return true
  }

  if (event.type === 'message.interim') {
    // The agent emitted interim assistant commentary (text alongside tool
    // calls, or the attempted final answer before a verify-on-stop nudge).
    // Finalize it as its own sealed bubble so message.complete doesn't wipe
    // it — the text was already streamed via message.delta and is visible.
    if (sessionId) {
      flushQueuedDeltas(sessionId)
      const text = coerceGatewayText(payload?.text)

      if (text) {
        finalizeInterimAssistantMessage(sessionId, text, occurredAt, segmentStamp(payload))
      }
    }

    return true
  }

  if (event.type === 'thinking.delta') {
    // Most thinking.delta frames are kawaii spinner rewrites and stay out
    // of the transcript. Explained provider waits are different: the core
    // emits them after prolonged silence, so name that wait in the existing
    // bottom-of-thread status row instead of leaving only an unlabeled timer.
    if (sessionId) {
      setSessionProviderWait(sessionId, providerWaitText(coerceGatewayText(payload?.text)))
    }

    return true
  }

  if (event.type === 'reaction') {
    // Core-detected affection (ily / <3 / good bot) on the user's message.
    // Play hearts only for the visible session so background turns stay quiet.
    if (isActiveEvent && (payload?.kind ?? 'vibe') === 'vibe') {
      burstVibeHearts()
    }

    return true
  }

  if (event.type === 'reasoning.delta') {
    if (sessionId) {
      appendReasoningDelta(sessionId, coerceThinkingText(payload?.text), false, occurredAt)
    }

    if (isActiveEvent) {
      setPetActivity({ reasoning: true })
    }

    return true
  }

  if (event.type === 'reasoning.available') {
    if (sessionId) {
      appendReasoningDelta(sessionId, coerceThinkingText(payload?.text), true, occurredAt)
    }

    if (isActiveEvent) {
      setPetActivity({ reasoning: true })
    }

    return true
  }

  if (event.type === 'moa.reference') {
    // MoA reference-model output — surface as a labelled thinking chunk
    // (tagged with the source model) before the aggregator's response, so
    // the mixture-of-agents process is visible. Reuses the reasoning
    // disclosure rather than introducing a parallel surface.
    if (sessionId) {
      appendMoaReference(ctx, sessionId)
    }

    if (isActiveEvent) {
      setPetActivity({ reasoning: true })
    }

    return true
  }

  if (event.type === 'moa.aggregating') {
    // Status transition only; the aggregator's reply arrives via the normal
    // message stream. No reasoning/transcript mutation here.
    if (isActiveEvent) {
      setPetActivity({ reasoning: true })
    }

    return true
  }

  if (event.type === 'moa.progress') {
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

      appendReasoningDelta(sessionId, line, payload.refs_done <= 1, occurredAt)
      flushQueuedDeltas(sessionId)
    }

    if (isActiveEvent) {
      setPetActivity({ reasoning: true })
    }

    return true
  }

  if (event.type === 'moa.phase') {
    // Phase transition — currently only phase="aggregator" (fan-out done,
    // aggregator acting). Append a one-line marker; the first
    // moa.reference that follows replaces the whole block.
    if (sessionId && payload?.phase === 'aggregator') {
      appendReasoningDelta(sessionId, '◇ MoA aggregating…\n', false, occurredAt)
      flushQueuedDeltas(sessionId)
    }

    if (isActiveEvent) {
      setPetActivity({ reasoning: true })
    }

    return true
  }

  if (event.type === 'message.complete') {
    if (!sessionId) {
      return true
    }

    // Turn ended — drop any blocking prompt still open for THIS session
    // (e.g. interrupted, or the approval already resolved). Scoped to the
    // session so a background turn finishing can't wipe the active chat's
    // prompt, and vice versa.
    clearAllPrompts(sessionId)
    clearSettledClarifyRequest(sessionId)
    // Turn ended without a final `todo` update — drop a still-unfinished
    // list so "Tasks N/M" doesn't stay pinned above the composer with the
    // last item stuck pending/in_progress. Finished lists keep their linger.
    clearActiveSessionTodos(sessionId)
    setSessionCompacting(sessionId, false)

    flushQueuedDeltas(sessionId)

    // Keyed by session so only one window beeps when several are open.
    playCompletionSound(sessionId)
    lightOwnMessagingTurn(sessionId, event.admission_id, event.profile)

    const finalText = coerceGatewayText(payload?.text) || coerceGatewayText(payload?.rendered)

    // The view mirrors this runtime only while bound to it (settled-turns.ts).
    noteTurnSettled(sessionId, deps.activeSessionIdRef.current === sessionId)
    completeAssistantMessage(
      sessionId,
      finalText,
      payload?.response_previewed,
      turnFailure(payload, finalText),
      occurredAt,
      payload?.persisted_turn,
      Boolean(payload?.response_transformed),
      typeof payload?.status === 'string' ? payload.status : undefined,
      payload?.response_reused === true
    )

    if (payload?.status === 'complete') {
      noteFreeTierTurnComplete()
    }

    // Structured billing wall forwarded by the gateway (out of credits /
    // payment required) — cache it for the banner (toast only off screen). A window
    // plan's limit is said once, by the turn's error card (its buttons are the
    // ways on); a toast and banner would only repeat it.
    const usageWindowWall = parseErrorSurface(payload?.error_surface)?.code === 'usage_window'

    if (payload?.billing && !usageWindowWall) {
      surfaceBillingBlock(sessionId, payload.billing, !isActiveEvent)
    }

    // History-commit note (e.g. a mid-turn desync) the gateway chose to surface.
    if (typeof payload?.warning === 'string' && payload.warning.trim()) {
      notify({ kind: 'warning', message: payload.warning })
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
        usage: mergeUsage({ calls: 0, input: 0, output: 0, total: 0, ...state.usage }, payload.usage)
      }))

      if (isActiveEvent) {
        setCurrentUsage(current => mergeUsage(current, payload.usage))
      }
    }

    // Refresh only the structured-control sessions already proven capable.
    // Initial hydration owns the unknown capability probe.
    void refreshSupportedSessionControlAfterTurn(sessionId)

    return true
  }

  return false
}
