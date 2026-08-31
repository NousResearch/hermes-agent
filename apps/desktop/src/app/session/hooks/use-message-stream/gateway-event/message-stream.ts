import type { BillingBlock } from '@hermes/shared'

import { burstVibeHearts } from '@/components/chat/vibe-hearts'
import { translateNow } from '@/i18n'
import { coerceGatewayText, coerceThinkingText } from '@/lib/chat-runtime'
import { playCompletionSound } from '@/lib/completion-sound'
import { parseErrorSurface } from '@/lib/error-surface'
import { triggerHaptic } from '@/lib/haptics'
import { billingCtaLabel, clearBillingBlock, runBillingRecovery, setBillingBlock } from '@/store/billing-block'
import { clearClarifyRequest } from '@/store/clarify'
import { setSessionCompacting } from '@/store/compaction'
import { notify } from '@/store/notifications'
import { flashPetActivity, markPetUnread, setPetActivity } from '@/store/pet'
import { clearAllPrompts } from '@/store/prompts'
import { providerWaitText, setSessionProviderWait } from '@/store/provider-wait'
import { setCurrentUsage, setTurnStartedAt } from '@/store/session'
import { pruneFinishedSessionSubagents } from '@/store/subagents'
import { clearActiveSessionTodos } from '@/store/todos'

import type { GatewayEventContext } from './types'

function firstBillingLine(text: string): string {
  return (text || '').split('\n')[0]?.trim() ?? ''
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

/** The message/reasoning/MoA streaming family: message.start → deltas →
 *  interim → complete, thinking/reasoning deltas, moa.* progress, reaction. */
export function handleMessageStreamEvent(ctx: GatewayEventContext): boolean {
  const { deps, event, payload, sessionId, isActiveEvent, occurredAt } = ctx

  const {
    appendAssistantDelta,
    appendReasoningDelta,
    compactedTurnRef,
    completeAssistantMessage,
    finalizeInterimAssistantMessage,
    flushQueuedDeltas,
    nativeSubagentSessionsRef,
    sessionStateByRuntimeIdRef,
    updateSessionState
  } = deps

  if (event.type === 'message.start') {
    if (!sessionId) {
      return true
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
        // A message.start begins a distinct backend turn. Never let a
        // dropped completion from the prior turn make its streaming bubble
        // the target for this turn's first delta.
        streamId: null,
        // Backend accepted the turn — the no-payload settle gate below may
        // now treat a running=false heartbeat as a real turn end.
        turnLive: true,
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

  if (event.type === 'message.delta') {
    if (sessionId) {
      appendAssistantDelta(sessionId, coerceGatewayText(payload?.text), occurredAt)
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
        finalizeInterimAssistantMessage(sessionId, text, occurredAt)
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
      const label = coerceGatewayText(payload?.label) || 'reference'
      const idx = typeof payload?.index === 'number' ? payload.index : undefined
      const cnt = typeof payload?.count === 'number' ? payload.count : undefined
      const header = idx && cnt ? `◇ Reference ${idx}/${cnt} — ${label}` : `◇ Reference — ${label}`
      const body = coerceThinkingText(payload?.text)
      const text = `${header}\n${body}\n\n`

      if (idx === undefined || idx <= 1) {
        // First reference: clear any stale reasoning left over from
        // before this turn's references start, same as before.
        appendReasoningDelta(sessionId, text, true, occurredAt)
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
        appendReasoningDelta(sessionId, text, false, occurredAt)
        flushQueuedDeltas(sessionId)
      }
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
    // structured fields: `error` is the message, `partial` marks
    // `text` as streamed output to keep rather than the error string, and
    // `error_surface` (newer gateways) names the failing layer for the card.
    const failure =
      payload?.status === 'error'
        ? {
            error: coerceGatewayText(payload.error).trim() || finalText || 'Hermes reported an error',
            partial: Boolean(payload.partial),
            surface: parseErrorSurface(payload.error_surface)
          }
        : undefined

    completeAssistantMessage(sessionId, finalText, payload?.response_previewed, failure, occurredAt)

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

    return true
  }

  return false
}
