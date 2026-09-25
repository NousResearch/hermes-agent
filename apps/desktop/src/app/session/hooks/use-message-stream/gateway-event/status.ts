import { isSessionNotOwnedError } from '@/app/session/hooks/use-prompt-actions/utils'
import { translateNow, TRANSLATIONS } from '@/i18n'
import { getRuntimeI18nLocale } from '@/i18n/runtime'
import { textPart } from '@/lib/chat-messages'
import { coerceGatewayText } from '@/lib/chat-runtime'
import type { ErrorSurface } from '@/lib/error-surface'
import { errorCardText } from '@/lib/error-surface-copy'
import { isProviderSetupErrorMessage } from '@/lib/provider-setup-errors'
import { type AgentNoticePayload, clearAgentNotice, nativeNoticeInput, showAgentNotice } from '@/store/agent-notices'
import { clearClarifyRequest } from '@/store/clarify'
import { reconcileSessionCompacting, setSessionCompacting, takeCompressDeferred } from '@/store/compaction'
import { refreshBackgroundProcesses } from '@/store/composer-status'
import { applyGoalStatusText } from '@/store/goals'
import { dispatchNativeNotification } from '@/store/native-notifications'
import { isDiskFullErrorMessage, notify, notifyError } from '@/store/notifications'
import { requestDesktopOnboarding } from '@/store/onboarding'
import { flashPetActivity, setPetActivity } from '@/store/pet'
import { clearAllPrompts } from '@/store/prompts'
import { setTurnStartedAt } from '@/store/session'
import { clearActiveSessionTodos } from '@/store/todos'

import type { GatewayEventContext } from './types'

/** Lifecycle text that announces the session left its selected model for a fallback. */
export function isFallbackSwitchStatus(kind: string | undefined, text: string): boolean {
  if (kind === 'fallback') {
    return Boolean(text.trim())
  }

  return /model fallback|provider fallback|switched to fallback|switching to fallback/i.test(text)
}

/** status.update / review.summary / notification.show / notification.clear /
 *  error — the status-and-notice tail of the dispatcher. */
export function handleStatusEvent(ctx: GatewayEventContext): boolean {
  const { deps, event, payload, sessionId, isActiveEvent, occurredAt } = ctx

  const {
    compactedTurnRef,
    failAssistantMessage,
    flushQueuedDeltas,
    hydrateFromStoredSession,
    queryClient,
    sessionStateByRuntimeIdRef,
    updateSessionState
  } = deps

  if (event.type === 'status.update') {
    // `compacting`/`compacted` is auto-compaction's pair. Manual /compress
    // pins `compressing` and always clears it with `ready` (the `finally` in
    // methods_session._compress_live). Both spellings drive the same phase —
    // the TUI has matched the pair since createGatewayEventHandler.ts:904;
    // without `compressing` the desktop showed no progress for /compress.
    if (sessionId && (payload?.kind === 'compacting' || payload?.kind === 'compressing')) {
      setSessionCompacting(sessionId, true)
      compactedTurnRef.current.add(sessionId)
    } else if (sessionId && (payload?.kind === 'compacted' || payload?.kind === 'ready')) {
      reconcileSessionCompacting(sessionId, 'terminal')
      compactedTurnRef.current.delete(sessionId)

      // That same `pending` reply returned before the compress handler could
      // render the summary or flip its "still running in the background"
      // toast, so this edge is the only completion the client ever sees.
      // Without announcing it a deferred /compress finishes in silence: the
      // corner toast merely expires and the transcript changes underneath the
      // user with no acknowledgement it ever ran.
      const deferredCompress = takeCompressDeferred(sessionId)

      const completionText =
        coerceGatewayText(payload?.text).trim() || translateNow('notifications.compressDeferredDone')

      if (deferredCompress) {
        // Reuse the id the compress handler notified under, so this replaces
        // the pending toast in place instead of stacking a second one.
        notify({
          durationMs: 5_000,
          id: `session-compress:${sessionId}`,
          kind: 'success',
          message: completionText
        })
      }

      const announceDeferredCompress = () => {
        if (!deferredCompress) {
          return
        }

        flushQueuedDeltas(sessionId)
        updateSessionState(sessionId, state => ({
          ...state,
          messages: [
            ...state.messages,
            {
              id: `compress-complete-${occurredAt ?? Date.now()}`,
              role: 'system',
              parts: [textPart(completionText, occurredAt)],
              timestamp: occurredAt
            }
          ]
        }))
      }

      // A compress that finished with no live turn (manual /compress whose
      // RPC answered `pending` because the compute host outlived the wait,
      // #97948) has no turn-end hydrate to refresh the transcript — the
      // summarized bubbles would stay on screen forever. Mid-turn compaction
      // still defers to the turn's own settle path. The notice is appended
      // only after that hydrate resolves: it replaces the transcript wholesale
      // and would otherwise drop the line we just added.
      const state = sessionStateByRuntimeIdRef.current.get(sessionId)

      if (isActiveEvent && state && !state.busy && !state.awaitingResponse && !state.streamId) {
        void hydrateFromStoredSession(3, state.storedSessionId, sessionId).then(announceDeferredCompress)
      } else {
        announceDeferredCompress()
      }
    } else if (sessionId && payload?.kind === 'process') {
      // The gateway's notification poller announces background process
      // completions / watch matches here — re-sync the status stack.
      void refreshBackgroundProcesses(sessionId)
    } else if (sessionId && payload?.kind === 'goal') {
      applyGoalStatusText(sessionId, coerceGatewayText(payload?.text))
    } else if (sessionId && isFallbackSwitchStatus(payload?.kind, coerceGatewayText(payload?.text))) {
      // A provider/model switch is durable: the TUI paints it on the status rail, but Desktop
      // used to swallow every non-compaction status.update, so the reply came from a different
      // model with no indication.
      const text = coerceGatewayText(payload?.text).trim()

      flushQueuedDeltas(sessionId)
      updateSessionState(sessionId, state => ({
        ...state,
        messages: [
          ...state.messages,
          {
            id: `fallback-switch-${occurredAt}`,
            role: 'system',
            parts: [textPart(text, occurredAt)],
            timestamp: occurredAt
          }
        ]
      }))
    }

    return true
  }

  if (event.type === 'btw.complete') {
    // prompt.btw answers a side question and emits this on the originating
    // session. Persistent transcript line, matching the TUI's `[btw "q"]`
    // — without it Desktop only ever showed the acknowledgement (#99065).
    const text = coerceGatewayText(payload?.text).trim()

    if (text && sessionId) {
      const taskId = String(payload?.task_id ?? '').trim()
      const question = coerceGatewayText(payload?.question).trim()
      const header = `[btw${question ? ` "${question}"` : ''}${taskId ? ` (${taskId})` : ''}]`

      flushQueuedDeltas(sessionId)
      updateSessionState(sessionId, state => ({
        ...state,
        messages: [
          ...state.messages,
          {
            id: `btw-complete-${taskId || Date.now()}`,
            role: 'system',
            parts: [textPart(`${header}\n${text}`, occurredAt)],
            timestamp: occurredAt
          }
        ]
      }))
    }

    return true
  }

  if (event.type === 'review.summary') {
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
            parts: [textPart(`review:${text}`, occurredAt)],
            timestamp: occurredAt
          }
        ]
      }))
    }

    return true
  }

  if (event.type === 'notification.show') {
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

    return true
  }

  if (event.type === 'notification.clear') {
    // Key-matched dismissal (e.g. credits restored clears the depleted
    // notice). notify() keys the toast by the notice key, so this maps
    // straight to dismissNotification(key).
    clearAgentNotice((event.payload as AgentNoticePayload | undefined)?.key)

    return true
  }

  if (event.type === 'error') {
    const errorMessage = payload?.message || 'Hermes reported an error'
    const looksLikeProviderSetup = isProviderSetupErrorMessage(errorMessage)

    // The gateway's `error` event carries no error_surface (prompt_turn.py
    // emits it for pre-turn refusals). Recover the two codes it CAN mean from
    // the text so the card and toast get the same plain copy + button gating
    // as a classified turn: a live-owner refusal (SESSION_NOT_OWNED, #106217)
    // is deterministic — Retry hits the same wall, only a new chat helps —
    // and disk-full is a machine problem, not a provider one.
    const surface: ErrorSurface | null = isSessionNotOwnedError(new Error(errorMessage))
      ? { code: 'SESSION_NOT_OWNED', layer: 'gateway', retryable: false }
      : isDiskFullErrorMessage(errorMessage)
        ? { code: 'disk_full', layer: 'disk', retryable: false }
        : null

    // When a code was recovered, the glossed card sentence explains it better
    // than the raw refusal. When none was, the server's own text IS the plain
    // copy (tui_gateway/user_messages.py writes actionable sentences for
    // pre-turn failures — agent init, resume, cancelled-before-ready), and
    // burying it under a generic "couldn't finish" gloss would hide the one
    // instruction the user needs.
    const card = surface ? errorCardText(TRANSLATIONS[getRuntimeI18nLocale()].assistant.thread, surface) : null
    const toastMessage = card ? `${card.title}. ${card.body}` : errorMessage

    // A turn that errors out has also ended — drop any open blocking prompt
    // for this session so an approval/sudo/secret overlay can't linger past
    // the failed turn (same intent as the message.complete clear).
    if (sessionId) {
      clearAllPrompts(sessionId)
      clearClarifyRequest(undefined, sessionId)
      clearActiveSessionTodos(sessionId)
      reconcileSessionCompacting(sessionId, 'terminal')
      compactedTurnRef.current.delete(sessionId)
    }

    if (isActiveEvent) {
      setPetActivity({ reasoning: false, toolRunning: false })
      flashPetActivity({ error: true })
    }

    dispatchNativeNotification({
      body: toastMessage,
      kind: 'turnError',
      sessionId,
      title: translateNow('notifications.native.turnErrorTitle')
    })

    if (looksLikeProviderSetup) {
      requestDesktopOnboarding(errorMessage)
    } else if (surface?.code === 'disk_full') {
      notifyError(new Error(errorMessage), translateNow('notifications.errors.diskFull'))
    } else {
      // Toast globally, not just when the failing thread is focused: a
      // turn-ending error (e.g. out of funds) blocks every thread, so the
      // inline error alone is too easy to miss. The stable id collapses the
      // same error from multiple blocked threads into one toast. For a
      // recovered code the message is the card's glossed sentence with the raw
      // gateway text as the dimmed detail; otherwise the server copy is the
      // message and there is no separate detail to repeat.
      // No Retry action: assistant-ui's reload is per-thread, and for the
      // codes recovered above a retry would fail identically anyway.
      notify({
        detail: surface ? errorMessage : undefined,
        id: `gateway-error:${errorMessage}`,
        kind: 'error',
        message: toastMessage,
        title: translateNow('assistant.thread.errorToastTitle')
      })
    }

    if (sessionId) {
      flushQueuedDeltas(sessionId)
      failAssistantMessage(sessionId, errorMessage, occurredAt, surface)
    }

    if (isActiveEvent) {
      setTurnStartedAt(null)
    }

    return true
  }

  return false
}
