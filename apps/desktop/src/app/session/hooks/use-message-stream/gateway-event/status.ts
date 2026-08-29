import { translateNow } from '@/i18n'
import { textPart } from '@/lib/chat-messages'
import { coerceGatewayText } from '@/lib/chat-runtime'
import { isProviderSetupErrorMessage } from '@/lib/provider-setup-errors'
import { type AgentNoticePayload, clearAgentNotice, nativeNoticeInput, showAgentNotice } from '@/store/agent-notices'
import { clearClarifyRequest } from '@/store/clarify'
import { reconcileSessionCompacting, setSessionCompacting } from '@/store/compaction'
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

/** status.update / review.summary / notification.show / notification.clear /
 *  error — the status-and-notice tail of the dispatcher. */
export function handleStatusEvent(ctx: GatewayEventContext): boolean {
  const { deps, event, payload, sessionId, isActiveEvent, occurredAt } = ctx
  const { compactedTurnRef, failAssistantMessage, flushQueuedDeltas, queryClient, updateSessionState } = deps

  if (event.type === 'status.update') {
    if (sessionId && payload?.kind === 'compacting') {
      setSessionCompacting(sessionId, true)
      compactedTurnRef.current.add(sessionId)
    } else if (sessionId && payload?.kind === 'compacted') {
      reconcileSessionCompacting(sessionId, 'terminal')
      compactedTurnRef.current.delete(sessionId)
    } else if (sessionId && payload?.kind === 'process') {
      // The gateway's notification poller announces background process
      // completions / watch matches here — re-sync the status stack.
      void refreshBackgroundProcesses(sessionId)
    } else if (sessionId && payload?.kind === 'goal') {
      applyGoalStatusText(sessionId, coerceGatewayText(payload?.text))
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
      failAssistantMessage(sessionId, errorMessage, occurredAt)
    }

    if (isActiveEvent) {
      setTurnStartedAt(null)
    }

    return true
  }

  return false
}
