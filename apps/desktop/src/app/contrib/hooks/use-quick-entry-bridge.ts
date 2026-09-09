import { useEffect, useRef } from 'react'

import {
  initQuickEntryBridge,
  QUICK_TARGET_CURRENT,
  QUICK_TARGET_NEW,
  type QuickEntrySessionOption,
  type QuickEntrySubmitResult,
  setQuickEntrySubmitHandler
} from '@/store/quick-entry'
import { $gatewayState, $sessions } from '@/store/session'
import { sessionTileDelegate } from '@/store/session-states'
import { isAuxiliaryWindow } from '@/store/windows'

interface QuickEntryBridgeParams {
  submitText: (
    text: string,
    options?: { onAccepted?: (identity: QuickEntryAcceptedIdentity) => void }
  ) => Promise<boolean> | boolean
  submitTextToNewSession: (
    text: string,
    owner?: string
  ) => Promise<{ runtimeSessionId: string; sessionId: string }>
}

/** Exact session identity that accepted a prompt (runtime + durable stored id). */
interface QuickEntryAcceptedIdentity {
  runtimeSessionId: string
  storedSessionId: null | string
}

// The picker is a capture aid, not a session browser — a handful of recent
// rows is the whole point.
const QUICK_ENTRY_SESSION_OPTIONS = 5

/**
 * `submitText` resolves false when a pre-submit guard declines the prompt
 * without throwing (for example, while the target is busy). That is a failed
 * delivery, not an acknowledgement of success.
 */
export function quickEntrySubmitAck(
  submitted: boolean,
  identity?: null | QuickEntryAcceptedIdentity
): QuickEntrySubmitResult {
  if (!submitted) {
    return {
      code: 'submit-rejected',
      message: 'The prompt was not accepted.',
      ok: false,
      retryable: true
    }
  }

  // An accepted prompt names the session that accepted it. A submit that
  // resolved true without an identity callback (slash commands, which never
  // reach prompt.submit) still acknowledges, but claims no backend session.
  return identity
    ? { ok: true, runtimeSessionId: identity.runtimeSessionId, sessionId: identity.storedSessionId }
    : { ok: true }
}

function sessionOptions(): QuickEntrySessionOption[] {
  return $sessions
    .get()
    .filter(session => !session.archived)
    .slice(0, QUICK_ENTRY_SESSION_OPTIONS)
    .map(session => ({
      id: session.id,
      title: session.title?.trim() || session.preview?.trim() || session.id
    }))
}

/**
 * Wires the global-hotkey Quick Entry window back into the app, both ways:
 *
 * - **Inbound:** text captured there is routed by target and submitted through
 *   THIS window's normal prompt machinery — current chat rides `submitText`, a
 *   picked stored session rides the session-tile delegate (resume + submit,
 *   background, without touching the primary view — the same path tiled
 *   sessions use), and "new session" is a fresh draft + submit, exactly what
 *   clicking New Chat and typing does. One submit pipeline, no bespoke RPC.
 * - **Outbound:** gateway connection state + the recent-session list are pushed
 *   to the quick window (via main, which caches the latest push), so its input
 *   disables with a reconnect hint whenever the backend is unreachable.
 *
 * Handlers register ONCE through refs tracking the latest callbacks —
 * re-registering on identity churn leaves a nulled-handler window that can drop
 * a submit (the same bug shape use-pet-bridge guards). Primary window only: a
 * secondary session window must not also claim the global capture channel, or
 * one keystroke would send N prompts.
 */
export function useQuickEntryBridge({
  submitText,
  submitTextToNewSession
}: QuickEntryBridgeParams): void {
  const submitTextRef = useRef(submitText)
  submitTextRef.current = submitText
  const submitNewRef = useRef(submitTextToNewSession)
  submitNewRef.current = submitTextToNewSession

  useEffect(() => {
    if (isAuxiliaryWindow()) {
      return
    }

    setQuickEntrySubmitHandler(async ({ correlationId, target, text }) => {
      let acknowledged = false

      const ack = (result: QuickEntrySubmitResult) => {
        if (acknowledged) {
          return
        }

        acknowledged = true
        window.hermesDesktop?.quickEntry.ackSubmit(correlationId, result)
      }

      if (target === QUICK_TARGET_NEW) {
        // Create and submit as one route-neutral operation so drift cannot
        // orphan the new session (#85590).
        try {
          const created = await submitNewRef.current(text, correlationId)
          ack({ ok: true, runtimeSessionId: created.runtimeSessionId, sessionId: created.sessionId })
        } catch (error) {
          ack({ code: 'submit-failed', message: error instanceof Error ? error.message : String(error), ok: false, retryable: true })
        }

        return
      }

      if (target !== QUICK_TARGET_CURRENT) {
        // A picked stored session: resume + submit in the background through
        // the session-tile delegate so the primary view stays where it is.
        const delegate = sessionTileDelegate()

        if (delegate) {
          let promptDispatched = false

          try {
            const runtimeId = await delegate.resumeTile(target)
            promptDispatched = true
            const accepted = await delegate.submitToSession(runtimeId, text)

            if (accepted.storedSessionId !== target) {
              // The backend accepted the prompt, but the accepted runtime is not
              // bound to the requested stored session (or the binding is
              // unprovable). Never report success for a session this request
              // did not prove it used.
              ack({
                code: 'submit-identity-mismatch',
                message: 'The prompt was accepted by a different session than the one requested.',
                ok: false,
                retryable: false
              })

              return
            }

            ack({ ok: true, runtimeSessionId: accepted.runtimeSessionId, sessionId: accepted.storedSessionId })
          } catch (error) {
            if (promptDispatched) {
              ack({
                code: 'submit-failed',
                message: 'The selected session prompt was dispatched, but backend acceptance is unknown.',
                ok: false,
                retryable: false
              })

              return
            }

            ack({ code: 'submit-failed', message: error instanceof Error ? error.message : String(error), ok: false, retryable: true })
          }

          return
        }

        ack({
          code: 'submit-unavailable',
          message: 'The selected session submit path is unavailable.',
          ok: false,
          retryable: true
        })
        return
      }

      let acceptedIdentity: null | QuickEntryAcceptedIdentity = null

      try {
        const submitted = await submitTextRef.current(text, {
          onAccepted: identity => {
            acceptedIdentity = identity
          }
        })
        ack(quickEntrySubmitAck(submitted, acceptedIdentity))
      } catch (error) {
        ack({ code: 'submit-failed', message: error instanceof Error ? error.message : String(error), ok: false, retryable: true })
      }
    })

    const dispose = initQuickEntryBridge()

    return () => {
      setQuickEntrySubmitHandler(null)
      dispose()
    }
  }, [])

  // Push gateway truth into the quick window whenever it changes: connection
  // state gates its input; the recent-session list feeds its target picker.
  useEffect(() => {
    if (isAuxiliaryWindow()) {
      return
    }

    const api = window.hermesDesktop?.quickEntry

    if (!api?.pushState) {
      return
    }

    const push = () => {
      api.pushState({ connected: $gatewayState.get() === 'open', sessions: sessionOptions() })
    }

    push()

    const offGateway = $gatewayState.listen(push)
    const offSessions = $sessions.listen(push)

    return () => {
      offGateway()
      offSessions()
    }
  }, [])
}
