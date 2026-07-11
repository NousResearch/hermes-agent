import { setModelAssignment } from '@/hermes'
import { translateNow } from '@/i18n'
import type { GatewayEventPayload } from '@/lib/chat-messages'
import { $gateway } from '@/store/gateway'
import { notify } from '@/store/notifications'
import { requestManagedReSignIn } from '@/store/onboarding'

// hc-511 — managed relay-key recovery.
//
// A chat turn whose relay request is rejected with an auth error (HTTP 401/403)
// arrives as a terminal `error` event carrying { code:'auth', status_code }
// (see tui_gateway/server.py). Rather than dead-ending on a generic error, the
// desktop tries to self-heal the managed relay key and retry once — and, when
// it can't (no reusable login token: a `*.local`/env seed key or an expired
// JWT), routes to a visible re-sign-in instead of letting every send silently
// 401. relay/scheduler are unchanged; this only aligns to their existing
// provision-key semantics via the electron self-heal bridge.

// Classify an error-event payload as a relay/provider auth failure. The runtime
// tags these with code 'auth'; the status code is a defensive fallback for any
// path that carries the number but not the code.
export function isManagedRelayAuthError(payload: GatewayEventPayload | undefined): boolean {
  if (!payload) {
    return false
  }

  const code = typeof payload.code === 'string' ? payload.code : ''
  const status = typeof payload.status_code === 'number' ? payload.status_code : 0

  return code === 'auth' || status === 401 || status === 403
}

// Regenerate the active turn in place (no duplicate user bubble). Registered by
// the controller — which owns the prompt actions — so this detached recovery
// path can retry once after a heal without threading a callback through every
// hook layer. Module-level mirrors the existing store patterns in this app
// (e.g. onboarding's pending-provider handoff).
let activeTurnResend: (() => Promise<void> | void) | null = null

export function registerActiveTurnResend(fn: (() => Promise<void> | void) | null): void {
  activeTurnResend = fn
}

// Sessions with a self-heal in flight, so a duplicate error event (or a retry
// that also fails) can't spin up a second concurrent recovery for the same turn.
// The 401 storm itself is bounded server-side by the electron re-provision
// cooldown (a second failure within the window resolves to needsSignIn, which
// breaks the retry loop after exactly one attempt).
const recovering = new Set<string>()

export interface ManagedRelayRecoveryArgs {
  /** Runtime session id the failed turn belongs to (may be unresolved). */
  sessionId: string | null | undefined
  /** True when the failed turn is the one the user is looking at (retry target). */
  isActive: boolean
}

// Attempt to recover from a relay auth failure. Returns true when this owns the
// outcome (healed + retried, or routed to re-sign-in) so the caller skips its
// generic error toast; false when it was NOT a managed-relay auth problem (e.g.
// a BYOK provider's own 401) and the generic path should surface it as usual.
export async function recoverFromManagedRelayAuthError(args: ManagedRelayRecoveryArgs): Promise<boolean> {
  const bridge = typeof window !== 'undefined' ? window.hermesDesktop?.managed : undefined

  // No desktop bridge (web dashboard / dev preview) or an older main process
  // without self-heal → nothing to do; let the generic error UI handle it.
  if (!bridge?.selfHeal) {
    return false
  }

  const guardKey = args.sessionId || '*'

  if (recovering.has(guardKey)) {
    return true
  }

  recovering.add(guardKey)

  try {
    const outcome = await bridge.selfHeal()

    // Relay accepted the key: this failure wasn't a managed-relay auth problem
    // (a BYOK provider 401, say) — defer to the generic error handling.
    if (!outcome || !outcome.relayUnauthorized) {
      return false
    }

    if (outcome.healed && outcome.assignment) {
      // Apply the freshly minted key the same way sign-in does, then reload the
      // runtime env so the in-flight process picks it up before the retry.
      await setModelAssignment(outcome.assignment)
      await $gateway
        .get()
        ?.request('reload.env')
        .catch(() => undefined)

      if (args.isActive && activeTurnResend) {
        notify({
          id: 'managed-relay-healed',
          kind: 'info',
          title: translateNow('managedRecovery.healed.title'),
          message: translateNow('managedRecovery.healed.retrying')
        })
        await activeTurnResend()
      } else {
        // Background turn (or no resend wired): the key is fixed, but we don't
        // silently resend a non-focused turn — tell the user it's ready to retry.
        notify({
          id: 'managed-relay-healed',
          kind: 'info',
          title: translateNow('managedRecovery.healed.title'),
          message: translateNow('managedRecovery.healed.resend')
        })
      }

      return true
    }

    // Could not heal — no reusable login token, or the stored JWT is itself
    // expired. Make it a visible, actionable state: a persistent notice plus the
    // managed sign-in flow, instead of another silent 401.
    notify({
      id: 'managed-relay-signin',
      kind: 'error',
      title: translateNow('managedRecovery.signInRequired.title'),
      message: translateNow('managedRecovery.signInRequired.message')
    })
    requestManagedReSignIn(translateNow('managedRecovery.signInRequired.reason'))

    return true
  } catch {
    // A recovery that itself throws must not swallow the error — fall back to the
    // generic error UI so the failure is never invisible.
    return false
  } finally {
    recovering.delete(guardKey)
  }
}
