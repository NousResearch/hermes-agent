import { hasGatewaySessionTurn } from './gateway'
import { setAwaitingResponse, setBusy } from './session'
import { $sessionStates, publishSessionState, sessionScopeForRuntimeId, sessionTileDelegate } from './session-states'

/** Repair stale, unheld UI claims after reconnect. A socket opening does not
 * prove that the backend restarted or that its turns settled: routed turn
 * leases keep their busy/awaiting claims until authoritative events arrive.
 *
 * Secondary scopes match only their own event provenance; the primary's
 * undefined scope touches only scope-less runtimes. Needs-input is untouched.
 * Retire through the delegate first so cache, mirror and focused draft latches
 * agree (#93059); the mirror is a fallback when no wiring cache holds the row.
 */
export function reconcileBusyStatesOnReconnect(scope?: string) {
  const states = $sessionStates.get()

  for (const [runtimeId, state] of Object.entries(states)) {
    if (!state || (!state.busy && !state.awaitingResponse)) {
      continue
    }

    const recorded = sessionScopeForRuntimeId(runtimeId)

    if (scope === undefined ? recorded !== undefined : recorded !== scope) {
      continue
    }

    // A held turn can survive a transport blip. Its replay/settlement, not a
    // presentation repair, owns the busy -> idle edge and queue eligibility.
    if (scope !== undefined && hasGatewaySessionTurn(scope, runtimeId)) {
      continue
    }

    sessionTileDelegate()?.retireBusyClaim?.(runtimeId)

    // Re-read — the write path may have republished (and released) this entry.
    const published = $sessionStates.get()[runtimeId]

    if (published?.busy || published?.awaitingResponse) {
      publishSessionState(runtimeId, { ...published, awaitingResponse: false, busy: false })
    }
  }

  if (scope === undefined) {
    setBusy(false)
    setAwaitingResponse(false)
  }
}
