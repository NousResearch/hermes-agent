import { $activeGatewayProfile } from '@/store/profile'
import { $gatewayState } from '@/store/session'

import {
  backendTransitionFor,
  beginBackendTransition,
  endBackendTransition,
  transitionProfileKey
} from './backend-transition'

/** The longest a switch's state may cover the window's reconnect to the new route. */
export const SWITCH_SETTLE_MAX_MS = 30_000
// Without a visible drop, the old socket may still read "open" for a moment.
const SWITCH_SETTLE_MIN_MS = 2_000

/**
 * The longest a plugin's `prepare` or `beforeApply` may hold the composer and
 * the queue. The SDK guide states it.
 */
export const SWITCH_HOOK_MAX_MS = 10 * 60_000

/**
 * Begin the plugin switch's state. It lasts for as long as the plugin's hook
 * runs and the window then needs to reach the new route: its deadline covers
 * both, so it only ends a hook that never returns.
 */
export function beginSwitchTransition(profile: string, kind: 'starting' | 'stopping'): void {
  beginBackendTransition(profile, kind, { maxMs: SWITCH_HOOK_MAX_MS + SWITCH_SETTLE_MAX_MS, source: 'switch' })
}

/**
 * End the plugin switch's "Starting sandbox…" / "Stopping sandbox…" state.
 * When the switch applied and the window shows that profile, keep it until the
 * window's socket for the new route is open, so the reconnect in between reads
 * as the same state rather than "Backend connecting" / "offline". A switch
 * that did not apply, or a profile in the background, ends at once.
 */
export function endSwitchTransition(profile: string, applied: boolean): void {
  const key = transitionProfileKey(profile)

  if (!applied || transitionProfileKey($activeGatewayProfile.get()) !== key) {
    endBackendTransition(key, { ok: applied, source: 'switch' })

    return
  }

  const startedAt = Date.now()
  // The state this switch leaves to settle. A newer switch of the profile replaces it, and ends its own.
  const settling = backendTransitionFor(key)
  let sawDrop = $gatewayState.get() !== 'open'
  let off: () => void = () => undefined

  const finish = (ok: boolean) => {
    off()
    clearTimeout(minTimer)
    clearTimeout(maxTimer)

    if (backendTransitionFor(key) === settling) {
      endBackendTransition(key, { ok, source: 'switch' })
    }
  }

  const check = () => {
    const state = $gatewayState.get()

    if (state !== 'open') {
      sawDrop = true

      return
    }

    if (sawDrop || Date.now() - startedAt >= SWITCH_SETTLE_MIN_MS) {
      finish(true)
    }
  }

  const minTimer = setTimeout(check, SWITCH_SETTLE_MIN_MS)
  const maxTimer = setTimeout(() => finish(false), SWITCH_SETTLE_MAX_MS)

  off = $gatewayState.listen(check)
}
