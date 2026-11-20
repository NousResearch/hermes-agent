/**
 * The deferred liveness probe one pooled (secondary) socket gets when a forced
 * wake would otherwise blind-close it (#94769 review).
 *
 * Its own module because it is a self-contained policy over a single entry —
 * one timer, one streak, one decision — and because store/gateway.ts is over
 * its own line cap: the pooled route's four moving parts (its retry ladder, its
 * liveness probe, its min-lifetime grace, the live-work pruner) each belong to a
 * topical sibling rather than to the registry façade.
 *
 * The registry supplies the two facts this policy cannot know: whether the entry
 * it is probing is still the one the registry holds with an open socket, and
 * which scopes have live work. Everything else — the streak tolerance, the
 * compatibility carve-out, the bounded re-probe — lives here.
 */
import { decideLivenessForceClose, LIVENESS_REPROBE_DELAY_MS, probeGatewayLiveness } from '@/lib/gateway-liveness-policy'
import { isMissingRpcMethod } from '@/lib/gateway-rpc'
import type { Secondary } from '@/store/gateway-socket-entry'

export interface SecondaryLivenessHost {
  /** Is this exact entry still the registry's, with a socket that reports open? */
  liveSocket: (entry: Secondary) => boolean
  /** Scopes (or bare profiles) with work in flight, from the registry's keep-set hook. */
  liveScopes: () => undefined | ReadonlySet<string>
}

/** A deferred liveness re-probe for one entry: cleared when the probe is
 *  answered, the socket is redialed (fresh streak), or the entry is disposed. */
export function clearSecondaryLivenessReprobe(entry: Secondary): void {
  if (entry.livenessReprobeTimer !== null) {
    clearTimeout(entry.livenessReprobeTimer)
    entry.livenessReprobeTimer = null
  }
}

/** Probe a live-in-use secondary instead of blind-closing it on a forced wake,
 *  and close it only when the probe proves it not alive. A half-open TCP
 *  connection (sleep/wake, silent network drop) reports connectionState
 *  'open' forever and fires no close event, so without this close an in-flight
 *  request rides a dead transport until its per-call timeout — prompt.submit's
 *  is 30 minutes. Closing arms the entry's ordinary reconnect backoff via its
 *  onState('closed') handler; a healthy-but-busy backend answers the ping and
 *  keeps its socket (#94769 review). */
export function probeSecondaryLiveness(entry: Secondary, host: SecondaryLivenessHost): void {
  void probeGatewayLiveness(entry.gateway).then(
    () => {
      entry.livenessProbeFailures = 0
      clearSecondaryLivenessReprobe(entry)
    },
    (error: unknown) => {
      // -32601 (method not found) = a version-skewed but HEALTHY backend that
      // predates the ping method — the same compatibility carve-out the
      // primary's probe makes in use-gateway-boot.
      if (isMissingRpcMethod(error)) {
        entry.livenessProbeFailures = 0
        clearSecondaryLivenessReprobe(entry)

        return
      }

      // The entry may have been pruned or redialed while the probe was
      // pending; only the very same socket may be torn down.
      if (!host.liveSocket(entry)) {
        return
      }

      // ONE missed ping is not proof of death: a live backend mid tool call
      // can starve its event loop past the probe budget, and force-closing it
      // feeds the backend's ws_orphan_reap, interrupting the valid turn
      // (#94769 review). Apply the SAME streak policy as the primary's probe
      // (decideLivenessForceClose): defer the first failure while work is
      // in flight behind a bounded re-probe, close only when the streak is
      // exhausted — or immediately when nothing is in flight to protect.
      entry.livenessProbeFailures += 1

      // Counted RPCs alone under-report in-flight work: prompt.submit returns
      // before the turn ends, so a foreground turn mid tool call shows
      // activeRequests 0. The registry's live-scope hook supplies the turn.
      const liveScopes = host.liveScopes()

      const turnInFlight =
        liveScopes && (liveScopes.has(entry.scope) || (!entry.connectionId && liveScopes.has(entry.profile))) ? 1 : 0

      const decision = decideLivenessForceClose({
        workingSessionCount: entry.activeRequests + turnInFlight,
        consecutiveFailures: entry.livenessProbeFailures
      })

      if (!decision.close) {
        if (entry.livenessReprobeTimer === null) {
          entry.livenessReprobeTimer = setTimeout(() => {
            entry.livenessReprobeTimer = null

            // The entry may have been pruned or redialed while the re-probe
            // waited; only the same open socket may be probed again.
            if (!host.liveSocket(entry)) {
              return
            }

            probeSecondaryLiveness(entry, host)
          }, LIVENESS_REPROBE_DELAY_MS)
        }

        return
      }

      entry.livenessProbeFailures = 0
      clearSecondaryLivenessReprobe(entry)
      entry.gateway.close()
    }
  )
}
