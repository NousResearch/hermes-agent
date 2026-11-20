/**
 * The #95327 liveness probe for one window's primary gateway socket.
 *
 * Extracted from the useGatewayBoot effect (#C2a): the streak, the deferred
 * re-probe and the decision are one concern, and two callers share it — the
 * window that owns the socket, and the restart-followthrough recovery that must
 * not blind-close a socket the restart never touched.
 *
 * The policy itself lives in lib/gateway-liveness-policy.ts; this module owns
 * the per-window state around it: how many consecutive probes went unanswered,
 * the bounded re-probe a mid-turn deferral arms, and the close that ends it.
 *
 * NOT a wake-path concern any more: a wake DIALS (use-gateway-boot-wake). The
 * probe survives for exactly the case it was written for — a socket that must
 * not be torn out from under a running turn — plus the restart follow-through.
 */
import { JsonRpcGatewayError } from '@hermes/shared'

import type { HermesGateway } from '@/hermes'
import { decideLivenessForceClose, probeGatewayLiveness } from '@/lib/gateway-liveness-policy'
import { $workingSessionIds } from '@/store/session-states'

export type PrimaryProbeVerdict = 'alive' | 'dead' | 'deferred'

export interface PrimaryLivenessProbeHost {
  /** True while a probe may run at all: booted, no switch, not torn down, not parked. */
  ready: () => boolean
  /** The window's current primary client. */
  primary: () => HermesGateway
  /** Tag the close the probe is about to make as a timeout (C1 telemetry). */
  closeAsTimeout: () => void
  /**
   * One bounded retry before a mid-turn teardown: the first probe timeout while
   * work is in flight is inconclusive (a busy backend starves its own loop
   * without being dead), so the deferral re-probes after this delay instead of
   * force-closing a socket a running turn still rides on (#95327).
   */
  reprobeDelayMs: number
}

export interface PrimaryLivenessProbe {
  /** Cancel a deferred re-probe (a fresh socket opened, or the window unmounted). */
  cancelReprobe: () => void
  /** A probe went unanswered with work in flight: the caller's own deferral. */
  noteDeferred: () => void
  /** The socket answered (or a new one opened): the streak is over. */
  noteHealthy: () => void
  /** Arm the bounded re-probe a deferral asks for. */
  scheduleReprobe: () => void
  /** Consecutive unanswered probes, for a caller that decides for itself. */
  streak: () => number
  /** Probe the open socket once and apply the policy above. */
  probeOnce: () => Promise<PrimaryProbeVerdict>
}

export function createPrimaryLivenessProbe(host: PrimaryLivenessProbeHost): PrimaryLivenessProbe {
  let failures = 0
  let reprobeTimer: null | ReturnType<typeof setTimeout> = null

  function cancelReprobe() {
    if (reprobeTimer !== null) {
      clearTimeout(reprobeTimer)
      reprobeTimer = null
    }
  }

  function noteHealthy() {
    failures = 0
    cancelReprobe()
  }

  function scheduleReprobe() {
    if (reprobeTimer !== null || !host.ready()) {
      return
    }

    reprobeTimer = setTimeout(() => {
      reprobeTimer = null
      void probeOnce()
    }, host.reprobeDelayMs)
  }

  async function probeOnce(): Promise<PrimaryProbeVerdict> {
    const gateway = host.primary()

    if (!host.ready() || gateway.connectionState !== 'open') {
      return 'alive'
    }

    try {
      await probeGatewayLiveness(gateway)
      noteHealthy()

      return 'alive'
    } catch (probeErr) {
      // A version-skewed backend that predates the ping method answers -32601
      // (method not found) — a HEALTHY response, not a dead socket.
      // Force-closing on it would spin the reconnect loop forever. Every other
      // failure (timeout on a swallowed ping, transport error) means the socket
      // is not PROVABLY alive and must eventually be rebuilt.
      if (probeErr instanceof JsonRpcGatewayError && probeErr.code === -32601) {
        noteHealthy()

        return 'alive'
      }

      failures += 1

      const decision = decideLivenessForceClose({
        workingSessionCount: $workingSessionIds.get().length,
        consecutiveFailures: failures
      })

      if (!decision.close) {
        scheduleReprobe()

        return 'deferred'
      }

      failures = 0
      host.closeAsTimeout()

      return 'dead'
    }
  }

  return {
    cancelReprobe,
    noteDeferred: () => {
      failures += 1
    },
    noteHealthy,
    probeOnce,
    scheduleReprobe,
    streak: () => failures
  }
}
