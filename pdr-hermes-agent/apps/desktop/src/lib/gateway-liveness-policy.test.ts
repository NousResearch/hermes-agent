import { describe, expect, it } from 'vitest'

import { decideLivenessForceClose, LIVENESS_PROBE_FAILURE_STREAK, probeGatewayLiveness } from './gateway-liveness-policy'

describe('decideLivenessForceClose', () => {
  it('keeps the socket on the first timeout while a turn is in flight (#95327)', () => {
    const decision = decideLivenessForceClose({ workingSessionCount: 1, consecutiveFailures: 1 })

    expect(decision.close).toBe(false)
    expect(decision.reason).toBe('in-flight-work-deferred')
  })

  it('defers for every busy session count, not just one', () => {
    for (const workingSessionCount of [1, 2, 7]) {
      const decision = decideLivenessForceClose({ workingSessionCount, consecutiveFailures: 1 })

      expect(decision).toEqual({ close: false, reason: 'in-flight-work-deferred' })
    }
  })

  it('closes once the unanswered streak reaches the limit even while busy', () => {
    const decision = decideLivenessForceClose({
      workingSessionCount: 3,
      consecutiveFailures: LIVENESS_PROBE_FAILURE_STREAK
    })

    expect(decision.close).toBe(true)
    expect(decision.reason).toBe('failure-streak-exhausted')
  })

  it('closes immediately with no work in flight (legacy shape unchanged)', () => {
    const decision = decideLivenessForceClose({ workingSessionCount: 0, consecutiveFailures: 1 })

    expect(decision.close).toBe(true)
    expect(decision.reason).toBe('no-in-flight-work')
  })

  it('never lets a deferred defer outlive the streak boundary', () => {
    // Walk the whole streak: every failure below the limit defers; the limit
    // itself and anything beyond close.
    for (let failures = 1; failures <= LIVENESS_PROBE_FAILURE_STREAK + 2; failures += 1) {
      const decision = decideLivenessForceClose({ workingSessionCount: 1, consecutiveFailures: failures })

      expect(decision.close).toBe(failures >= LIVENESS_PROBE_FAILURE_STREAK)
    }
  })

  it('coerces malformed counters defensively instead of throwing', () => {
    expect(decideLivenessForceClose({ workingSessionCount: Number.NaN, consecutiveFailures: 0 })).toEqual({
      close: true,
      reason: 'no-in-flight-work'
    })
    expect(decideLivenessForceClose({ workingSessionCount: -3, consecutiveFailures: -10 })).toEqual({
      close: true,
      reason: 'no-in-flight-work'
    })
  })
})

describe('probeGatewayLiveness', () => {
  // The backend answers `gateway.ping` on its socket reader; every other RPC, `ping` included,
  // waits behind the connection's serial control lane. A probe stuck behind a slow control RPC
  // closed a healthy socket (#145737). The reader fast path itself is pinned by
  // tests/tui_gateway/test_tui_gateway_ws.py::test_ws_ping_is_answered_while_an_earlier_rpc_blocks_dispatch.
  it('sends the reader-answered gateway.ping, never a method queued behind control RPCs', async () => {
    const gateway = {
      request: (method: string) =>
        method === 'gateway.ping' ? Promise.resolve({ ok: true }) : new Promise<unknown>(() => undefined)
    }

    await expect(probeGatewayLiveness(gateway)).resolves.toEqual({ ok: true })
  })
})
