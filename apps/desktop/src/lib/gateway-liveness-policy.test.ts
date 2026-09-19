import { describe, expect, it } from 'vitest'

import {
  decideLivenessForceClose,
  LIVENESS_PROBE_FAILURE_STREAK,
  LIVENESS_REPROBE_DELAY_MS,
  LIVENESS_SERVED_GRACE_MS
} from './gateway-liveness-policy'

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

  it('a pending RPC makes silence as inconclusive as a running turn (cold session.resume)', () => {
    // No turn is in flight, but a session.resume is grinding through a cold
    // agent build — the probe starving is expected, not proof of death.
    const decision = decideLivenessForceClose({
      workingSessionCount: 0,
      consecutiveFailures: 1,
      pendingRpcCount: 1
    })

    expect(decision).toEqual({ close: false, reason: 'in-flight-work-deferred' })
  })

  it('closes despite pending RPCs once the streak exhausts with nothing served', () => {
    // A genuinely dead socket also has pending RPCs (they will never
    // complete) — it must be rebuilt within the streak bound, never left to
    // the RPC's own multi-minute timeout.
    const decision = decideLivenessForceClose({
      workingSessionCount: 0,
      consecutiveFailures: LIVENESS_PROBE_FAILURE_STREAK,
      pendingRpcCount: 2
    })

    expect(decision).toEqual({ close: true, reason: 'failure-streak-exhausted' })
  })

  it('recently delivered frames are proof of life: defer past any streak while work is in flight', () => {
    // The backend is streaming other sessions' deltas / answering heartbeats
    // between event-loop stalls — the transport provably serves, so a
    // starving probe must never tear it down mid-work.
    const decision = decideLivenessForceClose({
      workingSessionCount: 0,
      consecutiveFailures: LIVENESS_PROBE_FAILURE_STREAK + 10,
      pendingRpcCount: 1,
      msSinceLastServed: LIVENESS_SERVED_GRACE_MS - 1
    })

    expect(decision).toEqual({ close: false, reason: 'recent-service-deferred' })
  })

  it('service proof outside the grace window does not defer past the streak', () => {
    const decision = decideLivenessForceClose({
      workingSessionCount: 1,
      consecutiveFailures: LIVENESS_PROBE_FAILURE_STREAK,
      msSinceLastServed: LIVENESS_SERVED_GRACE_MS
    })

    expect(decision).toEqual({ close: true, reason: 'failure-streak-exhausted' })
  })

  it('recent service does NOT keep an idle dead socket alive (no work in flight)', () => {
    // With nothing riding the socket there is no reason to wait: legacy
    // shape unchanged even when a frame arrived moments ago.
    const decision = decideLivenessForceClose({
      workingSessionCount: 0,
      consecutiveFailures: 1,
      pendingRpcCount: 0,
      msSinceLastServed: 10
    })

    expect(decision).toEqual({ close: true, reason: 'no-in-flight-work' })
  })

  it('coerces malformed pendingRpcCount/msSinceLastServed defensively', () => {
    expect(
      decideLivenessForceClose({
        workingSessionCount: 0,
        consecutiveFailures: 1,
        pendingRpcCount: Number.NaN,
        msSinceLastServed: Number.NaN
      })
    ).toEqual({ close: true, reason: 'no-in-flight-work' })

    expect(
      decideLivenessForceClose({
        workingSessionCount: 1,
        consecutiveFailures: 1,
        msSinceLastServed: Number.NaN
      })
    ).toEqual({ close: false, reason: 'in-flight-work-deferred' })
  })

  it('keeps the re-probe delay far below the probe budget stack so detection stays bounded', () => {
    // The re-probe is a second 5s liveness ping after this delay; together
    // they must stay well under the reconnect-escalation horizon (5 min).
    expect(LIVENESS_REPROBE_DELAY_MS).toBeGreaterThan(0)
    expect(LIVENESS_REPROBE_DELAY_MS).toBeLessThan(60_000)
  })
})
