import { describe, expect, it } from 'vitest'

import {
  decideLivenessForceClose,
  LIVENESS_PROBE_FAILURE_STREAK,
  LIVENESS_REPROBE_DELAY_MS
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

  it('keeps the re-probe delay far below the probe budget stack so detection stays bounded', () => {
    // The re-probe is a second 5s liveness ping after this delay; together
    // they must stay well under the reconnect-escalation horizon (5 min).
    expect(LIVENESS_REPROBE_DELAY_MS).toBeGreaterThan(0)
    expect(LIVENESS_REPROBE_DELAY_MS).toBeLessThan(60_000)
  })
})
