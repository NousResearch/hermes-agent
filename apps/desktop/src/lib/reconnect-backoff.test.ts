import { describe, expect, it, vi } from 'vitest'

import { reconnectBackoffDelayMs } from './reconnect-backoff'

describe('reconnectBackoffDelayMs', () => {
  it('increases the delay ceiling across consecutive failed attempts', () => {
    // Pin Math.random so we can read the ceiling directly through the
    // returned value instead of statistically sampling it.
    const randomSpy = vi.spyOn(Math, 'random').mockReturnValue(1)

    try {
      const delays = [0, 1, 2, 3, 4].map(attempt => reconnectBackoffDelayMs(attempt, { baseDelayMs: 300 }))

      expect(delays).toEqual([300, 600, 1200, 2400, 4800])

      for (let i = 1; i < delays.length; i++) {
        expect(delays[i]).toBeGreaterThan(delays[i - 1])
      }
    } finally {
      randomSpy.mockRestore()
    }
  })

  it('caps the delay ceiling instead of growing unbounded', () => {
    const randomSpy = vi.spyOn(Math, 'random').mockReturnValue(1)

    try {
      // Attempt 10 would be 300 * 2**10 = 307_200ms uncapped — must clamp.
      expect(reconnectBackoffDelayMs(10, { baseDelayMs: 300, capMs: 15_000 })).toBe(15_000)
      expect(reconnectBackoffDelayMs(50, { baseDelayMs: 300, capMs: 15_000 })).toBe(15_000)
    } finally {
      randomSpy.mockRestore()
    }
  })

  it('applies full jitter: delay is uniformly within [0, ceiling)', () => {
    const randomSpy = vi.spyOn(Math, 'random')

    try {
      randomSpy.mockReturnValue(0)
      expect(reconnectBackoffDelayMs(3, { baseDelayMs: 300 })).toBe(0)

      randomSpy.mockReturnValue(0.5)
      expect(reconnectBackoffDelayMs(3, { baseDelayMs: 300 })).toBe(1200)

      randomSpy.mockReturnValue(0.999)
      expect(reconnectBackoffDelayMs(3, { baseDelayMs: 300 })).toBeCloseTo(2400 * 0.999, 5)
    } finally {
      randomSpy.mockRestore()
    }
  })

  it('resets to the attempt-0 ceiling after a successful connection (caller passes attempt back to 0)', () => {
    const randomSpy = vi.spyOn(Math, 'random').mockReturnValue(1)

    try {
      // Simulates: fail, fail, fail (attempt climbs), succeed (caller resets
      // its counter to 0), fail again — the very next delay must be back at
      // the base ceiling, not continuing the climb.
      reconnectBackoffDelayMs(0, { baseDelayMs: 300 })
      reconnectBackoffDelayMs(1, { baseDelayMs: 300 })
      const afterSeveralFailures = reconnectBackoffDelayMs(2, { baseDelayMs: 300 })
      const afterReset = reconnectBackoffDelayMs(0, { baseDelayMs: 300 })

      expect(afterSeveralFailures).toBe(1200)
      expect(afterReset).toBe(300)
    } finally {
      randomSpy.mockRestore()
    }
  })

  it('treats negative attempt numbers as attempt 0 rather than throwing or returning a negative delay', () => {
    const randomSpy = vi.spyOn(Math, 'random').mockReturnValue(1)

    try {
      expect(reconnectBackoffDelayMs(-5, { baseDelayMs: 300 })).toBe(300)
    } finally {
      randomSpy.mockRestore()
    }
  })

  it('uses sane defaults when no options are passed', () => {
    const randomSpy = vi.spyOn(Math, 'random').mockReturnValue(1)

    try {
      expect(reconnectBackoffDelayMs(0)).toBe(300)
      expect(reconnectBackoffDelayMs(100)).toBe(15_000)
    } finally {
      randomSpy.mockRestore()
    }
  })
})
