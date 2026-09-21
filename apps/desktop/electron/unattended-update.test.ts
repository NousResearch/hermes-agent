import { describe, expect, it } from 'vitest'

import {
  DEFAULT_UNATTENDED_SCHEDULE,
  armScheduledUpdate,
  decideScheduledAttempt,
  isWithinScheduledWindow,
  localTimeOf,
  msUntilNext,
  parseUnattendedSchedule
} from './unattended-update'

// Local-time helpers for a fixed synthetic clock (system timezone agnostic:
// we build Dates whose LOCAL getters are the values we assert on, so the test
// is portable across test runners with diverging TZ).
function at(hour: number, minute: number, second = 0, ms = 0): Date {
  const d = new Date(2026, 0, 5, hour, minute, second, ms)

  return d
}

describe('parseUnattendedSchedule (fail-closed)', () => {
  it('defaults to DISABLED', () => {
    expect(parseUnattendedSchedule(undefined)).toEqual(DEFAULT_UNATTENDED_SCHEDULE)
    expect(parseUnattendedSchedule(null)).toEqual(DEFAULT_UNATTENDED_SCHEDULE)
    expect(parseUnattendedSchedule('garbage')).toEqual(DEFAULT_UNATTENDED_SCHEDULE)
    expect(parseUnattendedSchedule(42)).toEqual(DEFAULT_UNATTENDED_SCHEDULE)
  })

  it('only true enables; truthy non-boolean never enables', () => {
    expect(parseUnattendedSchedule({ enabled: 1, hour: 3, minute: 0 })).toEqual(
      DEFAULT_UNATTENDED_SCHEDULE
    )
    expect(parseUnattendedSchedule({ enabled: 'true', hour: 3, minute: 0 })).toEqual(
      DEFAULT_UNATTENDED_SCHEDULE
    )
    expect(parseUnattendedSchedule({ enabled: true, hour: 3, minute: 0 })).toEqual({
      enabled: true,
      hour: 3,
      minute: 0
    })
  })

  it('clamps out-of-range times back to the default and fails closed on NaN', () => {
    expect(parseUnattendedSchedule({ enabled: true, hour: 30, minute: 0 })).toEqual({
      enabled: true,
      hour: DEFAULT_UNATTENDED_SCHEDULE.hour,
      minute: 0
    })
    expect(parseUnattendedSchedule({ enabled: true, hour: -1, minute: 0 })).toEqual({
      enabled: true,
      hour: DEFAULT_UNATTENDED_SCHEDULE.hour,
      minute: 0
    })
    expect(parseUnattendedSchedule({ enabled: true, hour: 3, minute: 99 })).toEqual({
      enabled: true,
      hour: 3,
      minute: DEFAULT_UNATTENDED_SCHEDULE.minute
    })
    expect(
      parseUnattendedSchedule({ enabled: true, hour: Number.NaN, minute: 0 })
    ).toEqual({
      enabled: true,
      hour: DEFAULT_UNATTENDED_SCHEDULE.hour,
      minute: 0
    })
  })

  it('accepts a valid explicit schedule', () => {
    expect(parseUnattendedSchedule({ enabled: true, hour: 23, minute: 59 })).toEqual({
      enabled: true,
      hour: 23,
      minute: 59
    })
  })
})

describe('localTimeOf / msUntilNext', () => {
  it('reads the local hour/minute', () => {
    expect(localTimeOf(at(4, 30))).toEqual({ hour: 4, minute: 30 })
  })

  it('returns a positive delay to the next occurrence, rolling to tomorrow when today passed', () => {
    // 03:00 scheduled, now 02:59 -> ~1 minute away (60_000ms).
    expect(msUntilNext(at(2, 59), { enabled: true, hour: 3, minute: 0 })).toBe(60_000)
    // Exact hit at 03:00:00 -> 0.
    expect(msUntilNext(at(3, 0), { enabled: true, hour: 3, minute: 0 })).toBe(0)
    // Just past the slot (03:00:30) -> rolls to tomorrow ~23h59m30s.
    expect(msUntilNext(at(3, 0, 30), { enabled: true, hour: 3, minute: 0 })).toBe(
      24 * 3_600_000 - 30_000
    )
    // Mid-morning -> later that same day.
    expect(msUntilNext(at(10, 15), { enabled: true, hour: 18, minute: 0 })).toBe(
      (18 * 60 - (10 * 60 + 15)) * 60_000
    )
  })
})

describe('isWithinScheduledWindow', () => {
  const s = { enabled: true, hour: 3, minute: 0 }

  it('accepts times inside the default 5-minute grace window', () => {
    expect(isWithinScheduledWindow(at(2, 57, 30), s)).toBe(true)
    expect(isWithinScheduledWindow(at(3, 0), s)).toBe(true)
    expect(isWithinScheduledWindow(at(3, 2, 29), s)).toBe(true)
  })

  it('rejects times outside the grace window and when disabled', () => {
    // Grace window (5 min, centered) = [02:55:00, 03:05:00]. Just outside = 1s beyond.
    expect(isWithinScheduledWindow(at(2, 54, 59), s)).toBe(false)
    expect(isWithinScheduledWindow(at(3, 5, 1), s)).toBe(false)
    expect(isWithinScheduledWindow(at(3, 0), { ...s, enabled: false })).toBe(false)
  })

  it('honors a caller-supplied graceMs', () => {
    expect(isWithinScheduledWindow(at(3, 4, 0), s, 5 * 60_000)).toBe(true)
    expect(isWithinScheduledWindow(at(3, 4, 0), s, 3 * 60_000)).toBe(false)
  })
})

describe('decideScheduledAttempt (refuses unless due)', () => {
  const schedule = { enabled: true, hour: 3, minute: 0 }

  it('runs only when enabled, on Windows, in-window and not recently run', () => {
    const decision = decideScheduledAttempt({
      schedule,
      now: at(3, 0),
      lastRunAtMs: null,
      cooldownMs: 6 * 3_600_000,
      isWindows: true
    })

    expect(decision).toEqual({ shouldRun: true, reason: 'enabled-and-due' })
  })

  it('refuses when disabled', () => {
    expect(
      decideScheduledAttempt({
        schedule: { ...schedule, enabled: false },
        now: at(3, 0),
        lastRunAtMs: null,
        cooldownMs: 6 * 3_600_000,
        isWindows: true
      })
    ).toEqual({ shouldRun: false, reason: 'disabled' })
  })

  it('refuses on non-Windows (feature is Windows-only by policy)', () => {
    expect(
      decideScheduledAttempt({
        schedule,
        now: at(3, 0),
        lastRunAtMs: null,
        cooldownMs: 6 * 3_600_000,
        isWindows: false
      })
    ).toEqual({ shouldRun: false, reason: 'not-windows' })
  })

  it('refuses outside the scheduled window', () => {
    expect(
      decideScheduledAttempt({
        schedule,
        now: at(9, 30),
        lastRunAtMs: null,
        cooldownMs: 6 * 3_600_000,
        isWindows: true
      })
    ).toEqual({ shouldRun: false, reason: 'not-due' })
  })

  it('refuses when it already ran recently', () => {
    const now = at(3, 0).getTime()
    const decision = decideScheduledAttempt({
      schedule,
      now: at(3, 0),
      lastRunAtMs: now - 30_000,
      cooldownMs: 6 * 3_600_000,
      isWindows: true
    })

    expect(decision).toEqual({ shouldRun: false, reason: 'recently-ran' })
  })
})

describe('armScheduledUpdate (timer driver)', () => {
  function fakeClock(start: Date) {
    let current = start.getTime()
    let nextId = 1
    const timers: { id: number; at: number; cb: () => void }[] = []

    return {
      advance(ms: number) {
        current += ms
        const pending = timers.filter(t => t.at > current)
        const due = timers.filter(t => t.at <= current).sort((a, b) => a.at - b.at)
        timers.length = 0
        timers.push(...pending)

        for (const t of due) {
          t.cb()
        }
      },
      setters: {
        now: () => new Date(current),
        setTimeoutFn(cb: () => void, ms: number) {
          timers.push({ id: nextId, at: current + ms, cb })

          return nextId++
        },
        clearTimeoutFn(id: unknown) {
          const index = timers.findIndex(t => t.id === id)

          if (index !== -1) {
            timers.splice(index, 1)
          }
        }
      }
    }
  }

  it('does nothing (handle pre-disposed) when the schedule is disabled', () => {
    const clock = fakeClock(at(2, 59))
    const { cancel, delayMs } = armScheduledUpdate({
      schedule: { ...DEFAULT_UNATTENDED_SCHEDULE, enabled: false },
      now: clock.setters.now,
      setTimeoutFn: clock.setters.setTimeoutFn,
      clearTimeoutFn: clock.setters.clearTimeoutFn,
      onDue: () => {
        throw new Error('must never fire while disabled')
      }
    })

    expect(delayMs).toBe(0)
    expect(() => clock.advance(3_600_000)).not.toThrow()
    cancel()
  })

  it('fires once on the scheduled slot, re-arms for the next day, and cancels cleanly', () => {
    let fires = 0
    const clock = fakeClock(at(2, 58))
    const handle = armScheduledUpdate({
      schedule: { enabled: true, hour: 3, minute: 0 },
      now: clock.setters.now,
      setTimeoutFn: clock.setters.setTimeoutFn,
      clearTimeoutFn: clock.setters.clearTimeoutFn,
      onDue: () => {
        fires++
      }
    })

    // 2 minutes to 03:00.
    expect(handle.delayMs).toBe(120_000)

    // Advance 90s -> still before the slot; no fire.
    clock.advance(90_000)
    expect(fires).toBe(0)

    // Advance another 45s -> past 03:00; exactly one fire.
    clock.advance(45_000)
    expect(fires).toBe(1)

    // The driver re-armed for tomorrow (~24h); a small advance must not refire.
    clock.advance(60_000)
    expect(fires).toBe(1)

    handle.cancel()
    // After cancel, advancing a full day must not fire again.
    clock.advance(86_400_000)
    expect(fires).toBe(1)
  })
})