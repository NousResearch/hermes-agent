import { describe, expect, it } from 'vitest'

import { relativeTime, startOfLocalDay } from './time'

// Pin TZ for deterministic day-boundary math; relativeTime reads the local
// clock, so a TZ override makes the test resilient to host TZ. Asia/Shanghai
// (UTC+8, no DST) matches the bug reporter's setup in #76725.
process.env.TZ = 'Asia/Shanghai'

// Build a UTC timestamp that, in Asia/Shanghai (UTC+8), reads as the given
// wall-clock value. Working in UTC and shifting by -8h keeps the math
// timezone-independent and lets the test run anywhere.
function local(
  year: number,
  month: number,
  day: number,
  hour = 0,
  minute = 0,
): number {
  return Date.UTC(year, month - 1, day, hour - 8, minute)
}

describe('relativeTime day-bucket calendar boundary (#76725)', () => {
  it('the reported bug: 32 h to next-next-day must read "in 2 days", not "tomorrow"', () => {
    // 8/2 18:00 → 8/4 02:00 — the legacy code rounded 32 / 24 h to 1 day and
    // said "in 1 day" / "tomorrow". The cron sidebar's "Next run" meta
    // label inherits this bug; 02:00 the day-after-next is two calendar
    // days away, not one.
    const now = local(2026, 8, 2, 18, 0)
    const target = local(2026, 8, 4, 2, 0)
    const formatted = relativeTime(target, now)

    expect(formatted).toMatch(/2 days?/)
    expect(formatted).not.toMatch(/tomorrow/)
  })

  it('the same fix must not regress "next day" rendering (still "tomorrow")', () => {
    // 8/2 02:00 → 8/3 02:00 — one calendar day. The legacy code already
    // returned "tomorrow" here; the new calendar-day math must agree.
    const now = local(2026, 8, 2, 2, 0)
    const target = local(2026, 8, 3, 2, 0)

    expect(relativeTime(target, now)).toMatch(/tomorrow|in 1 day/)
  })

  it('a 44 h target across two calendar days still reads "in 2 days"', () => {
    // 8/2 06:00 → 8/4 02:00 — 44 h, always correct under the legacy code;
    // ensures the new code agrees (regression guard).
    const now = local(2026, 8, 2, 6, 0)
    const target = local(2026, 8, 4, 2, 0)

    expect(relativeTime(target, now)).toMatch(/2 days?/)
  })

  it('a 23 h same-calendar-day target stays in the hour bucket', () => {
    // 8/2 00:30 → 8/2 23:30 — same calendar day, 23 h apart; the hour bucket
    // should still apply because absolute ms < 24 h. The day-bucket path is
    // bypassed and sign is correctly applied.
    const now = local(2026, 8, 2, 0, 30)
    const target = local(2026, 8, 2, 23, 30)

    expect(relativeTime(target, now)).toMatch(/hr/)
  })

  it('past timestamps two calendar days back read "2 days ago" (sign survives)', () => {
    // 8/2 10:00 → 7/31 02:00 — sign must survive the day-bucket path so
    // history panels render "2 days ago", not "in 2 days". The first
    // attempt at the fix multiplied dayDiff by sign and accidentally
    // flipped the direction; this test catches that regression.
    const now = local(2026, 8, 2, 10, 0)
    const target = local(2026, 7, 31, 2, 0)
    const formatted = relativeTime(target, now)

    expect(formatted).toMatch(/ago/)
    expect(formatted).not.toMatch(/in /)
  })

  it('a 28 h target across three calendar days reads "in 3 days"', () => {
    // 8/2 22:00 → 8/5 02:00 — 28 h apart but three calendar days away. The
    // legacy 86_400_000 ms math said "in 2 days"; the new math says
    // "in 3 days", which is what a user reads on the clock.
    const now = local(2026, 8, 2, 22, 0)
    const target = local(2026, 8, 5, 2, 0)

    expect(relativeTime(target, now)).toMatch(/3 days?/)
  })
})

describe('startOfLocalDay (helper for the day bucket)', () => {
  it('snaps to local midnight', () => {
    const at = local(2026, 8, 2, 14, 30)
    const dayStart = startOfLocalDay(at)
    const nextDayStart = startOfLocalDay(at + 24 * 3_600_000)

    expect(new Date(dayStart).getDate()).toBe(new Date(at).getDate())
    expect(nextDayStart - dayStart).toBe(24 * 3_600_000)
  })
})