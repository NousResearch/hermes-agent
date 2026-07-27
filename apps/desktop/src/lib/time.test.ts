import { describe, expect, it } from 'vitest'

import { calendarBucket, DAY, formatAgo, HOUR, MINUTE, nominalDayStart, SECOND, sessionBucketLabel } from './time'

const labels = {
  ageNow: 'now',
  ageSeconds: (s: number) => `${s}s ago`,
  ageMinutes: (m: number) => `${m}m ago`,
  ageHours: (h: number) => `${h}h ago`,
  ageDays: (d: number) => `${d}d ago`
}

const now = 1_000 * DAY
const ago = (delta: number) => formatAgo(now - delta, labels, now)

describe('formatAgo', () => {
  it('reads "now" under two seconds, then seconds', () => {
    expect(ago(0)).toBe('now')
    expect(ago(1.5 * SECOND)).toBe('now')
    expect(ago(5 * SECOND)).toBe('5s ago')
  })

  it('buckets to the coarsest unit, floored', () => {
    expect(ago(3 * MINUTE)).toBe('3m ago')
    expect(ago(2 * HOUR + 59 * MINUTE)).toBe('2h ago')
    expect(ago(5 * DAY)).toBe('5d ago')
  })

  it('clamps future timestamps to "now"', () => {
    expect(ago(-HOUR)).toBe('now')
  })
})

// Thursday 18 Jun 2026, local noon (15 Jun 2026 is a Monday).
const THU_NOON = new Date(2026, 5, 18, 12, 0, 0).getTime()

const secondsAt = (year: number, month: number, day: number, hour = 10) =>
  Math.floor(new Date(year, month, day, hour, 0, 0).getTime() / 1000)

describe('nominalDayStart', () => {
  it('rolls the day boundary at 4 AM, not midnight', () => {
    // 1 AM Saturday still belongs to Friday's run.
    expect(nominalDayStart(new Date(2026, 5, 20, 1, 30).getTime())).toBe(new Date(2026, 5, 19).getTime())
    expect(nominalDayStart(new Date(2026, 5, 20, 4, 30).getTime())).toBe(new Date(2026, 5, 20).getTime())
  })
})

describe('calendarBucket', () => {
  // Monday week start: the current week began Mon 15 Jun, last week is Jun 8-14.
  const MONDAY = 1

  const kindAt = (year: number, month: number, day: number, hour = 10) =>
    calendarBucket(secondsAt(year, month, day, hour), THU_NOON, MONDAY).kind

  it('buckets the current day (and, defensively, the future) as today', () => {
    // The head run normally absorbs these; "Earlier today" covers the rest.
    expect(kindAt(2026, 5, 18, 5)).toBe('today')
    expect(kindAt(2026, 5, 18, 23)).toBe('today')
    expect(kindAt(2026, 5, 19)).toBe('today')
  })

  it('assigns the small hours to the previous evening', () => {
    // 1 AM today (before the 4 AM rollover) is part of yesterday's run.
    expect(kindAt(2026, 5, 18, 1)).toBe('yesterday')

    // And viewed at 00:58, last evening's sessions are still the current day.
    const smallHours = new Date(2026, 5, 19, 0, 58).getTime()

    expect(calendarBucket(secondsAt(2026, 5, 18, 23), smallHours, MONDAY).kind).toBe('today')
    expect(calendarBucket(secondsAt(2026, 5, 18, 10), smallHours, MONDAY).kind).toBe('today')
    expect(calendarBucket(secondsAt(2026, 5, 17, 15), smallHours, MONDAY).kind).toBe('yesterday')
  })

  it('uses coarse, non-overlapping ranges that coarsen with age', () => {
    expect(kindAt(2026, 5, 17)).toBe('yesterday')
    expect(kindAt(2026, 5, 16)).toBe('thisWeek') // Tue this week
    expect(kindAt(2026, 5, 15)).toBe('thisWeek') // Mon this week
    expect(kindAt(2026, 5, 14)).toBe('lastWeek') // Sun last week
    expect(kindAt(2026, 5, 8)).toBe('lastWeek') // Mon last week
    expect(kindAt(2026, 5, 7)).toBe('thisMonth') // earlier in June
    expect(kindAt(2026, 5, 1)).toBe('thisMonth')
    expect(kindAt(2026, 4, 28)).toBe('month') // May, same year
    expect(kindAt(2025, 11, 3)).toBe('monthYear') // December, prior year
  })

  it('respects a Sunday week start', () => {
    // With the week starting Sun 14 Jun, that Sunday is this week, not last.
    expect(calendarBucket(secondsAt(2026, 5, 14), THU_NOON, 0).kind).toBe('thisWeek')
    expect(calendarBucket(secondsAt(2026, 5, 13), THU_NOON, 0).kind).toBe('lastWeek')
  })

  it('keys same-month sessions together and disambiguates across years', () => {
    expect(calendarBucket(secondsAt(2026, 2, 3), THU_NOON, MONDAY).key).toBe('m-2026-2')
    expect(calendarBucket(secondsAt(2026, 2, 20), THU_NOON, MONDAY).key).toBe('m-2026-2')
    expect(calendarBucket(secondsAt(2025, 2, 3), THU_NOON, MONDAY).key).toBe('my-2025-2')
  })
})

describe('sessionBucketLabel', () => {
  const labels = {
    lastWeek: 'Last week',
    thisMonth: 'Earlier this month',
    thisWeek: 'Earlier this week',
    today: 'Earlier today',
    yesterday: 'Yesterday'
  }

  const labelAt = (year: number, month: number, day: number) =>
    sessionBucketLabel(calendarBucket(secondsAt(year, month, day), THU_NOON, 1), labels)

  it('uses fixed labels for the relative buckets', () => {
    expect(labelAt(2026, 5, 18)).toBe('Earlier today')
    expect(labelAt(2026, 5, 17)).toBe('Yesterday')
    expect(labelAt(2026, 5, 16)).toBe('Earlier this week')
    expect(labelAt(2026, 5, 10)).toBe('Last week')
    expect(labelAt(2026, 5, 2)).toBe('Earlier this month')
  })

  it('formats month (same year) and month + year (prior year) via Intl', () => {
    // en-US default in the test env: month name, plus year for the prior year.
    expect(labelAt(2026, 2, 3)).toBe('March')
    expect(labelAt(2025, 11, 3)).toBe('December 2025')
  })
})
