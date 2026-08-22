import { describe, expect, it } from 'vitest'

import {
  calendarBucket,
  DAY,
  fmtMonth,
  fmtMonthYear,
  formatAgo,
  HOUR,
  MINUTE,
  nominalDayStart,
  SECOND,
  sessionBucketLabel
} from './time'

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
    expect(kindAt(2026, 5, 18, 1)).toBe('today')
    expect(kindAt(2026, 5, 18, 23)).toBe('today')
    expect(kindAt(2026, 5, 19)).toBe('today')
  })

  it('uses daily groups for the current week, then four calendar weeks, then months', () => {
    expect(kindAt(2026, 5, 17)).toBe('yesterday')
    expect(kindAt(2026, 5, 16)).toBe('day') // Tue this week
    expect(kindAt(2026, 5, 15)).toBe('day') // Mon this week
    expect(kindAt(2026, 5, 14)).toBe('lastWeek') // Sun last week
    expect(kindAt(2026, 5, 8)).toBe('lastWeek') // Mon last week
    expect(kindAt(2026, 5, 7)).toBe('week')
    expect(kindAt(2026, 4, 18)).toBe('week') // fourth full week back
    expect(kindAt(2026, 4, 11)).toBe('month') // older than the four-week window
    expect(kindAt(2025, 11, 3)).toBe('monthYear') // December, prior year
  })

  it('respects a Sunday week start', () => {
    // With the week starting Sun 14 Jun, that Sunday is this week, not last.
    expect(calendarBucket(secondsAt(2026, 5, 14), THU_NOON, 0).kind).toBe('day')
    expect(calendarBucket(secondsAt(2026, 5, 13), THU_NOON, 0).kind).toBe('lastWeek')
  })

  it('uses stable technical keys for days, weeks, and months', () => {
    expect(calendarBucket(secondsAt(2026, 5, 18), THU_NOON, MONDAY).key).toBe('day:2026-06-18')
    expect(calendarBucket(secondsAt(2026, 5, 10), THU_NOON, MONDAY).key).toBe('week:2026-06-08')
    expect(calendarBucket(secondsAt(2026, 2, 3), THU_NOON, MONDAY).key).toBe('month:2026-03')
    expect(calendarBucket(secondsAt(2026, 2, 20), THU_NOON, MONDAY).key).toBe('month:2026-03')
    expect(calendarBucket(secondsAt(2025, 2, 3), THU_NOON, MONDAY).key).toBe('month:2025-03')
  })
})

describe('sessionBucketLabel', () => {
  const labels = {
    lastWeek: 'Last week',
    thisMonth: 'Earlier this month',
    thisWeek: 'Earlier this week',
    today: 'Today',
    yesterday: 'Yesterday'
  }

  const labelAt = (year: number, month: number, day: number) =>
    sessionBucketLabel(calendarBucket(secondsAt(year, month, day), THU_NOON, 1), labels, 'en-US', THU_NOON)

  it('uses relative labels, named current-week days, and labelled week ranges', () => {
    expect(labelAt(2026, 5, 18)).toBe('Today')
    expect(labelAt(2026, 5, 17)).toBe('Yesterday')
    expect(labelAt(2026, 5, 16)).toBe('Tuesday, June 16')
    expect(labelAt(2026, 5, 10)).toBe('Last week')
    expect(labelAt(2026, 5, 2)).toMatch(/^June 1/)
  })

  it('formats month (same year) and month + year (prior year) via Intl', () => {
    // Locale-agnostic contract: same-year month buckets render via fmtMonth,
    // prior-year buckets via fmtMonthYear. Assert against the shared
    // formatters instead of frozen en-US strings so the test passes under
    // any host locale (the formatters intentionally use the runtime locale).
    const monthBucket = calendarBucket(secondsAt(2026, 2, 3), THU_NOON, 1)

    if (monthBucket.kind !== 'month') {
      throw new Error(`expected month bucket, got ${monthBucket.kind}`)
    }

    expect(sessionBucketLabel(monthBucket, labels)).toBe(fmtMonth.format(monthBucket.at))

    const monthYearBucket = calendarBucket(secondsAt(2025, 11, 3), THU_NOON, 1)

    if (monthYearBucket.kind !== 'monthYear') {
      throw new Error(`expected monthYear bucket, got ${monthYearBucket.kind}`)
    }

    expect(sessionBucketLabel(monthYearBucket, labels)).toBe(fmtMonthYear.format(monthYearBucket.at))
  })
})
