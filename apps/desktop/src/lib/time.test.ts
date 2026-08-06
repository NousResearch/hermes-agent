import { describe, expect, it } from 'vitest'

import {
  CALENDAR_WEEKS_BEFORE_MONTHS,
  calendarBucket,
  DAY,
  formatAgo,
  HOUR,
  localeWeekStartDay,
  MINUTE,
  nominalDayStart,
  SECOND,
  sessionBucketLabel
} from './time'

const agoLabels = {
  ageNow: 'now',
  ageSeconds: (s: number) => `${s}s ago`,
  ageMinutes: (m: number) => `${m}m ago`,
  ageHours: (h: number) => `${h}h ago`,
  ageDays: (d: number) => `${d}d ago`
}

const elapsedNow = 1_000 * DAY
const ago = (delta: number) => formatAgo(elapsedNow - delta, agoLabels, elapsedNow)

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

describe('nominalDayStart', () => {
  it('keeps the existing 4 AM thread-day helper unchanged', () => {
    expect(nominalDayStart(new Date(2026, 5, 20, 1, 30).getTime())).toBe(new Date(2026, 5, 19).getTime())
    expect(nominalDayStart(new Date(2026, 5, 20, 4, 30).getTime())).toBe(new Date(2026, 5, 20).getTime())
  })
})

// Friday 31 July 2026, local noon. The current French week is Mon 27 Jul–Sun 2 Aug.
const NOW = new Date(2026, 6, 31, 12, 0, 0).getTime()
const MONDAY = 1

const at = (year: number, month: number, day: number, hour = 10) =>
  Math.floor(new Date(year, month, day, hour, 0, 0).getTime() / SECOND)

const bucketAt = (year: number, month: number, day: number, hour = 10) =>
  calendarBucket(at(year, month, day, hour), NOW, MONDAY)

const frLabels = {
  lastWeek: (range: string) => `Semaine dernière · ${range}`,
  today: 'Aujourd’hui',
  week: (range: string) => `Semaine du ${range}`,
  yesterday: 'Hier'
}

const normalized = (value: string) => value.replace(/\s+/g, ' ').replace(/\u202f/g, ' ')

describe('calendarBucket', () => {
  it('uses exact today and yesterday calendar groups', () => {
    expect(bucketAt(2026, 6, 31)).toMatchObject({ key: 'day:2026-07-31', kind: 'today' })
    expect(bucketAt(2026, 6, 30)).toMatchObject({ key: 'day:2026-07-30', kind: 'yesterday' })
  })

  it('normalizes future activity to the current local day and Today key', () => {
    expect(bucketAt(2026, 7, 3)).toMatchObject({
      at: new Date(2026, 6, 31).getTime(),
      key: 'day:2026-07-31',
      kind: 'today'
    })
  })

  it('uses one stable group per remaining day in the current week', () => {
    expect(bucketAt(2026, 6, 29)).toMatchObject({ key: 'day:2026-07-29', kind: 'day' })
    expect(bucketAt(2026, 6, 27)).toMatchObject({ key: 'day:2026-07-27', kind: 'day' })
  })

  it('uses last week plus three earlier complete weeks before months', () => {
    expect(CALENDAR_WEEKS_BEFORE_MONTHS).toBe(4)
    expect(bucketAt(2026, 6, 26)).toMatchObject({ key: 'week:2026-07-20', kind: 'lastWeek' })
    expect(bucketAt(2026, 6, 13)).toMatchObject({ key: 'week:2026-07-13', kind: 'week' })
    expect(bucketAt(2026, 5, 29)).toMatchObject({ key: 'week:2026-06-29', kind: 'week' })
    expect(bucketAt(2026, 5, 28)).toMatchObject({ key: 'month:2026-06', kind: 'month' })
  })

  it('uses real calendar midnight instead of the separate 4 AM thread-day rule', () => {
    expect(bucketAt(2026, 6, 31, 1).kind).toBe('today')
    expect(bucketAt(2026, 6, 30, 23).kind).toBe('yesterday')
  })

  it('handles French Monday weeks, month/year boundaries and leap day', () => {
    expect(localeWeekStartDay('fr-FR')).toBe(1)

    const janNow = new Date(2027, 0, 1, 12).getTime()
    expect(calendarBucket(at(2026, 11, 30), janNow, MONDAY)).toMatchObject({
      key: 'day:2026-12-30',
      kind: 'day'
    })

    const marchNow = new Date(2024, 2, 1, 12).getTime()
    expect(calendarBucket(at(2024, 1, 29), marchNow, MONDAY)).toMatchObject({
      key: 'day:2024-02-29',
      kind: 'yesterday'
    })
  })

  it('keeps a date key stable when today becomes yesterday after midnight', () => {
    const activity = at(2026, 6, 31, 12)
    const today = calendarBucket(activity, new Date(2026, 6, 31, 23, 59).getTime(), MONDAY)
    const tomorrow = calendarBucket(activity, new Date(2026, 7, 1, 0, 1).getTime(), MONDAY)

    expect(today).toMatchObject({ key: 'day:2026-07-31', kind: 'today' })
    expect(tomorrow).toMatchObject({ key: 'day:2026-07-31', kind: 'yesterday' })
  })

  it('keeps technical keys independent from translated labels', () => {
    const bucket = bucketAt(2026, 6, 13)
    const french = sessionBucketLabel(bucket, frLabels, 'fr-FR', NOW)

    const english = sessionBucketLabel(
      bucket,
      {
        lastWeek: range => `Last week · ${range}`,
        today: 'Today',
        week: range => `Week of ${range}`,
        yesterday: 'Yesterday'
      },
      'en-US',
      NOW
    )

    expect(bucket.key).toBe('week:2026-07-13')
    expect(french).not.toBe(english)
    expect(bucket.key).toBe('week:2026-07-13')
  })
})

describe('sessionBucketLabel', () => {
  it('formats explicit French day, week and month labels naturally', () => {
    expect(normalized(sessionBucketLabel(bucketAt(2026, 6, 31), frLabels, 'fr-FR', NOW))).toBe('Aujourd’hui')
    expect(normalized(sessionBucketLabel(bucketAt(2026, 6, 30), frLabels, 'fr-FR', NOW))).toBe('Hier')
    expect(normalized(sessionBucketLabel(bucketAt(2026, 6, 29), frLabels, 'fr-FR', NOW))).toBe('mercredi 29 juillet')
    expect(normalized(sessionBucketLabel(bucketAt(2026, 6, 26), frLabels, 'fr-FR', NOW))).toBe(
      'Semaine dernière · 20–26 juillet'
    )
    expect(normalized(sessionBucketLabel(bucketAt(2026, 5, 29), frLabels, 'fr-FR', NOW))).toBe(
      'Semaine du 29 juin – 5 juillet'
    )
    expect(normalized(sessionBucketLabel(bucketAt(2026, 5, 28), frLabels, 'fr-FR', NOW))).toBe('juin 2026')
  })

  it('includes years when a complete week crosses a year boundary', () => {
    const boundaryNow = new Date(2027, 0, 29, 12).getTime()
    const bucket = calendarBucket(at(2026, 11, 30), boundaryNow, MONDAY)
    const label = normalized(sessionBucketLabel(bucket, frLabels, 'fr-FR', boundaryNow))

    expect(bucket).toMatchObject({ key: 'week:2026-12-28', kind: 'week' })
    expect(label).toContain('2026')
    expect(label).toContain('2027')
  })
})
