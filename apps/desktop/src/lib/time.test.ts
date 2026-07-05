import { describe, expect, it } from 'vitest'

import { DAY, formatAgo, HOUR, MINUTE, relativeTime, SECOND } from './time'

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

describe('relativeTime', () => {
  it('rolls to the coarser unit at the top edge of a bucket', () => {
    // 59.5 min rounds to 60 minutes → should read "in 1 hr", not "in 60 min".
    expect(relativeTime(3_570_000, 0)).toBe(relativeTime(HOUR, 0))
    // 59.7 s rounds to 60 seconds → should read "in 1 min", not "in 60 sec".
    expect(relativeTime(59_700, 0)).toBe(relativeTime(MINUTE, 0))
    // 23.5 h rounds to 24 hours → should read the coarser day form, not "in 24 hr".
    expect(relativeTime(84_600_000, 0)).toBe(relativeTime(DAY, 0))
  })

  it('keeps non-boundary values in their own bucket', () => {
    expect(relativeTime(2 * HOUR, 0)).toBe('in 2 hr.')
    expect(relativeTime(5 * MINUTE, 0)).toBe('in 5 min.')
    expect(relativeTime(30 * SECOND, 0)).toBe('in 30 sec.')
  })

  it('preserves the past direction when a value carries to the coarser unit', () => {
    // 59.5 min in the past → "1 hr ago", not "60 min ago".
    expect(relativeTime(0, 3_570_000)).toBe(relativeTime(0, HOUR))
  })
})
