import { describe, expect, it } from 'vitest'

import { formatClockTimestamp, formatMessageTimestamp, formatTimelineRange, formatTimelineTimestamp } from './timestamp'

const labels = {
  today: (time: string) => `Today at ${time}`,
  yesterday: (time: string) => `Yesterday at ${time}`
}

describe('formatMessageTimestamp', () => {
  it('returns an empty string for missing values', () => {
    expect(formatMessageTimestamp(undefined, labels)).toBe('')
    expect(formatMessageTimestamp('not-a-date', labels)).toBe('')
  })

  it('uses the today label for timestamps earlier today', () => {
    const now = new Date()
    const earlierToday = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 30)
    expect(formatMessageTimestamp(earlierToday, labels)).toMatch(/^Today at /)
  })

  it('uses the yesterday label for timestamps the prior day', () => {
    const now = new Date()
    const yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 8, 0)
    yesterday.setDate(yesterday.getDate() - 1)
    expect(formatMessageTimestamp(yesterday, labels)).toMatch(/^Yesterday at /)
  })

  it('falls back to an absolute format for older timestamps', () => {
    const old = new Date(2020, 0, 15, 9, 30)
    const out = formatMessageTimestamp(old, labels)
    expect(out).not.toMatch(/^Today at /)
    expect(out).not.toMatch(/^Yesterday at /)
    expect(out.length).toBeGreaterThan(0)
  })
})

describe('precise timeline timestamps', () => {
  it('includes seconds and milliseconds for an event', () => {
    const local = new Date(2026, 4, 1, 13, 2, 3, 456)
    const formatted = formatTimelineTimestamp(local.getTime() / 1000)

    // Locale-agnostic: under ar-SA this renders "١:٠٢:٠٣٫٤٥٦ م", so match the
    // locale's own output rather than Latin digits.
    expect(formatted).toBe(
      new Intl.DateTimeFormat(undefined, {
        fractionalSecondDigits: 3,
        hour: 'numeric',
        minute: '2-digit',
        second: '2-digit'
      }).format(local)
    )

    // The contract that matters: sub-second detail is preserved, so instants
    // differing only in milliseconds stay distinguishable.
    const oneMsLater = formatTimelineTimestamp(local.getTime() / 1000 + 0.001)
    expect(oneMsLater).not.toBe(formatted)
  })

  it('renders start and finish as a range', () => {
    const start = new Date(2026, 4, 1, 13, 2, 3, 456).getTime() / 1000
    const finish = start + 1.25

    expect(formatTimelineRange(start, finish)).toBe(
      `${formatTimelineTimestamp(start)} → ${formatTimelineTimestamp(finish)}`
    )
  })

  it('returns an empty string for invalid timeline values', () => {
    expect(formatTimelineTimestamp(undefined)).toBe('')
    expect(formatTimelineTimestamp(Number.NaN)).toBe('')
    expect(formatTimelineRange(undefined, 10)).toBe('')
  })
})

describe('formatClockTimestamp', () => {
  it('renders a minute-precision wall clock without seconds or milliseconds', () => {
    const local = new Date(2026, 4, 1, 16, 30, 3, 456)
    const seconds = local.getTime() / 1000

    // Assert against the locale's own digits/day-period rendering while forcing
    // the owner's requested 12-hour clock contract in every locale.
    expect(formatClockTimestamp(seconds)).toBe(
      new Intl.DateTimeFormat(undefined, { hour: 'numeric', hour12: true, minute: '2-digit' }).format(local)
    )

    // The distinguishing contract: two instants in the same minute collapse to
    // one label, while the precise formatter still separates them.
    const laterInSameMinute = new Date(2026, 4, 1, 16, 30, 44, 789).getTime() / 1000
    expect(formatClockTimestamp(laterInSameMinute)).toBe(formatClockTimestamp(seconds))
    expect(formatTimelineTimestamp(laterInSameMinute)).not.toBe(formatTimelineTimestamp(seconds))
  })

  it('returns an empty string for invalid values', () => {
    expect(formatClockTimestamp(undefined)).toBe('')
    expect(formatClockTimestamp(Number.NaN)).toBe('')
    expect(formatClockTimestamp(0)).toBe('')
    expect(formatClockTimestamp(-5)).toBe('')
  })
})
