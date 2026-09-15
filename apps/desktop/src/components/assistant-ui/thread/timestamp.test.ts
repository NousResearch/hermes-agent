import { describe, expect, it } from 'vitest'

import {
  formatMessageTimestamp,
  formatTimelineDuration,
  formatTimelineRange,
  formatTimelineTimestamp
} from './timestamp'

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

    expect(formatted).toMatch(/13|1/)
    expect(formatted).toContain('02')
    expect(formatted).toContain('03')
    expect(formatted).toContain('456')
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

describe('formatTimelineDuration', () => {
  const start = 1_778_000_000

  it('renders a settled event as its duration, not a start → end range', () => {
    expect(formatTimelineDuration(start, start + 21)).toBe('21s')
    expect(formatTimelineDuration(start, start + 125)).toBe('2m 5s')
  })

  it('collapses sub-second steps instead of printing a full range', () => {
    // The 0.277s step the issue screenshots as a whole "11:09:01.682 AM →
    // 11:09:01.959 AM" range at reply weight.
    expect(formatTimelineDuration(start, start + 0.277)).toBe('<1s')
    expect(formatTimelineDuration(start, start)).toBe('<1s')
  })

  it('returns an empty string unless both boundaries are usable and ordered', () => {
    expect(formatTimelineDuration(undefined, start + 21)).toBe('')
    expect(formatTimelineDuration(start, undefined)).toBe('')
    expect(formatTimelineDuration(start, start - 5)).toBe('')
  })
})
