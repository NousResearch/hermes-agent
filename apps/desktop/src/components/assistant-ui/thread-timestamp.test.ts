import { describe, expect, it } from 'vitest'

import { formatMessageTimestamp } from './thread-timestamp'

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
