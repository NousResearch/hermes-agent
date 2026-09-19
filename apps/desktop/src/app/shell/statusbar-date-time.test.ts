import { describe, expect, it } from 'vitest'

import { formatStatusbarDateTime } from './statusbar-date-time'

describe('status bar date and time', () => {
  it('formats the local calendar date and clock time as separate readouts', () => {
    const value = new Date('2026-09-19T09:05:00.000Z')
    const parts = formatStatusbarDateTime(value, 'en-GB', 'UTC')

    expect(parts.date).toContain('Sat')
    expect(parts.date).toContain('19')
    expect(parts.date).toMatch(/Sep/)
    expect(parts.time).toBe('09:05')
  })
})
