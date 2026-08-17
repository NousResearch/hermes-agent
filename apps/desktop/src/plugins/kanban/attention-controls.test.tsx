import { describe, expect, it } from 'vitest'

import { formatLocalDateTime, parseLocalDateTime } from './datetime-local'

describe('attention local wake fields', () => {
  it.each([
    ['UTC', '2026-01-02T03:04'],
    ['America/Chicago', '2026-01-01T21:04'],
    ['Asia/Kathmandu', '2026-01-02T08:49']
  ])('formats local wall time in %s', (tz, expected) => {
    process.env.TZ = tz
    expect(formatLocalDateTime(new Date('2026-01-02T03:04:00Z'))).toBe(expected)
  })

  it('rejects normalized calendar values and DST gaps', () => {
    process.env.TZ = 'America/Chicago'
    expect(parseLocalDateTime('2026-02-30T09:00')).toBeNull()
    expect(parseLocalDateTime('2026-03-08T02:30')).toBeNull()
  })

  it('uses the earlier DST-fold occurrence and preserves its wall-clock field', () => {
    process.env.TZ = 'America/Chicago'
    const folded = parseLocalDateTime('2026-11-01T01:30')
    expect(folded).not.toBeNull()
    expect(formatLocalDateTime(folded!)).toBe('2026-11-01T01:30')
    expect(folded!.toISOString()).toBe('2026-11-01T06:30:00.000Z')
  })
})
