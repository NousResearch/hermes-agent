import { describe, expect, it } from 'vitest'

import { formatDurationSeconds } from './format'

describe('formatDurationSeconds', () => {
  it('renders sub-minute durations that round up to 60s as "1m"', () => {
    // The sub-minute branch formats with toFixed(0), which rounds up. Gating on
    // raw seconds let [59.5, 60) pass the "under a minute" gate yet render an
    // out-of-range "60s" label. Round-then-gate keeps the boundary consistent.
    expect(formatDurationSeconds(59.5)).toBe('1m')
    expect(formatDurationSeconds(59.9)).toBe('1m')
    expect(formatDurationSeconds(59.99)).toBe('1m')
  })

  it('keeps the existing labels below the rounding boundary', () => {
    expect(formatDurationSeconds(0.5)).toBe('500ms')
    expect(formatDurationSeconds(5.4)).toBe('5.4s')
    expect(formatDurationSeconds(59.4)).toBe('59s')
  })

  it('keeps the minute and higher labels unchanged', () => {
    expect(formatDurationSeconds(60)).toBe('1m')
    expect(formatDurationSeconds(90)).toBe('1m 30s')
  })
})
