/**
 * Regression for the bootstrap installer's stage timers (issue report:
 * "it says 744 hours or minutes — I don't know how to adjust the counter").
 *
 * ``formatElapsed`` rendered a running stage as ``m:ss`` with unbounded
 * minutes: a node-deps stage left hanging overnight showed ``744:38`` (12h24m
 * of minutes), which the user read as 744 hours. ``formatDuration`` (completed
 * stages) had the same unbounded-minutes shape. These pin the hour rollover.
 */
import { describe, expect, it } from 'vitest'

import {
  formatDuration,
  formatElapsed,
} from '../apps/bootstrap-installer/src/lib/format'

const SEC = 1000
const MIN = 60 * SEC
const HOUR = 60 * MIN

describe('formatElapsed (live stage timer)', () => {
  it('renders bare seconds under a minute', () => {
    expect(formatElapsed(0)).toBe('0s')
    expect(formatElapsed(59 * SEC)).toBe('59s')
  })

  it('renders m:ss between a minute and an hour', () => {
    expect(formatElapsed(MIN)).toBe('1:00')
    expect(formatElapsed(12 * MIN + 38 * SEC)).toBe('12:38')
    expect(formatElapsed(59 * MIN + 59 * SEC)).toBe('59:59')
  })

  it('rolls over to h:mm:ss past an hour', () => {
    expect(formatElapsed(HOUR)).toBe('1:00:00')
    // The overnight-hang report: 744 minutes 38 seconds must NOT render as
    // "744:38".
    expect(formatElapsed(744 * MIN + 38 * SEC)).toBe('12:24:38')
  })

  it('clamps negative input (clock skew) to zero', () => {
    expect(formatElapsed(-5 * SEC)).toBe('0s')
  })
})

describe('formatDuration (completed stage)', () => {
  it('keeps the sub-hour shapes', () => {
    expect(formatDuration(999)).toBe('999ms')
    expect(formatDuration(1500)).toBe('1.5s')
    expect(formatDuration(2 * MIN + 5 * SEC)).toBe('2m 5s')
  })

  it('rolls over to hours past 60 minutes', () => {
    expect(formatDuration(HOUR)).toBe('1h 0m')
    expect(formatDuration(12 * HOUR + 24 * MIN)).toBe('12h 24m')
  })
})
