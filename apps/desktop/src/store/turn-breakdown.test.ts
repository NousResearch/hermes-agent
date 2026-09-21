import { beforeEach, describe, expect, it } from 'vitest'

import {
  $turnBreakdownBySession,
  addToolSeconds,
  beginTurnBreakdown,
  clearTurnBreakdown,
  endTurnBreakdown
} from './turn-breakdown'

describe('turn breakdown store', () => {
  beforeEach(() => {
    $turnBreakdownBySession.set({})
  })

  it('tracks one turn per session without cross-talk', () => {
    beginTurnBreakdown('a')

    expect($turnBreakdownBySession.get().a?.startedAt).toBeGreaterThan(0)
    expect($turnBreakdownBySession.get().b).toBeUndefined()

    addToolSeconds('a', 3.5)
    addToolSeconds('a', 1.25)
    endTurnBreakdown('a')

    const a = $turnBreakdownBySession.get().a

    expect(a?.toolSeconds).toBeCloseTo(4.75)
    // Same-millisecond begin/end is legitimate (vi runners can be that fast),
    // so the clock invariant is ">=", not ">".
    expect(a?.completedAt ?? 0).toBeGreaterThanOrEqual(a?.startedAt ?? 0)
  })

  it('resets the accumulation window on a new turn', () => {
    beginTurnBreakdown('a')
    addToolSeconds('a', 10)
    endTurnBreakdown('a')

    beginTurnBreakdown('a')

    expect($turnBreakdownBySession.get().a?.toolSeconds).toBe(0)
    expect($turnBreakdownBySession.get().a?.completedAt).toBeNull()
  })

  it('ignores absent, negative, and non-finite tool durations', () => {
    beginTurnBreakdown('a')
    addToolSeconds('a', Number.NaN)
    addToolSeconds('a', -2)
    addToolSeconds('a', 0)

    expect($turnBreakdownBySession.get().a?.toolSeconds).toBe(0)
  })

  it('drops a session cleanly', () => {
    beginTurnBreakdown('a')
    clearTurnBreakdown('a')

    expect($turnBreakdownBySession.get().a).toBeUndefined()
  })
})
