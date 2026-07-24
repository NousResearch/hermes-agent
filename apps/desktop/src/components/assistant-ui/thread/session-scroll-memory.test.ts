// Tests for the per-session scroll memory behind #70101: sessions the user
// left at the bottom restore sticky-bottom; sessions left mid-read restore
// their exact distance-from-bottom, resilient to content-height changes.
import { beforeEach, describe, expect, it } from 'vitest'

import {
  BOTTOM,
  distanceFromBottom,
  recallSessionScroll,
  recordSessionScroll,
  resetSessionScrollMemory,
  SCROLL_MEMORY_LIMIT,
  stateFromMetrics,
  STICKY_BOTTOM_THRESHOLD_PX,
  targetScrollTop
} from './session-scroll-memory'

beforeEach(() => resetSessionScrollMemory())

describe('distanceFromBottom', () => {
  it('measures the gap between the viewport bottom and the content bottom', () => {
    expect(distanceFromBottom({ clientHeight: 500, scrollHeight: 2000, scrollTop: 900 })).toBe(600)
  })

  it('is 0 when pinned to the bottom', () => {
    expect(distanceFromBottom({ clientHeight: 500, scrollHeight: 2000, scrollTop: 1500 })).toBe(0)
  })

  it('clamps overscroll bounce below the bottom to 0', () => {
    expect(distanceFromBottom({ clientHeight: 500, scrollHeight: 2000, scrollTop: 1520 })).toBe(0)
  })
})

describe('stateFromMetrics', () => {
  it('treats exact bottom as sticky-bottom', () => {
    expect(stateFromMetrics({ clientHeight: 500, scrollHeight: 2000, scrollTop: 1500 })).toEqual({ kind: 'bottom' })
  })

  it('treats within-threshold as sticky-bottom', () => {
    expect(
      stateFromMetrics({ clientHeight: 500, scrollHeight: 2000, scrollTop: 1500 - STICKY_BOTTOM_THRESHOLD_PX })
    ).toEqual({ kind: 'bottom' })
  })

  it('records an exact offset once past the threshold', () => {
    expect(
      stateFromMetrics({ clientHeight: 500, scrollHeight: 2000, scrollTop: 1499 - STICKY_BOTTOM_THRESHOLD_PX })
    ).toEqual({ fromBottom: STICKY_BOTTOM_THRESHOLD_PX + 1, kind: 'offset' })
  })

  it('honors a custom threshold', () => {
    expect(stateFromMetrics({ clientHeight: 500, scrollHeight: 2000, scrollTop: 1400 }, 100)).toEqual({
      kind: 'bottom'
    })
  })
})

describe('targetScrollTop', () => {
  it('sends sticky-bottom to the max scrollTop', () => {
    expect(targetScrollTop(BOTTOM, { clientHeight: 500, scrollHeight: 2000 })).toBe(1500)
  })

  it('reapplies an offset relative to the bottom at the CURRENT height', () => {
    // Content grew from 2000 → 3000 while away; the reading position keeps
    // its distance from the bottom rather than a stale absolute scrollTop.
    expect(targetScrollTop({ fromBottom: 600, kind: 'offset' }, { clientHeight: 500, scrollHeight: 3000 })).toBe(1900)
  })

  it('clamps an offset larger than the scroll range to the top', () => {
    expect(targetScrollTop({ fromBottom: 9999, kind: 'offset' }, { clientHeight: 500, scrollHeight: 2000 })).toBe(0)
  })

  it('clamps to 0 when content is shorter than the viewport', () => {
    expect(targetScrollTop(BOTTOM, { clientHeight: 500, scrollHeight: 300 })).toBe(0)
    expect(targetScrollTop({ fromBottom: 50, kind: 'offset' }, { clientHeight: 500, scrollHeight: 300 })).toBe(0)
  })
})

describe('recordSessionScroll / recallSessionScroll', () => {
  it('round-trips a recorded state by session key', () => {
    recordSessionScroll('session-a', { fromBottom: 240, kind: 'offset' })
    recordSessionScroll('session-b', BOTTOM)

    expect(recallSessionScroll('session-a')).toEqual({ fromBottom: 240, kind: 'offset' })
    expect(recallSessionScroll('session-b')).toEqual({ kind: 'bottom' })
  })

  it('defaults unknown sessions to sticky-bottom (the pre-fix behavior)', () => {
    expect(recallSessionScroll('never-seen')).toEqual({ kind: 'bottom' })
  })

  it('ignores falsy keys instead of polluting the map', () => {
    recordSessionScroll('', { fromBottom: 100, kind: 'offset' })
    recordSessionScroll(null, { fromBottom: 100, kind: 'offset' })
    recordSessionScroll(undefined, { fromBottom: 100, kind: 'offset' })

    expect(recallSessionScroll('')).toEqual({ kind: 'bottom' })
    expect(recallSessionScroll(null)).toEqual({ kind: 'bottom' })
  })

  it('overwrites on re-record', () => {
    recordSessionScroll('session-a', { fromBottom: 240, kind: 'offset' })
    recordSessionScroll('session-a', BOTTOM)

    expect(recallSessionScroll('session-a')).toEqual({ kind: 'bottom' })
  })

  it('evicts the least-recently-recorded session past the limit', () => {
    for (let i = 0; i < SCROLL_MEMORY_LIMIT; i++) {
      recordSessionScroll(`session-${i}`, { fromBottom: i + 1, kind: 'offset' })
    }

    // Touch session-0 so session-1 becomes the oldest, then overflow.
    recordSessionScroll('session-0', { fromBottom: 777, kind: 'offset' })
    recordSessionScroll('one-too-many', { fromBottom: 42, kind: 'offset' })

    expect(recallSessionScroll('session-1')).toEqual({ kind: 'bottom' })
    expect(recallSessionScroll('session-0')).toEqual({ fromBottom: 777, kind: 'offset' })
    expect(recallSessionScroll('one-too-many')).toEqual({ fromBottom: 42, kind: 'offset' })
  })

  it('resets to empty', () => {
    recordSessionScroll('session-a', { fromBottom: 240, kind: 'offset' })
    resetSessionScrollMemory()

    expect(recallSessionScroll('session-a')).toEqual({ kind: 'bottom' })
  })
})
