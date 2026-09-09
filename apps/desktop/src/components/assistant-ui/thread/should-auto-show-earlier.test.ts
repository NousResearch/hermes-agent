import { describe, expect, it } from 'vitest'

import { resolveShowEarlierAction, shouldAutoShowEarlier, TOP_EDGE_PX } from './transcript-window'

const settledTop = {
  action: 'dom' as const,
  isAtBottom: false,
  loadSettled: true,
  restorePending: false,
  scrollTop: 0,
  topEdgePx: TOP_EDGE_PX
}

describe('TOP_EDGE_PX', () => {
  it('is a small top-edge slack, not a mid-scroll threshold', () => {
    expect(TOP_EDGE_PX).toBeGreaterThanOrEqual(48)
    expect(TOP_EDGE_PX).toBeLessThanOrEqual(64)
  })
})

describe('shouldAutoShowEarlier', () => {
  it('loads earlier when the reader is at the top edge with a DOM page to spend', () => {
    expect(shouldAutoShowEarlier({ ...settledTop, action: 'dom' })).toBe(true)
  })

  it('loads earlier when the reader is at the top edge with a store window to expand', () => {
    expect(
      shouldAutoShowEarlier({
        ...settledTop,
        action: resolveShowEarlierAction(0, true)
      })
    ).toBe(true)
  })

  it('treats scrollTop at the edge pixel as top-edge reading intent', () => {
    expect(shouldAutoShowEarlier({ ...settledTop, scrollTop: TOP_EDGE_PX })).toBe(true)
  })

  it('loads earlier on an upward wheel while already at the top edge', () => {
    expect(shouldAutoShowEarlier({ ...settledTop, wheelDeltaY: -40 })).toBe(true)
  })

  it('does not fire when resolveShowEarlierAction is a no-op', () => {
    expect(shouldAutoShowEarlier({ ...settledTop, action: null })).toBe(false)
    expect(
      shouldAutoShowEarlier({
        ...settledTop,
        action: resolveShowEarlierAction(0, false)
      })
    ).toBe(false)
  })

  it('does not fire until the session load has settled', () => {
    expect(shouldAutoShowEarlier({ ...settledTop, loadSettled: false })).toBe(false)
  })

  it('does not fire while a prepend restore is still pending', () => {
    expect(shouldAutoShowEarlier({ ...settledTop, restorePending: true })).toBe(false)
  })

  it('does not fire while the reader is following the bottom', () => {
    expect(shouldAutoShowEarlier({ ...settledTop, isAtBottom: true })).toBe(false)
  })

  it('does not fire in the middle of the transcript', () => {
    expect(shouldAutoShowEarlier({ ...settledTop, scrollTop: TOP_EDGE_PX + 1 })).toBe(false)
    expect(shouldAutoShowEarlier({ ...settledTop, scrollTop: 400 })).toBe(false)
  })

  it('does not treat a downward or zero wheel at the top as earlier-reading intent', () => {
    expect(shouldAutoShowEarlier({ ...settledTop, wheelDeltaY: 40 })).toBe(false)
    expect(shouldAutoShowEarlier({ ...settledTop, wheelDeltaY: 0 })).toBe(false)
  })
})
