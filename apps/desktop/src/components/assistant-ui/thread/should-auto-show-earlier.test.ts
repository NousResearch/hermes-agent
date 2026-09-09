import { describe, expect, it } from 'vitest'

import { resolveShowEarlierAction, shouldAutoShowEarlier, TOP_EDGE_PX } from './transcript-window'

const settledTop = {
  action: 'dom' as const,
  isAtBottom: false,
  loadSettled: true,
  restorePending: false,
  scrollTop: 0
}

describe('shouldAutoShowEarlier', () => {
  it('pages only for a settled reader at the top edge with older content to show', () => {
    // Reading intent: a scroll into the top edge, or an upward wheel while
    // already clamped there (browsers emit no scroll event at scrollTop 0).
    expect(shouldAutoShowEarlier(settledTop)).toBe(true)
    expect(shouldAutoShowEarlier({ ...settledTop, scrollTop: TOP_EDGE_PX })).toBe(true)
    expect(shouldAutoShowEarlier({ ...settledTop, wheelDeltaY: -40 })).toBe(true)
    expect(shouldAutoShowEarlier({ ...settledTop, action: resolveShowEarlierAction(0, true) })).toBe(true)

    // Every gate that must keep a page from loading on its own.
    expect(shouldAutoShowEarlier({ ...settledTop, action: resolveShowEarlierAction(0, false) })).toBe(false)
    expect(shouldAutoShowEarlier({ ...settledTop, loadSettled: false })).toBe(false)
    expect(shouldAutoShowEarlier({ ...settledTop, restorePending: true })).toBe(false)
    expect(shouldAutoShowEarlier({ ...settledTop, isAtBottom: true })).toBe(false)
    expect(shouldAutoShowEarlier({ ...settledTop, scrollTop: TOP_EDGE_PX + 1 })).toBe(false)
    expect(shouldAutoShowEarlier({ ...settledTop, wheelDeltaY: 40 })).toBe(false)
    expect(shouldAutoShowEarlier({ ...settledTop, wheelDeltaY: 0 })).toBe(false)
  })
})
