import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { HUD_CLOSE_GRACE_MS, type HudCloseWindowLike, requestHudClose } from './hud-close'

function fakeWindow() {
  let destroyed = false
  let closedListener: (() => void) | null = null

  const win: HudCloseWindowLike & { closeCalls: number; destroyCalls: number; emitClosed(): void } = {
    closeCalls: 0,
    destroyCalls: 0,
    close() {
      this.closeCalls += 1
    },
    destroy() {
      this.destroyCalls += 1
      destroyed = true
      closedListener?.()
    },
    isDestroyed: () => destroyed,
    once(_event, listener) {
      closedListener = listener
    },
    emitClosed() {
      destroyed = true
      closedListener?.()
    }
  }

  return win
}

test('a renderer that answers the close is never destroyed', () => {
  vi.useFakeTimers()

  try {
    const win = fakeWindow()

    requestHudClose(win)
    assert.equal(win.closeCalls, 1)

    win.emitClosed()
    vi.advanceTimersByTime(HUD_CLOSE_GRACE_MS * 2)

    assert.equal(win.destroyCalls, 0)
  } finally {
    vi.useRealTimers()
  }
})

test('a renderer that never answers is destroyed at the grace deadline', () => {
  vi.useFakeTimers()

  try {
    const win = fakeWindow()

    requestHudClose(win)
    vi.advanceTimersByTime(HUD_CLOSE_GRACE_MS - 1)
    assert.equal(win.destroyCalls, 0)

    vi.advanceTimersByTime(1)
    assert.equal(win.destroyCalls, 1)
    assert.equal(win.isDestroyed(), true)
  } finally {
    vi.useRealTimers()
  }
})
