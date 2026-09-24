import assert from 'node:assert/strict'

import { test } from 'vitest'

import { applyBoundsWithResizeFlip } from './resize-flip'

function fakeWin(overrides: Partial<{ resizable: boolean; throwOnSetBounds: boolean; destroyed: boolean }> = {}) {
  const state = {
    destroyed: overrides.destroyed ?? false,
    resizable: overrides.resizable ?? false,
    size: [320, 420] as [number, number],
    applied: [] as Array<{ x: number; y: number; width: number; height: number }>
  }

  return {
    state,
    win: {
      getSize: () => state.size,
      isDestroyed: () => state.destroyed,
      isResizable: () => state.resizable,
      setResizable: (value: boolean) => {
        state.resizable = value
      },
      setBounds: (bounds: { x: number; y: number; width: number; height: number }) => {
        if (overrides.throwOnSetBounds) {
          throw new Error('window disappeared')
        }

        state.size = [bounds.width, bounds.height]
        state.applied.push(bounds)
      }
    }
  }
}

test('pure move does not flip resizable', () => {
  const { win, state } = fakeWin({ resizable: false })

  assert.equal(applyBoundsWithResizeFlip(win, { x: 10, y: 20, width: 320, height: 420 }), true)
  assert.equal(state.resizable, false)
  assert.deepEqual(state.applied, [{ x: 10, y: 20, width: 320, height: 420 }])
})

test('size change flips resizable on, then restores the lock', () => {
  const { win, state } = fakeWin({ resizable: false })

  assert.equal(applyBoundsWithResizeFlip(win, { x: 0, y: 0, width: 400, height: 500 }), true)
  assert.equal(state.resizable, false)
  assert.equal(state.applied.length, 1)
})

test('setBounds throw still restores the resize lock (try/finally)', () => {
  const { win, state } = fakeWin({ resizable: false, throwOnSetBounds: true })

  assert.equal(applyBoundsWithResizeFlip(win, { x: 0, y: 0, width: 400, height: 500 }), false)
  // The lock came back even though the native call threw.
  assert.equal(state.resizable, false)
})

test('already-resizable window is left untouched', () => {
  const { win, state } = fakeWin({ resizable: true })

  assert.equal(applyBoundsWithResizeFlip(win, { x: 0, y: 0, width: 400, height: 500 }), true)
  assert.equal(state.resizable, true)
})

test('destroyed window fails closed', () => {
  const { win } = fakeWin({ destroyed: true, resizable: false })

  assert.equal(applyBoundsWithResizeFlip(win, { x: 0, y: 0, width: 400, height: 500 }), false)
})
