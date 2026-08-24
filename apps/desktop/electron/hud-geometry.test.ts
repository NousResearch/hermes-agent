import assert from 'node:assert/strict'

import { test } from 'vitest'

import { applyHudResetBounds, defaultHudBounds } from './hud-geometry'

test('defaultHudBounds restores the standard centered bottom layout', () => {
  assert.deepEqual(defaultHudBounds({ x: 0, y: 25, width: 1440, height: 875 }), {
    x: 410,
    y: 508,
    width: 620,
    height: 320
  })
})

test('defaultHudBounds fits the default layout to a small work area', () => {
  assert.deepEqual(defaultHudBounds({ x: -800, y: 0, width: 480, height: 240 }), {
    x: -800,
    y: 0,
    width: 480,
    height: 240
  })
})

test('defaultHudBounds keeps the spawn fallback when no display is available', () => {
  assert.deepEqual(defaultHudBounds(), { x: undefined, y: undefined, width: 620, height: 320 })
})

test('applyHudResetBounds restores the resize lock and reports native failure', () => {
  let resizable = false
  const win = {
    isDestroyed: () => false,
    isResizable: () => resizable,
    setResizable: (value: boolean) => {
      resizable = value
    },
    setBounds: () => {
      throw new Error('window disappeared')
    }
  }

  assert.equal(applyHudResetBounds(win, { x: 0, y: 0, width: 620, height: 320 }), false)
  assert.equal(resizable, false)
})

test('applyHudResetBounds flips resizable on while the size changes', () => {
  let resizable = false
  const applied: Array<{ height: number; width: number }> = []
  const win = {
    isDestroyed: () => false,
    isResizable: () => resizable,
    setResizable: (value: boolean) => {
      resizable = value
    },
    setBounds: (bounds: { height: number; width: number }) => {
      assert.equal(resizable, true)
      applied.push({ width: bounds.width, height: bounds.height })
    }
  }

  assert.equal(applyHudResetBounds(win, { x: 10, y: 20, width: 620, height: 320 }), true)
  assert.equal(resizable, false)
  assert.deepEqual(applied, [{ width: 620, height: 320 }])
})
