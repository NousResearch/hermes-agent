import assert from 'node:assert/strict'

import { test } from 'vitest'

import { restoreMainWindowSurface, shouldArmHudRestore } from './hud-restore'

// #88513: the arm decision must NOT depend on visibility — a minimized or
// hidden main window still needs a surface back when the HUD closes.

test('arms restore for a live main window regardless of visibility', () => {
  assert.equal(shouldArmHudRestore({ isDestroyed: () => false }), true)
})

test('does not arm restore without a live main window', () => {
  assert.equal(shouldArmHudRestore(null), false)
  assert.equal(shouldArmHudRestore(undefined), false)
  assert.equal(shouldArmHudRestore({ isDestroyed: () => true }), false)
})

test('restore routes through the focus ladder when armed', () => {
  const win = { isDestroyed: () => false }
  let focused: unknown = null

  const restored = restoreMainWindowSurface(true, win, target => {
    focused = target
  })

  assert.equal(restored, true)
  assert.equal(focused, win)
})

test('restore is a no-op when not armed', () => {
  const restored = restoreMainWindowSurface(false, { isDestroyed: () => false }, () =>
    assert.fail('must not focus when unarmed')
  )

  assert.equal(restored, false)
})
