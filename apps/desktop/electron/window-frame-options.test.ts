import assert from 'node:assert/strict'

import { test } from 'vitest'

import { windowTitleBarStyle } from './window-frame-options'

test('uses the native title bar on WSL', () => {
  assert.equal(windowTitleBarStyle(true), 'default')
})

test('preserves the hidden title bar outside WSL', () => {
  assert.equal(windowTitleBarStyle(false), 'hidden')
})
