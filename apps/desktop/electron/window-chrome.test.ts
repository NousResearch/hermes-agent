import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  CLASSIC_TITLEBAR_HEIGHT,
  PORTAL_TITLEBAR_HEIGHT,
  portalShellPreferred,
  SHELL_CHOICE_KEY,
  windowButtonPosition,
  windowTitlebarHeight
} from './window-chrome'

test('the redesign is on until the user rolls it back', () => {
  assert.equal(portalShellPreferred(null), true)
  assert.equal(portalShellPreferred({}), true)
  assert.equal(portalShellPreferred({ [SHELL_CHOICE_KEY]: 'true' }), true)
  assert.equal(portalShellPreferred({ [SHELL_CHOICE_KEY]: 'false' }), false)
})

test('shell windows follow the shell; the popped-out Browser stays classic', () => {
  assert.equal(windowTitlebarHeight({ kind: 'chat', portal: true }), PORTAL_TITLEBAR_HEIGHT)
  assert.equal(windowTitlebarHeight({ kind: 'chat', portal: false }), CLASSIC_TITLEBAR_HEIGHT)
  assert.equal(windowTitlebarHeight({ kind: 'browser', portal: true }), CLASSIC_TITLEBAR_HEIGHT)
})

test('traffic lights sit centred in whichever band the window draws', () => {
  for (const height of [CLASSIC_TITLEBAR_HEIGHT, PORTAL_TITLEBAR_HEIGHT]) {
    const { y } = windowButtonPosition(height)

    assert.ok(Math.abs(y + 7 - height / 2) <= 0.5)
  }

  assert.deepEqual(windowButtonPosition(CLASSIC_TITLEBAR_HEIGHT), { x: 12, y: 10 })
  assert.deepEqual(windowButtonPosition(PORTAL_TITLEBAR_HEIGHT), { x: 23, y: 23 })
})
