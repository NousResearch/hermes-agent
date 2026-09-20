import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createTrayLifecycle, readCloseToTraySetting } from './tray-lifecycle'

test('reads only an explicit desktop.close_to_tray true value', () => {
  assert.equal(readCloseToTraySetting('desktop:\n  close_to_tray: true\n'), true)
  assert.equal(readCloseToTraySetting('desktop:\n  close_to_tray: false\n'), false)
  assert.equal(readCloseToTraySetting('desktop:\n  close_to_tray: "true"\n'), false)
  assert.equal(readCloseToTraySetting('close_to_tray: true\n'), false)
  assert.equal(readCloseToTraySetting('desktop:\n  font_family: Test\n'), false)
})

test('hides a Windows window on close and restores it from the tray', () => {
  let hidden = 0
  let shown = 0
  let focused = 0
  let prevented = 0
  const lifecycle = createTrayLifecycle({ enabled: true, isWindows: true })

  assert.equal(
    lifecycle.handleWindowClose(
      { preventDefault: () => prevented++ },
      { hide: () => hidden++, isDestroyed: () => false }
    ),
    true
  )
  assert.equal(prevented, 1)
  assert.equal(hidden, 1)

  lifecycle.restoreWindow({ show: () => shown++, focus: () => focused++, isDestroyed: () => false })
  assert.equal(shown, 1)
  assert.equal(focused, 1)
})

test('does not intercept close after explicit quit or when the setting is off', () => {
  let prevented = 0
  const window = { hide: () => assert.fail('window should close'), isDestroyed: () => false }
  const event = { preventDefault: () => prevented++ }

  const disabled = createTrayLifecycle({ enabled: false, isWindows: true })
  assert.equal(disabled.handleWindowClose(event, window), false)

  const lifecycle = createTrayLifecycle({ enabled: true, isWindows: true })
  lifecycle.requestQuit()
  assert.equal(lifecycle.handleWindowClose(event, window), false)
  assert.equal(prevented, 0)
})
