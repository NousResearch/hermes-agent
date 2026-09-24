import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { createQuitFinalization } from './quit-finalization'

/**
 * #116376: on Windows, closing the last window ran the whole JS quit path
 * (window-all-closed → app.quit → before-quit teardown → will-quit) and the
 * process still stayed resident with 0 windows. The fallback must force the
 * exit once, only on Windows, and only if Electron never reports `quit`.
 */

test('forces a single Windows exit once the admitted quit exceeds its deadline; never arms off Windows', () => {
  let onTimeout: (() => void) | undefined
  const hardExit = vi.fn()

  const finalization = createQuitFinalization({
    isWindows: true,
    schedule: callback => {
      onTimeout = callback

      return 'timer'
    },
    hardExit
  })

  finalization.arm()
  finalization.arm()
  assert.ok(onTimeout)
  onTimeout()
  onTimeout()
  assert.deepEqual(hardExit.mock.calls, [[0]])

  const schedule = vi.fn()
  const posixExit = vi.fn()
  createQuitFinalization({ isWindows: false, schedule, hardExit: posixExit }).arm()
  assert.equal(schedule.mock.calls.length, 0)
  assert.equal(posixExit.mock.calls.length, 0)
})

test('a completed quit cancels the fallback and it never re-arms', () => {
  let onTimeout: (() => void) | undefined
  const cancel = vi.fn()
  const hardExit = vi.fn()

  const schedule = vi.fn((callback: () => void) => {
    onTimeout = callback

    return 'timer'
  })

  const finalization = createQuitFinalization({ isWindows: true, schedule, cancel, hardExit })

  finalization.arm()
  finalization.cancel()
  onTimeout?.()
  finalization.arm()

  assert.deepEqual(cancel.mock.calls, [['timer']])
  assert.equal(hardExit.mock.calls.length, 0)
  assert.equal(schedule.mock.calls.length, 1)
})
