import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { createQuitFinalization, shouldArmQuitFinalization } from './quit-finalization'

/**
 * #116376 (Windows) and macOS deferred-quit stalls: Electron can finish JS
 * before-quit teardown (or enter will-quit) and still remain resident with
 * zero windows and permanently aborted backend lifecycles. The fallback must
 * arm on Windows and macOS, stay disabled on other platforms, force the exit
 * once after its deadline, and cancel cleanly once Electron emits `quit`.
 */

test('arms quit finalization on Windows and macOS only', () => {
  assert.equal(shouldArmQuitFinalization('win32'), true)
  assert.equal(shouldArmQuitFinalization('darwin'), true)
  assert.equal(shouldArmQuitFinalization('linux'), false)
})

test('forces a single exit once the admitted quit exceeds its deadline; never arms when disabled', () => {
  let onTimeout: (() => void) | undefined
  const hardExit = vi.fn()

  const finalization = createQuitFinalization({
    enabled: true,
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
  const disabledExit = vi.fn()
  createQuitFinalization({ enabled: false, schedule, hardExit: disabledExit }).arm()
  assert.equal(schedule.mock.calls.length, 0)
  assert.equal(disabledExit.mock.calls.length, 0)
})

test('a completed quit cancels the fallback and it never re-arms', () => {
  let onTimeout: (() => void) | undefined
  const cancel = vi.fn()
  const hardExit = vi.fn()

  const schedule = vi.fn((callback: () => void) => {
    onTimeout = callback

    return 'timer'
  })

  const finalization = createQuitFinalization({ enabled: true, schedule, cancel, hardExit })

  finalization.arm()
  finalization.cancel()
  onTimeout?.()
  finalization.arm()

  assert.deepEqual(cancel.mock.calls, [['timer']])
  assert.equal(hardExit.mock.calls.length, 0)
  assert.equal(schedule.mock.calls.length, 1)
})
