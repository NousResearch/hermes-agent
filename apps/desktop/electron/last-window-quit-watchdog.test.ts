import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createLastWindowQuitWatchdog, LAST_WINDOW_QUIT_WATCHDOG_MS } from './last-window-quit-watchdog'

function harness() {
  let callback: (() => void) | null = null
  let delay: number | null = null
  let clears = 0
  let exits = 0
  let logged = 0
  let windows = false

  const watchdog = createLastWindowQuitWatchdog({
    hasWindows: () => windows,
    forceExit: () => {
      exits += 1
    },
    onForcedExit: () => {
      logged += 1
    },
    timers: {
      clearTimeout: () => {
        clears += 1
      },
      setTimeout: (fn, ms) => {
        callback = fn
        delay = ms

        return 1
      }
    }
  })

  return {
    watchdog,
    fire: () => callback?.(),
    setWindows: (value: boolean) => {
      windows = value
    },
    get clears() {
      return clears
    },
    get delay() {
      return delay
    },
    get exits() {
      return exits
    },
    get logged() {
      return logged
    }
  }
}

test('forces an exit when graceful teardown leaves the app windowless', () => {
  const state = harness()

  state.watchdog.arm()
  assert.equal(state.delay, LAST_WINDOW_QUIT_WATCHDOG_MS)
  state.fire()

  assert.equal(state.logged, 1)
  assert.equal(state.exits, 1)
})

test('does not exit when a replacement window appeared before the deadline', () => {
  const state = harness()

  state.watchdog.arm()
  state.setWindows(true)
  state.fire()

  assert.equal(state.logged, 0)
  assert.equal(state.exits, 0)
})

test('arm is idempotent and cancel disarms the watchdog', () => {
  const state = harness()

  state.watchdog.arm()
  state.watchdog.arm()
  state.watchdog.cancel()

  assert.equal(state.clears, 1)
})
