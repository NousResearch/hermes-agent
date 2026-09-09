import assert from 'node:assert/strict'
import test from 'node:test'

import { PERF_ELECTRON_ENV, PERF_ELECTRON_FLAGS } from './launch.mjs'

test('isolated perf Electron disables GPU without changing the installed desktop or system settings', () => {
  assert.ok(PERF_ELECTRON_FLAGS.includes('--disable-gpu'))
  assert.ok(PERF_ELECTRON_FLAGS.includes('--disable-background-timer-throttling'))
  assert.equal(PERF_ELECTRON_ENV.HERMES_DESKTOP_DISABLE_GPU, '1')
})
