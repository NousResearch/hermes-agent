import assert from 'node:assert/strict'

import { test } from 'vitest'

import { shouldIgnoreDiscoveredLocalRuntimes } from './local-runtime-ignore'

test('the flag is honoured only when set to exactly 1', () => {
  assert.equal(shouldIgnoreDiscoveredLocalRuntimes({ HERMES_DESKTOP_IGNORE_EXISTING: '1' }), true)
  assert.equal(shouldIgnoreDiscoveredLocalRuntimes({ HERMES_DESKTOP_IGNORE_EXISTING: 'true' }), false)
  assert.equal(shouldIgnoreDiscoveredLocalRuntimes({ HERMES_DESKTOP_IGNORE_EXISTING: '0' }), false)
  assert.equal(shouldIgnoreDiscoveredLocalRuntimes({}), false)
})

test('an inherited or unset environment never blocks local resolution', () => {
  // Regression guard for the bootstrap-testing path that originally motivated
  // the gate: without the flag, rung 3 (the active install at ACTIVE_HERMES_ROOT)
  // must keep resolving exactly as before.
  assert.equal(shouldIgnoreDiscoveredLocalRuntimes({}), false)
})
